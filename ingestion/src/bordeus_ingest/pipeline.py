"""Orchestrazione della pipeline di ingestion.

Gli embedding sono raggruppati per **area Sub-ATO**, non per singolo
comune: un gestore può servire più aree con contenuti diversi (es.
Quendoz, Valle d'Aosta, gestisce sia il Sub-ATO C sia il D con pagine
guida separate), quindi l'area — non il gestore — è la chiave giusta.
Un'area può avere più di una fonte da ingerire (es. una pagina specifica
dell'area + un vocabolario/glossario condiviso da tutte le aree dello
stesso gestore): fetch_to_knowledge accetta una lista di URL, non uno
solo.

Non tutto il contenuto di un'area è però davvero condiviso: il
calendario di raccolta porta a porta, ad esempio, può variare da un
comune all'altro della stessa area per motivi logistici, anche sotto lo
stesso gestore (caso reale: TeknoService Italia, Sub-ATO E, Donnas e
Pont-Saint-Martin — comuni confinanti — hanno calendari diversi).
`run_sub_ato` accetta quindi anche `comune_urls`, fonti aggiuntive
specifiche di un singolo comune: i chunk risultanti restano nella stessa
collection dell'area (non una collection separata per comune — vedi
`bordeus_common.vectorstore`), ma taggati con `comune_id` nel
metadata, così il bot può filtrare in fase di retrieval (contenuto
condiviso dell'area PIÙ quello specifico del comune dell'utente, mai
quello di un comune vicino). Nessun vincolo su quali categorie possano
essere specifiche di un comune: lo decide chi lancia l'ingestion,
caso per caso, in base a come il gestore reale organizza i contenuti.

Passaggi:

0. db.upsert_sub_ato + db.upsert_comune — registra l'area e i comuni
   che vi appartengono (non serve che esistano già)
1. fetch_source (per ogni URL, area-wide o per-comune) + save_to_knowledge
   — un giro di rete per URL, scritto su disco
   (knowledge/<area_id>/<categoria>/ o
   knowledge/<area_id>/_comuni/<comune_id>/<categoria>/)
2. loaders.load_knowledge — carica con i loader di LangChain
3. chunk.split_documents — chunking con i text splitter di LangChain
4. vectorstore.add_chunks — embedding + scrittura su Postgres (langchain-postgres)

Un PDF o Markdown che fallisce il fetch non blocca gli altri (solo
loggato) — stessa filosofia della versione precedente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from tqdm import tqdm

from bordeus_common import db
from bordeus_common.vectorstore import add_chunks, get_vectorstore

from . import classify, knowledge
from .chunk import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_documents
from .fetch import fetch_markdown_text, fetch_page, fetch_pdf_bytes, pdf_text_preview
from .loaders import load_knowledge

logger = logging.getLogger("bordeus_ingest")


@dataclass
class ComuneInput:
    """Un comune coperto dall'area Sub-ATO da ingerire: giusto quello
    che serve per l'upsert in tabella `comuni` (id + nome) — l'area di
    appartenenza è condivisa da tutti i comuni della stessa ingestion,
    vedi run_sub_ato."""

    id: str
    nome: str


@dataclass
class FetchedDoc:
    """Un documento scaricato, ancora in memoria: url, categoria stimata
    e contenuto grezzo (bytes per i PDF, str per HTML/Markdown).

    comune_id vuoto (default) = contenuto condiviso dall'area; se
    valorizzato, il documento è specifico di quel comune (es. un
    calendario di raccolta) e viene taggato di conseguenza fino al
    chunk finale nel vector store."""

    url: str
    kind: str  # "html" | "pdf" | "markdown"
    tipo: str
    content: bytes | str
    comune_id: str = ""


_EXT_BY_KIND = {"html": ".html", "pdf": ".pdf", "markdown": ".md"}


def fetch_source(source_url: str, comune_id: str = "") -> list[FetchedDoc]:
    """Un solo giro di rete: scarica la fonte data (pagina HTML, con
    scoperta dei suoi PDF/Markdown linkati — oppure un PDF o Markdown
    autonomo, se l'URL stesso punta direttamente a uno di questi, senza
    nessuna pagina HTML che lo linki: caso reale, alcuni gestori
    pubblicano calendari come PDF a sé stanti, non dentro una pagina),
    classificandoli per categoria, e li tiene in memoria — nessuna
    scrittura su disco qui (vedi save_to_knowledge).

    comune_id, se dato, marca l'intera fonte (più eventuali link
    scoperti, se la fonte è una pagina HTML) come specifica di quel
    comune, non condivisa dall'area — vedi il docstring del modulo."""
    suffix = source_url.split("?", 1)[0].split("#", 1)[0].lower()

    if suffix.endswith(".pdf"):
        content = fetch_pdf_bytes(source_url)
        preview = pdf_text_preview(content)
        tipo = classify.guess_doc_type(url=source_url, text_sample=preview)
        logger.info(
            "scaricato PDF autonomo %s (tipo=%s%s)",
            source_url,
            tipo,
            f", comune={comune_id}" if comune_id else "",
        )
        return [FetchedDoc(url=source_url, kind="pdf", tipo=tipo, content=content, comune_id=comune_id)]

    if suffix.endswith((".md", ".markdown")):
        text = fetch_markdown_text(source_url)
        tipo = classify.guess_doc_type(url=source_url, text_sample=text)
        logger.info(
            "scaricato Markdown autonomo %s (tipo=%s%s)",
            source_url,
            tipo,
            f", comune={comune_id}" if comune_id else "",
        )
        return [FetchedDoc(url=source_url, kind="markdown", tipo=tipo, content=text, comune_id=comune_id)]

    page = fetch_page(source_url)
    docs: list[FetchedDoc] = []

    tipo = classify.guess_doc_type(url=source_url, title=page.title, text_sample=page.text_sample)
    docs.append(
        FetchedDoc(url=source_url, kind="html", tipo=tipo, content=page.raw_html, comune_id=comune_id)
    )
    logger.info(
        "scaricata pagina HTML %s (tipo=%s%s)",
        source_url,
        tipo,
        f", comune={comune_id}" if comune_id else "",
    )

    pdf_bar = tqdm(page.pdf_links, desc="PDF", unit="doc")
    for pdf_url in pdf_bar:
        pdf_bar.set_postfix_str(knowledge.short(pdf_url))
        try:
            content = fetch_pdf_bytes(pdf_url)
        except Exception as exc:
            logger.warning("fetch PDF fallito %s: %s (salto e proseguo)", pdf_url, exc)
            continue
        # Anteprima leggera (poche pagine, pdfplumber) per classificare sul
        # contenuto reale invece che solo sul nome file — spesso un
        # codice criptico (es. "FRAZ-BARD-DONNAS.pdf") da cui è
        # impossibile indovinare la categoria. Non è la vera estrazione
        # per il RAG (quella resta compito di PDFPlumberLoader in
        # loaders.py), solo un campione per orientare la classificazione.
        preview = pdf_text_preview(content)
        tipo = classify.guess_doc_type(url=pdf_url, text_sample=preview)
        docs.append(FetchedDoc(url=pdf_url, kind="pdf", tipo=tipo, content=content, comune_id=comune_id))

    md_bar = tqdm(page.markdown_links, desc="Markdown", unit="doc")
    for md_url in md_bar:
        md_bar.set_postfix_str(knowledge.short(md_url))
        try:
            text = fetch_markdown_text(md_url)
        except Exception as exc:
            logger.warning("fetch Markdown fallito %s: %s (salto e proseguo)", md_url, exc)
            continue
        tipo = classify.guess_doc_type(url=md_url, text_sample=text)
        docs.append(FetchedDoc(url=md_url, kind="markdown", tipo=tipo, content=text, comune_id=comune_id))

    return docs


def save_to_knowledge(
    area_id: str, docs: list[FetchedDoc], manifest: dict[str, dict] | None = None
) -> dict[str, dict]:
    """Scrive su disco i documenti già scaricati da fetch_source —
    condiviso dall'area (knowledge/<area_id>/<categoria>/) o
    specifico di un comune
    (knowledge/<area_id>/_comuni/<comune_id>/<categoria>/), a
    seconda di doc.comune_id — aggiornando (in-place) il manifest
    passato, o uno nuovo caricato da disco se non fornito. Non lo
    persiste da sola: restituisce il manifest aggiornato, così
    `fetch_to_knowledge` può accumulare più chiamate (una per URL) prima
    di salvare una volta sola su disco."""
    if manifest is None:
        manifest = knowledge.load_manifest(area_id)

    for doc in docs:
        filename = knowledge.sanitize_filename(doc.url, _EXT_BY_KIND[doc.kind])
        file_path = knowledge.target_path(area_id, doc.tipo, filename, comune_id=doc.comune_id)
        if isinstance(doc.content, bytes):
            file_path.write_bytes(doc.content)
        else:
            file_path.write_text(doc.content, encoding="utf-8")
        knowledge.register_file(
            area_id, manifest, file_path, doc.url, doc.kind, doc.tipo, comune_id=doc.comune_id
        )

    return manifest


def fetch_to_knowledge(
    area_id: str, source_urls: list[str], comune_urls: dict[str, list[str]] | None = None
) -> int:
    """Fetch di una o più fonti condivise dall'area (`source_urls`, un
    giro di rete per URL) più, opzionalmente, fonti specifiche di
    singoli comuni (`comune_urls`: comune_id -> lista di URL — es. il
    calendario di raccolta di un comune, diverso da quello di un comune
    vicino nella stessa area). Tutto finisce nella stessa cartella
    knowledge/<area_id>/ (con o senza sottocartella _comuni/, a
    seconda della fonte) e nello stesso manifest. Restituisce il numero
    totale di file salvati."""
    manifest: dict[str, dict] = {}
    total = 0

    for url in source_urls:
        docs = fetch_source(url)
        manifest = save_to_knowledge(area_id, docs, manifest=manifest)
        total += len(docs)

    for comune_id, urls in (comune_urls or {}).items():
        for url in urls:
            docs = fetch_source(url, comune_id=comune_id)
            manifest = save_to_knowledge(area_id, docs, manifest=manifest)
            total += len(docs)

    knowledge.save_manifest(area_id, manifest)
    logger.info("fetch completato: %d file salvati in knowledge/%s/", total, area_id)
    return total


def run_sub_ato(
    sub_ato_id: str,
    sub_ato_nome: str,
    gestore: str,
    comuni: list[ComuneInput],
    source_urls: list[str],
    database_url: str,
    embeddings: Embeddings,
    comune_urls: dict[str, list[str]] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> PGVector:
    """Esegue la pipeline end-to-end per un'area Sub-ATO:

    0. INSERT/UPSERT dell'area e di ciascun comune che vi appartiene
       (upsert, non serve che esistano già — vedi db.upsert_sub_ato,
       db.upsert_comune)
    1-4. fetch (fonti condivise dall'area + eventuali fonti specifiche
         di un comune, vedi comune_urls) -> knowledge/<sub_ato_id>/ ->
         load -> split -> scrittura su Postgres (collection isolata per
         area, chunk taggati con comune_id quando pertinente)

    Restituisce il vector store dell'area, pronto per essere interrogato
    (vedi bot/rag.py per il filtro per comune in fase di retrieval, o il
    notebook per un esempio diretto con `.similarity_search()`)."""
    conn = db.connect(database_url)

    db.upsert_sub_ato(conn, sub_ato_id, sub_ato_nome, gestore)
    logger.info("area registrata in Postgres: %s (%r, gestore=%r)", sub_ato_id, sub_ato_nome, gestore)

    for c in comuni:
        db.upsert_comune(conn, c.id, c.nome, sub_ato_id)
        logger.info("comune registrato in Postgres: %s (%r) -> %s", c.id, c.nome, sub_ato_id)

    fetch_to_knowledge(sub_ato_id, source_urls, comune_urls=comune_urls)

    documents = load_knowledge(sub_ato_id)
    logger.info("%s: caricati %d documenti da knowledge/%s/", sub_ato_id, len(documents), sub_ato_id)

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("%s: %d chunk generati", sub_ato_id, len(chunks))

    vectorstore = get_vectorstore(database_url, sub_ato_id, embeddings)
    ids = add_chunks(vectorstore, chunks)
    logger.info("%s: %d chunk scritti su Postgres (collection=%s)", sub_ato_id, len(ids), sub_ato_id)

    return vectorstore
