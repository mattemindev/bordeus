"""Orchestrazione: da `knowledge/<area_id>/` a Postgres.

Due destinazioni distinte, non più una sola:

1. **Vector store** (una collection per area Sub-ATO) — vocabolario,
   guide, informazioni generali. Contenuto su cui ha senso una ricerca
   per similarità semantica: l'utente descrive un oggetto a parole sue e
   il retrieval trova la voce giusta anche se non è scritta con le stesse
   parole.
2. **Tabella `raccolta_date`** — i calendari. Contenuto su cui la ricerca
   semantica è lo strumento sbagliato: "la prossima raccolta dopo oggi" è
   un confronto fra date, e l'unico modo di sbagliarlo è chiedere a un
   modello di farlo leggendo un elenco. Vedi
   `bordeus_common.calendario` e `migrations/0003_raccolta_date.sql`.

Non c'è più nessun passaggio di rete qui dentro. La pipeline precedente
scaricava pagine HTML, ne seguiva i link a PDF, indovinava una categoria
con un'euristica a parole chiave e caricava tutto: molta meccanica per
produrre chunk che andavano comunque riscritti a mano, perché il
contenuto che conta è in tabelle e in immagini che l'estrazione
automatica appiattisce. Ora la rete la tocca solo chi lancia un
estrattore (`extract/`), una volta per revisione della fonte, e il
risultato passa da una rilettura umana prima di arrivare qui.

I calendari si scoprono da `_calendari/<comune>/`, con le frazioni in
`_calendari/<comune>/_frazioni/<hamlet>/`: il percorso dice a chi si
applica il file, il manifest dice solo quale flusso locale raccoglie
quale materiale.

Ordine delle operazioni, e perché: prima il manifest e la scoperta dei
calendari (che validano tutto e falliscono senza aver scritto niente),
poi l'anagrafica (area, comuni, frazioni), poi i calendari, poi il RAG. I calendari prima del RAG perché
sono la parte che il bot può sbagliare in silenzio: se una categoria non
combacia con il vocabolario, il log del passaggio calendari lo mostra
subito, prima che l'attesa dell'embedding di tutta l'area riempia il
terminale.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from bordeus_common import db
from bordeus_common.calendario import upsert_frazione
from bordeus_common.vectorstore import add_chunks, get_vectorstore, reset_collection
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from tqdm import tqdm

from . import calendario as calendario_ingest
from . import documents, knowledge, manifest
from .chunk import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_documents

logger = logging.getLogger("bordeus_ingest")


def load_manifest(area: str) -> manifest.AreaManifest:
    """Carica il manifest di un'area indicata per id o per percorso."""
    area_dir = knowledge.resolve_area_dir(area)
    return manifest.load(area_dir / manifest.MANIFEST_FILENAME)


def registra_anagrafica(
    conn, area: manifest.AreaManifest, calendari: list[knowledge.CalendarioTrovato]
) -> None:
    """Upsert di area, comuni e frazioni. Idempotente.

    Le frazioni arrivano da due fonti che si sommano: quelle dichiarate
    in `[[frazioni]]` (perché hanno un nome leggibile o uno schema di
    raccolta proprio) e quelle dedotte dalle cartelle sotto
    `_calendari/<comune>/_frazioni/`. Una frazione che ha soltanto un
    calendario diverso non ha bisogno di comparire nel manifest: la
    cartella basta.
    """
    db.upsert_sub_ato(conn, area.id, area.nome, area.gestore)
    logger.info(
        "area registrata: %s (%r, gestore=%r)", area.id, area.nome, area.gestore
    )

    comuni_noti = {c.id for c in area.comuni}
    for comune in area.comuni:
        db.upsert_comune(conn, comune.id, comune.nome, area.id)
        logger.info("comune registrato: %s (%r)", comune.id, comune.nome)

    # Un calendario in una cartella di un comune non dichiarato è quasi
    # sempre un refuso nel nome della cartella. Fermarsi qui evita di
    # scrivere date che nessun utente raggiungerà mai.
    ignoti = {c.comune_id for c in calendari} - comuni_noti
    if ignoti:
        raise manifest.ManifestError(
            f"_calendari/ contiene cartelle per comuni non dichiarati in "
            f"[[comuni]]: {', '.join(sorted(ignoti))}. Comuni dichiarati: "
            f"{', '.join(sorted(comuni_noti))}."
        )

    frazioni: dict[tuple[str, str], str] = {
        (c.comune_id, c.hamlet): c.hamlet for c in calendari if c.hamlet
    }
    for spec in area.frazioni:
        frazioni[(spec.comune_id, spec.hamlet)] = spec.nome
    for (comune_id, hamlet), nome in sorted(frazioni.items()):
        dichiarata = area.frazione(comune_id, hamlet)
        etichetta = dichiarata.nome if dichiarata else manifest.nome_leggibile(nome)
        upsert_frazione(conn, comune_id, hamlet, etichetta)
        logger.info("  frazione registrata: %s/%s (%r)", comune_id, hamlet, etichetta)


def _log_gruppi_identici(calendari: list[knowledge.CalendarioTrovato]) -> None:
    """Raggruppa i calendari per hash del contenuto e lo mostra a log.

    I calendari stanno in una cartella per comune, quindi lo stesso
    semestre esiste in più copie quando più comuni condividono il
    calendario (caso reale: Bard, Donnas e Hône). La duplicazione è il
    prezzo di un layout in cui si trova il calendario di un comune
    guardando la sua cartella; il rischio che porta è che una copia venga
    aggiornata e le altre no, cosa che non dà nessun errore — dà un
    comune che risponde con le date del semestre scorso.

    Mostrare i gruppi rende la divergenza visibile: finché Bard, Donnas e
    Hône compaiono in un solo gruppo sono allineati, e il giorno in cui
    ne compaiono due o è voluto o è il bug. Non è un controllo (non
    sappiamo quali file *debbano* essere identici), è lo stesso dato
    letto in un modo in cui l'anomalia salta all'occhio.
    """
    per_hash: dict[str, list[str]] = {}
    for trovato in calendari:
        digest = hashlib.sha256(trovato.path.read_bytes()).hexdigest()[:8]
        per_hash.setdefault(digest, []).append(trovato.etichetta)

    condivisi = {h: d for h, d in per_hash.items() if len(d) > 1}
    if condivisi:
        logger.info("calendari con contenuto identico:")
        for digest, destinazioni in sorted(condivisi.items()):
            logger.info("  [%s] %s", digest, ", ".join(sorted(destinazioni)))


def sync_calendari(
    conn, area: manifest.AreaManifest, calendari: list[knowledge.CalendarioTrovato]
) -> int:
    """Carica in `raccolta_date` i calendari scoperti sotto
    `_calendari/`, con lo schema di raccolta dichiarato nel manifest per
    ciascuna destinazione."""
    if not calendari:
        logger.warning(
            "nessun calendario sotto %s/: il tool del bot non avrà date da "
            "restituire per quest'area",
            knowledge.CALENDARI_SUBDIR,
        )
        return 0

    _log_gruppi_identici(calendari)

    # Raggruppati per destinazione: la mappatura dei materiali è per
    # comune/frazione, le date sono per file, e la validazione ha bisogno
    # di vedere tutti i semestri insieme (vedi carica_destinazione).
    per_destinazione: dict[tuple[str, str], list[tuple[Path, str]]] = {}
    for trovato in calendari:
        per_destinazione.setdefault((trovato.comune_id, trovato.hamlet), []).append(
            (trovato.path, trovato.source)
        )

    totale = 0
    barra = tqdm(sorted(per_destinazione.items()), desc="Calendari", unit="dest")
    for (comune_id, hamlet), files in barra:
        barra.set_postfix_str(f"{comune_id}/{hamlet}" if hamlet else comune_id)
        totale += calendario_ingest.carica_destinazione(
            conn,
            comune_id,
            hamlet,
            files,
            materiali=area.mappatura_per(comune_id, hamlet),
            fonte=area.fonte_per(files[0][1]),
        )

    logger.info("calendari: %d righe scritte in raccolta_date", totale)
    return totale


def sync_rag(
    area: manifest.AreaManifest,
    database_url: str,
    embeddings: Embeddings,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = False,
) -> PGVector:
    """Scoperta -> chunking -> embedding -> vector store, per una sola
    area (una collection per area Sub-ATO: un gestore può servire più
    aree con contenuti diversi, quindi l'area è la chiave giusta, non il
    gestore)."""
    docs = documents.discover(area.area_dir, area.id, fonti=area.fonte_per)
    logger.info("%s: %d documenti trovati", area.id, len(docs))

    if not docs:
        logger.warning(
            "nessun documento RAG in %s: controlla che i Markdown siano dentro "
            "una cartella di tipo (es. guide/), non nella radice dell'area",
            area.area_dir,
        )

    per_tipo: dict[str, int] = {}
    for doc in docs:
        per_tipo[doc.metadata["tipo"]] = per_tipo.get(doc.metadata["tipo"], 0) + 1
    for tipo, quanti in sorted(per_tipo.items()):
        logger.info("  tipo=%-14s %d documenti", tipo, quanti)

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("%s: %d chunk generati", area.id, len(chunks))

    if reset:
        logger.warning(
            "--reset: svuoto la collection %s prima di riscriverla", area.id
        )
        vectorstore = reset_collection(database_url, area.id, embeddings)
    else:
        vectorstore = get_vectorstore(database_url, area.id, embeddings)
    ids = add_chunks(vectorstore, chunks)
    logger.info(
        "%s: %d chunk scritti su Postgres (collection=%s)", area.id, len(ids), area.id
    )
    return vectorstore


def run_area(
    area: manifest.AreaManifest,
    database_url: str,
    embeddings: Embeddings | None = None,
    con_rag: bool = True,
    con_calendari: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = False,
) -> PGVector | None:
    """Ingestion completa di un'area. `con_rag`/`con_calendari`
    permettono di rifare solo una delle due metà: correggere una data in
    un calendario non deve costare il ricalcolo degli embedding di tutta
    l'area (minuti su GPU Pascal), e viceversa."""
    calendari = knowledge.discover_calendari(area.area_dir)

    conn = db.connect(database_url)
    try:
        registra_anagrafica(conn, area, calendari)
        if con_calendari:
            sync_calendari(conn, area, calendari)
    finally:
        conn.close()

    if not con_rag:
        return None

    if embeddings is None:
        raise ValueError("con_rag=True richiede un oggetto Embeddings")

    return sync_rag(
        area,
        database_url,
        embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset=reset,
    )
