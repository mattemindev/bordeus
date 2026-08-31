"""Chunking dei Markdown curati.

## Perché non `MarkdownTextSplitter`

`MarkdownTextSplitter` (usato nella pipeline precedente) è in realtà un
`RecursiveCharacterTextSplitter` con i separatori del Markdown:
*preferisce* tagliare sui confini `#`/`##`, ma resta vincolato a
`chunk_size`. Se lo spazio residuo finisce a ridosso di un'intestazione,
la include comunque e taglia subito dopo — l'intestazione finisce in
coda al chunk precedente, separata dal contenuto che descrive, e il
chunk successivo comincia con del testo che non dice più di cosa parla.

`MarkdownHeaderTextSplitter` non ha una soglia: taglia sempre e solo sui
confini dichiarati. La categoria arriva anche come metadata pulito
(`Header 1`/`Header 2`) invece che come testo da interpretare.

Resta il caso delle sezioni che superano comunque `chunk_size` — e in
questo progetto è il caso normale, non l'eccezione: i file del
vocabolario prodotti da `extract/vocabolario.py` sono una tabella per
lettera dell'alfabeto, e `C.md` da solo supera i 5000 caratteri. Per
quelle si ricade su un `RecursiveCharacterTextSplitter`, che spezza la
sezione **solo al proprio interno**, mai unendola a quella successiva.

## Il caso tabella: una riga = un chunk

Le tabelle di questo progetto (il vocabolario oggetto -> conferimento)
sono elenchi di **record indipendenti**: "Tazzina in ceramica -> RUR"
non ha niente a che vedere con la riga sopra o sotto, se non l'iniziale.

Metterne più di una nello stesso chunk rompe il retrieval, e in modo
misurato: con ~7 voci per chunk, l'embedding è la media di sette oggetti
scorrelati e il segnale di ciascuno vale un settimo. Caso reale
osservato nei log, con la domanda "tazza per caffè, ceramica gialla con
manico":

- la risposta ("Tazzina in ceramica -> RUR") era nel chunk di `T.md` che
  va da "Tappo in sughero" a "Televisore";
- il retrieval ha invece restituito il chunk successivo dello stesso
  file ("Termometro" -> "Tovagliolo"), più due chunk contenenti "Cialda
  caffè" e "Caffettiera" — cioè ha inseguito la parola *caffè* invece
  dell'oggetto;
- il modello ha risposto correttamente che il contesto non conteneva
  l'informazione. Il prompt aveva funzionato: aveva fallito il retrieval.

Ogni riga diventa quindi un chunk a sé, reso come frase autonoma
("Tazzina in ceramica — Conferimento: RUR"): l'embedding rappresenta
esattamente un oggetto, e la ricerca per similarità torna a fare ciò per
cui è adatta.

Non si ripete più l'intestazione della tabella in ogni chunk. Serviva a
non perdere il significato delle colonne quando un chunk conteneva
righe nude, ma con una riga sola per chunk quel testo sarebbe
maggioranza del contenuto — e un'intestazione identica in tutti i chunk
è una componente costante in ogni embedding, cioè esattamente il rumore
che si sta cercando di togliere. Il nome delle colonne finisce nel testo
del chunk ("Conferimento: RUR") e nei metadata.

Il numero di chunk cresce (58 -> ~400 per quest'area), che per pgvector
non è niente, e `k` chunk recuperati diventano `k` oggetti pertinenti
invece di `k * 7` voci di cui una utile.
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger("bordeus_ingest")

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

_HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2")]

# Riga separatrice di una tabella Markdown: |---|---| , |:---|---:| ecc.
_SEPARATORE_TABELLA = re.compile(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*$")

def _celle(riga: str) -> list[str]:
    """Celle di una riga Markdown, senza le pipe esterne."""
    return [c.strip() for c in riga.strip().strip("|").split("|")]


def _tabella(testo: str) -> tuple[list[str], list[str], list[str]] | None:
    """Se il testo contiene una tabella Markdown, restituisce
    (righe_prima, intestazioni, righe_dati). None altrimenti."""
    righe = testo.splitlines()
    for i, riga in enumerate(righe):
        if i == 0 or not _SEPARATORE_TABELLA.match(riga):
            continue
        if not righe[i - 1].lstrip().startswith("|"):
            continue
        intestazioni = _celle(righe[i - 1])
        dati = [r for r in righe[i + 1 :] if r.strip().startswith("|")]
        return righe[: i - 1], intestazioni, dati
    return None


def _riga_come_frase(intestazioni: list[str], riga: str) -> str:
    """Rende una riga di tabella come testo autonomo.

    La prima colonna è il soggetto (l'oggetto da smaltire), le altre
    sono attributi: "Tazzina in ceramica — Conferimento: RUR". Il nome
    della colonna resta nel testo, così il chunk si legge da solo senza
    dover ripetere l'intestazione della tabella.
    """
    celle = _celle(riga)
    if not celle or not celle[0]:
        return ""
    soggetto = celle[0]
    attributi = [
        f"{intestazioni[i] if i < len(intestazioni) else 'Valore'}: {valore}"
        for i, valore in enumerate(celle[1:], start=1)
        if valore
    ]
    return f"{soggetto} — {'; '.join(attributi)}" if attributi else soggetto


def _chunk_per_riga(sezione: Document) -> list[Document] | None:
    """Un chunk per riga di tabella, o None se la sezione non è una
    tabella (nel qual caso si ricade sullo splitter testuale)."""
    scomposizione = _tabella(sezione.page_content)
    if scomposizione is None:
        return None

    _, intestazioni, dati = scomposizione
    chunks: list[Document] = []
    for riga in dati:
        frase = _riga_come_frase(intestazioni, riga)
        if not frase:
            continue
        metadata = dict(sezione.metadata)
        celle = _celle(riga)
        # L'oggetto e la sua categoria anche come metadata: servono a
        # ispezionare il retrieval (quale voce è stata recuperata) senza
        # dover riparsare il testo del chunk.
        metadata["oggetto"] = celle[0]
        if len(celle) > 1:
            metadata["conferimento"] = celle[1]
        chunks.append(Document(page_content=frase, metadata=metadata))
    return chunks


def _spezza_sezione_lunga(
    sezione: Document, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Spezza una sezione di prosa troppo lunga. Le tabelle non passano
    di qui: sono già state ridotte a un chunk per riga."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n"],
    )
    return splitter.split_documents([sezione])


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    if not documents:
        return []

    # strip_headers=False: il testo dell'intestazione resta anche nel
    # page_content, non solo nei metadata. Il retrieval del bot legge
    # oggi solo page_content — toglierla priverebbe il modello di
    # un'informazione che vede ancora, senza guadagnare nulla finché
    # quei metadata non sono letti da nessuna parte.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON, strip_headers=False
    )

    chunks: list[Document] = []
    for doc in documents:
        sezioni = header_splitter.split_text(doc.page_content)
        for sezione in sezioni:
            # I metadata del documento originale (area_id, comune_id,
            # tipo, source) prima, quelli dell'intestazione dopo: se le
            # chiavi coincidessero (non dovrebbe, i nomi sono diversi)
            # vince l'intestazione, più specifica del chunk risultante.
            unita = Document(
                page_content=sezione.page_content,
                metadata={**doc.metadata, **sezione.metadata},
            )
            # Le tabelle sono elenchi di record indipendenti: una riga
            # per chunk, indipendentemente dalla lunghezza della
            # sezione. Vedi il docstring del modulo.
            righe = _chunk_per_riga(unita)
            if righe is not None:
                logger.debug(
                    "%s: tabella con %d righe -> %d chunk",
                    doc.metadata.get("source", "?"),
                    len(righe),
                    len(righe),
                )
                chunks.extend(righe)
                continue

            if len(unita.page_content) <= chunk_size:
                chunks.append(unita)
                continue

            pezzi = _spezza_sezione_lunga(unita, chunk_size, chunk_overlap)
            logger.debug(
                "sezione di %d caratteri in %s spezzata in %d chunk",
                len(unita.page_content),
                doc.metadata.get("source", "?"),
                len(pezzi),
            )
            chunks.extend(pezzi)

    return chunks
