"""Step 2 della pipeline: carica i file salvati in knowledge/<area_id>/
come Document di LangChain.

- .html/.htm -> `BSHTMLLoader` (BeautifulSoup, stesso parser già usato in
  fetch.py per la scoperta dei link — nessuna dipendenza aggiuntiva).
- .pdf -> `PDFPlumberLoader` (un Document per pagina).
- .md -> `TextLoader` (testo grezzo: la struttura Markdown la interpreta
  lo splitter dedicato nello step successivo, non il loader).

Non usiamo Docling (`langchain-docling`, provato in una versione
precedente di questa pipeline): la sua integrazione richiede
incondizionatamente `torch`+`torchvision`+`docling-ibm-models` anche
solo per l'import — verificato provando backend PDF più leggeri, non
evitabile con l'architettura attuale di `docling.document_converter`.
Nel nostro caso concreto ha causato più problemi (spazio disco esaurito,
installazioni interrotte a metà, nello stesso ambiente in cui abbiamo
sviluppato) di quanti ne risolvesse. `BSHTMLLoader`/`PDFPlumberLoader`
producono testo piatto, non Markdown strutturato (le tabelle diventano
righe di testo su più righe) — per le guide dei gestori, PDF generati
digitalmente e non scansioni, è comunque sufficiente. Vedi chunk.py: con
testo piatto invece che Markdown uniforme, servono due splitter diversi,
non più uno solo.

Ogni Document viene arricchito con i metadati registrati nel manifest al
momento del fetch (source_url, kind, tipo, area_id, comune_id —
vuoto per contenuto condiviso dall'area, valorizzato per contenuto
specifico di un comune, es. un calendario di raccolta) — servono più
avanti per il filtro per area/comune nel vector store (vedi
`bordeus_common.vectorstore`, `bot/rag.py`) e per l'attribuzione delle
fonti.
"""

from __future__ import annotations

import logging

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PDFPlumberLoader,
    TextLoader,
)
from langchain_core.documents import Document

from . import knowledge

logger = logging.getLogger("bordeus_ingest")

_LOADER_BY_SUFFIX = {
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".pdf": PDFPlumberLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
}


def load_knowledge(area_id: str) -> list[Document]:
    """Carica tutti i file registrati nel manifest di un'area. Un file
    presente nel manifest ma mancante su disco (es. cancellato a mano)
    viene saltato con un warning, non blocca il resto del caricamento —
    stessa filosofia "un elemento rotto non ferma la pipeline" usata
    altrove."""
    base_dir = knowledge.area_dir(area_id)
    manifest = knowledge.load_manifest(area_id)

    documents: list[Document] = []
    for rel_path, entry in manifest.items():
        file_path = base_dir / rel_path
        if not file_path.exists():
            logger.warning(
                "file nel manifest ma assente su disco: %s (salto)", file_path
            )
            continue

        loader_cls = _LOADER_BY_SUFFIX.get(file_path.suffix.lower())
        if loader_cls is None:
            logger.warning(
                "nessun loader per l'estensione %s (salto %s)",
                file_path.suffix,
                file_path,
            )
            continue

        loader_kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
        loader = loader_cls(str(file_path), **loader_kwargs)

        try:
            loaded = loader.load()
        except Exception as exc:
            logger.warning("caricamento fallito per %s: %s (salto)", file_path, exc)
            continue

        for doc in loaded:
            doc.metadata.update(
                {
                    "area_id": area_id,
                    "source_url": entry["source_url"],
                    "kind": entry["kind"],
                    "tipo": entry["tipo"],
                    "comune_id": entry.get("comune_id", ""),
                }
            )
            documents.append(doc)

    return documents
