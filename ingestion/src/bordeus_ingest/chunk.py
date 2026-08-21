"""Step 3 della pipeline: spezza i Document in chunk con LangChain.

Due splitter diversi, a seconda del contenuto (da quando il caricamento
non passa più da Docling — vedi loaders.py — non tutto è più Markdown
strutturato, quindi non basta più un solo splitter per tutti):

- `MarkdownTextSplitter` per i Document con metadata["kind"] == "markdown"
  (file .md nativi): rispetta la struttura del documento (non spezza a
  metà un header o un blocco di codice).
- `RecursiveCharacterTextSplitter` per tutto il resto (testo estratto da
  HTML o PDF via BSHTMLLoader/PDFPlumberLoader, senza struttura
  Markdown da preservare) — con separatori limitati a `["\\n\\n", "\\n"]`
  invece del default (che include anche spazio e stringa vuota): forza
  lo splitter a spezzare solo su confini di paragrafo/riga, mai a metà
  parola o a metà riga. Configurazione testata su un PDF reale del
  progetto (un elenco di materiali/orari di raccolta): senza questo
  vincolo, lo splitter di default può tagliare a metà un elemento di
  lista o una riga con più campi.

I chunk risultanti sono ancora Document: LangChain propaga
automaticamente i metadata del documento originale (area_id,
source_url, kind, tipo) su ciascun chunk.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Solo confini di paragrafo/riga: niente spazio (" ") né stringa vuota
# (""), che nel default di RecursiveCharacterTextSplitter permettono di
# tagliare a metà parola/riga come ultima risorsa quando un blocco è
# ancora troppo grande.
_PLAIN_TEXT_SEPARATORS = ["\n\n", "\n"]


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    if not documents:
        return []

    markdown_docs = [d for d in documents if d.metadata.get("kind") == "markdown"]
    other_docs = [d for d in documents if d.metadata.get("kind") != "markdown"]

    chunks: list[Document] = []

    if markdown_docs:
        md_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks.extend(md_splitter.split_documents(markdown_docs))

    if other_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_PLAIN_TEXT_SEPARATORS,
        )
        chunks.extend(text_splitter.split_documents(other_docs))

    return chunks
