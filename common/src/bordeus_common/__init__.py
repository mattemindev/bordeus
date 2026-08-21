"""bordeus_common: codice condiviso tra `bordeus_ingest` e
`bordeus_bot`.

- db.py: schema e accesso dati condivisi (comuni, users)
- embed.py: wrapper del modello di embedding Hugging Face
- vectorstore.py: scrittura/lettura su Postgres via langchain-postgres
"""

from __future__ import annotations

from . import db, embed, vectorstore

__all__ = ["db", "embed", "vectorstore"]
