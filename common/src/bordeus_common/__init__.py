"""bordeus_common: codice condiviso tra `bordeus_ingest` e
`bordeus_bot`.

- db.py: schema e accesso dati condivisi (sub-ATO, comuni, users)
- log.py: livello TRACE (5) e configurazione dei log condivisa
- calendario.py: date di raccolta porta a porta (tabella `raccolta_date`,
  fuori dal vector store — scritta dall'ingestion, letta dal tool del bot)
- embed.py: wrapper del modello di embedding Hugging Face
- vectorstore.py: scrittura/lettura su Postgres via langchain-postgres
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["calendario", "db", "embed", "log", "vectorstore"]

# Import pigri: `embed` tira dentro torch e sentence-transformers, che
# valgono secondi di avvio e diversi GB installati. Importarli qui
# significava che anche `from bordeus_common.log import get_logger` —
# poche righe di logging — li caricava, e che i test non potevano girare
# senza le dipendenze pesanti del progetto.
#
# `__getattr__` a livello di modulo (PEP 562) mantiene
# `bordeus_common.embed` funzionante per chi lo usa davvero, caricandolo
# al primo accesso invece che all'import del pacchetto.
if TYPE_CHECKING:  # pragma: no cover — solo per gli analizzatori statici
    from . import calendario, db, embed, log, vectorstore


def __getattr__(name: str):
    if name in __all__:
        modulo = importlib.import_module(f".{name}", __name__)
        globals()[name] = modulo
        return modulo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
