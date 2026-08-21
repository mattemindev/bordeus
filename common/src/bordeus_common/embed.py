"""Wrapper LangChain del modello di embedding Hugging Face.

Condiviso tra `bordeus_ingest` (che lo usa per embeddare i documenti in
fase di ingestion) e `bordeus_bot` (che lo usa per embeddare le query in
fase di retrieval, con `query_instruction` — vedi sotto): stesso modello,
stesso spazio vettoriale, un solo posto da aggiornare se cambia.

Espone un `langchain_core.embeddings.Embeddings`, il tipo richiesto da
`langchain-postgres` per calcolare gli embedding durante `add_documents`.
"""

from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEmbeddings

# Deve restare allineata alla dimensione del vettore usata dal resto
# della pipeline. Cambiare modello (via EMBEDDING_MODEL) con una
# dimensione diversa è supportato, ma va fatto consapevolmente: vedi il
# controllo esplicito in get_embeddings.
#
# 1024 = dimensione di microsoft/harrier-oss-v1-0.6b (confermata dalla
# model card ufficiale). Modello decoder-only con last-token pooling: a
# differenza dei modelli encoder-only usati in precedenza, le QUERY
# vanno prefissate con un'istruzione per ottenere risultati buoni (la
# FAQ del modello lo conferma esplicitamente), i DOCUMENTI no. L'ingestion
# (che embedda solo documenti/chunk) non passa mai query_instruction; il
# bot sì, per il retriever — vedi bot/src/bordeus_bot/rag.py.
EMBEDDING_DIM = 1024

DEFAULT_MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"


def get_embeddings(
    model_name: str | None = None, query_instruction: str | None = None
) -> HuggingFaceEmbeddings:
    """Costruisce l'oggetto Embeddings. Il modello è configurabile con la
    variabile d'ambiente EMBEDDING_MODEL — comodo per sperimentare con
    modelli diversi senza toccare il codice — oppure passando
    model_name esplicitamente, che ha la precedenza.

    query_instruction, se dato, viene usato SOLO per `embed_query` (non
    per `embed_documents`) tramite `query_encode_kwargs` — necessario per
    modelli come harrier-oss-v1 dove le query vanno prefissate con
    un'istruzione ma i documenti no. L'ingestion (che embedda solo
    documenti) non lo passa mai; il bot sì, per il retriever — vedi
    bot/src/bordeus_bot/rag.py.
    """
    name = model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    query_encode_kwargs = {"prompt": query_instruction} if query_instruction else {}
    embeddings = HuggingFaceEmbeddings(model_name=name, query_encode_kwargs=query_encode_kwargs)

    # Controllo esplicito della dimensione: un modello con dimensione
    # diversa da quella attesa scriverebbe silenziosamente dati
    # incompatibili nella collection Postgres (o farebbe fallire una
    # query di similarità contro chunk già scritti con un altro modello).
    # Costa una chiamata di embedding in più all'avvio, accettabile per
    # individuare l'errore subito invece che a metà ingestion.
    dim = len(embeddings.embed_query("dimensione di prova"))
    if dim != EMBEDDING_DIM:
        raise ValueError(
            f"Il modello {name!r} produce embedding a {dim} dimensioni, "
            f"ma il resto della pipeline si aspetta {EMBEDDING_DIM}. "
            "Se cambi modello consapevolmente, aggiorna EMBEDDING_DIM qui."
        )

    return embeddings
