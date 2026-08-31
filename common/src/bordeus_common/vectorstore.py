"""Scrittura e lettura su Postgres/pgvector tramite l'integrazione
ufficiale `langchain-postgres`, condivise tra `bordeus_ingest` (scrive
i chunk — `add_chunks`) e `bordeus_bot` (legge via retriever —
`get_vectorstore`).

`langchain-postgres` gestisce le proprie tabelle
(`langchain_pg_collection`, `langchain_pg_embedding`), con una
"collection" per **area Sub-ATO** (`collection_name = area_id`) e i
metadati (source_url, kind, tipo, area_id, comune_id) in una colonna
JSONB — non più per comune: un'area può coprire più comuni che
condividono le stesse guide (vedi migrations/0002_sub_ato.sql).

`comune_id` è vuoto per il contenuto condiviso dall'area, valorizzato
per contenuto specifico di un comune (es. un calendario di raccolta, che
può variare da un comune all'altro della stessa area per motivi
logistici — vedi `bordeus_ingest.pipeline`) — resta comunque nella
stessa collection dell'area, non una collection separata per comune: il
bot filtra per comune_id in fase di query (`bot/rag.py`), non
serve interrogare due collection diverse e fonderne i risultati.

Un solo schema, un solo linguaggio, niente da tenere allineato tra due
implementazioni diverse.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

# Nomi di tabella reali di langchain-postgres (per riferimento, usati
# solo in viz.py per la lettura diretta a scopo di visualizzazione —
# PGVector stesso non li espone come costanti pubbliche).
COLLECTION_TABLE = "langchain_pg_collection"
EMBEDDING_TABLE = "langchain_pg_embedding"


def to_sqlalchemy_url(database_url: str) -> str:
    """`langchain-postgres` si aspetta un URL in stile SQLAlchemy
    (`postgresql+psycopg://...`), non lo schema `postgres://` usato nel
    resto del progetto (`bordeus_common.db`). Conversione automatica
    invece di dover mantenere due variabili d'ambiente diverse per la
    stessa connessione."""
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg://" + database_url[len(prefix) :]
    raise ValueError(f"DATABASE_URL non riconosciuto: {database_url!r}")


def get_vectorstore(
    database_url: str, area_id: str, embeddings: Embeddings
) -> PGVector:
    """Una collection per area Sub-ATO (`collection_name = area_id`):
    tiene i dati di aree diverse logicamente separati all'interno delle
    stesse tabelle condivise di `langchain-postgres`, anche quando più
    aree condividono lo stesso gestore (es. Quendoz gestisce sia il
    Sub-ATO C sia il D, con contenuti diversi)."""
    return PGVector(
        embeddings=embeddings,
        collection_name=area_id,
        connection=to_sqlalchemy_url(database_url),
        use_jsonb=True,
    )


def stable_chunk_id(
    area_id: str, source_url: str, chunk_index: int, content: str
) -> str:
    """ID deterministico per un chunk: stessa fonte + stesso indice +
    stesso contenuto -> stesso id, così ri-lanciare la pipeline
    sullo stesso URL aggiorna (upsert) i chunk esistenti invece di
    duplicarli — `langchain-postgres` fa `ON CONFLICT (id) DO UPDATE`
    quando gli id sono espliciti (verificato leggendo il sorgente di
    `PGVector.add_embeddings`).

    Limite noto: se il chunking cambia (es. cambi chunk_size) i chunk
    prodotti in precedenza per la stessa fonte, con id diversi perché il
    contenuto è diverso, restano orfani invece di essere sovrascritti.
    Ripulirli richiederebbe un `pre_delete_collection` prima di ogni
    re-ingestion completa — non implementato in questo PoC."""
    raw = f"{area_id}:{source_url}:{chunk_index}:{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def add_chunks(vectorstore: PGVector, chunks: list[Document]) -> list[str]:
    """Scrive i chunk nel vector store, calcolando id stabili per un
    upsert idempotente. L'indice usato nell'id è per-sorgente (non
    globale sulla lista): due chunk dello stesso documento hanno indici
    progressivi 0,1,2..., così l'id resta stabile anche se cambia
    l'ordine con cui i documenti di aree/fonti diverse vengono passati a
    questa funzione."""
    if not chunks:
        # Lista vuota (es. tutti i documenti hanno fallito il
        # caricamento): PGVector.add_documents con una lista vuota tenta
        # comunque un INSERT e fallisce con una violazione di vincolo
        # (verificato) — meglio uscire subito con un no-op esplicito.
        return []

    counters: dict[str, int] = {}
    ids: list[str] = []
    for chunk in chunks:
        area_id = chunk.metadata.get("area_id", "")
        # "source" è la chiave che l'ingestion scrive (percorso relativo
        # all'area). "source_url" è il nome che usava la pipeline a
        # crawl: resta come ripiego per non rompere una collection
        # scritta da quella versione, ma non dovrebbe più comparire.
        # Senza questo, ogni chunk ricadeva sulla stessa chiave vuota e
        # l'indice diventava globale sulla lista: bastava aggiungere una
        # voce a un file per far slittare l'indice di tutti i chunk
        # successivi e cambiarne l'id, lasciando orfana metà collection.
        source = chunk.metadata.get("source") or chunk.metadata.get("source_url", "")
        idx = counters.get(source, 0)
        counters[source] = idx + 1
        ids.append(stable_chunk_id(area_id, source, idx, chunk.page_content))

    return vectorstore.add_documents(chunks, ids=ids)


def reset_collection(database_url: str, area_id: str, embeddings) -> PGVector:
    """Svuota la collection di un'area e la ricrea.

    Serve perché `stable_chunk_id` include il contenuto del chunk:
    l'upsert aggiorna i chunk che riesce a ricalcolare uguali, ma non
    cancella quelli che l'ingestion non produce più. Finché la strategia
    di chunking resta la stessa il problema non si vede; quando cambia
    (o quando una voce viene tolta da un Markdown), i vecchi chunk
    restano nel vector store e continuano a competere nel retrieval —
    con il risultato che una correzione ai dati sembra non avere
    effetto, o peggio ne ha metà.

    Non è il comportamento predefinito perché un `sync` normale deve
    poter aggiornare senza buttare via tutto: è esplicito
    (`sync --reset`), da usare dopo un cambio di chunking o quando si
    vuole la certezza che la collection rispecchi esattamente i file.
    """
    vectorstore = get_vectorstore(database_url, area_id, embeddings)
    vectorstore.delete_collection()
    return get_vectorstore(database_url, area_id, embeddings)
