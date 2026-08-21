# bordeus-common

Codice condiviso tra `../ingestion/` e `../bot/`: nessuno dei due
"possiede" queste responsabilità più dell'altro, per questo vivono in
un terzo membro del workspace invece che in uno dei due, con l'altro
che le importa in prestito. Non ha una CLI propria — è solo una
libreria interna, non pensata per essere usata fuori da questo
workspace.

## Cosa contiene

- **`db.py`** — schema e accesso dati condivisi: aree Sub-ATO, comuni e
  profili utente Telegram. `connect()` applica tutte le migrazioni in
  `../migrations/` (in ordine, non solo la prima) — va chiamata una
  volta sola all'avvio di un processo; `connect_light()` apre una
  connessione senza rifare quel controllo, per le query di una singola
  richiesta (es. un messaggio Telegram). L'ingestion crea/aggiorna aree
  e comuni (`upsert_sub_ato`, `upsert_comune`); il bot li legge per
  l'onboarding (`resolve_comune_by_name`, `get_sub_ato`, `get_comune`)
  e gestisce i profili utente (`get_user_profile`/`save_user_profile`,
  incluso lo stato di conferma in sospeso).
- **`embed.py`** — wrapper del modello di embedding Hugging Face
  (`get_embeddings()`), con un controllo esplicito sulla dimensione dei
  vettori prodotti (`EMBEDDING_DIM`) per evitare di scrivere
  silenziosamente dati incompatibili nel vector store se il modello
  cambia. Il parametro opzionale `query_instruction` attiva il prefisso
  istruzione richiesto dal modello per le query (non per i documenti) —
  usato solo dal bot in fase di retrieval, mai dall'ingestion.
- **`vectorstore.py`** — scrittura (`add_chunks`) e lettura
  (`get_vectorstore`) su Postgres/pgvector tramite `langchain-postgres`,
  con id di chunk deterministici (`stable_chunk_id`) per un upsert
  idempotente: ri-ingerire la stessa fonte aggiorna, non duplica. Una
  collection per **area Sub-ATO** (`collection_name = area_id`), non
  per comune — vedi `../README.md`, sezione "Aree Sub-ATO", per il
  perché.

La knowledge base RAG vera e propria (chunk con embedding) non è gestita
da `db.py`: la scrive `langchain-postgres` (`vectorstore.py`), che gestisce
da sola le proprie tabelle (`langchain_pg_collection`/
`langchain_pg_embedding`) — `db.py` si occupa solo di `sub_ato`, `comuni`
e `users`.

## GPU (NVIDIA GTX 1080)

`torch` è dichiarato qui come dipendenza **diretta** (non lasciato
transitivo via `sentence-transformers`): l'override di sorgente CUDA
dichiarato in `../pyproject.toml` si applica in modo affidabile solo
dove `torch` è dipendenza diretta di almeno un membro del workspace.
`ingestion/` e `bot/` lo ricevono transitivamente da qui, garantendo che
tutti e tre risolvano esattamente la stessa build — build diverse
potrebbero produrre embedding non identici a parità di modello. Vedi i
commenti in `pyproject.toml` (qui e alla radice del workspace) per i
dettagli.

## Struttura

```
pyproject.toml
src/bordeus_common/
├── __init__.py
├── db.py            # aree Sub-ATO, comuni, profili utente
├── embed.py         # wrapper HuggingFaceEmbeddings
└── vectorstore.py   # scrittura/lettura su Postgres via langchain-postgres
```
