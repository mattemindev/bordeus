-- Schema condiviso tra la pipeline di ingestion (ingestion/) e il bot
-- (bot/): comuni supportati e profili utente Telegram. Unica fonte di
-- verità, applicata da entrambe le parti allo stesso modo (stessa
-- tabella schema_migrations per il tracking, vedi i rispettivi db.py).
--
-- La knowledge base RAG (documenti + chunk con embedding) non è più
-- gestita da queste tabelle: da quando l'ingestion scrive tramite
-- l'integrazione langchain-postgres, quello storage vive nelle tabelle
-- che quella libreria gestisce da sola (langchain_pg_collection,
-- langchain_pg_embedding), create automaticamente al primo utilizzo di
-- PGVector — non c'è più bisogno di dichiararle qui.

CREATE TABLE IF NOT EXISTS comuni (
    id              TEXT PRIMARY KEY,
    nome            TEXT NOT NULL,
    gestore         TEXT NOT NULL DEFAULT '',
    source_base_url TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    chat_id     BIGINT PRIMARY KEY,
    comune_id   TEXT REFERENCES comuni (id) ON DELETE SET NULL,
    onboarded   BOOLEAN NOT NULL DEFAULT false,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
