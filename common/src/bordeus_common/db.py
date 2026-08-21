"""Schema e accesso dati condivisi tra `ingestion/` e `bot/`: aree
Sub-ATO, comuni e profili utente Telegram (migrazioni in `migrations/`,
alla radice del repo — applicate tutte, in ordine, non solo la prima).

Nessuno dei due componenti applicativi "possiede" questo schema più
dell'altro — l'ingestion crea/aggiorna Sub-ATO e comuni (`upsert_sub_ato`,
`upsert_comune`), il bot li legge per l'onboarding
(`resolve_comune_by_name`, `get_sub_ato`) e gestisce i profili utente
(`get_user_profile`/`save_user_profile`) — per questo vive in un terzo
membro del workspace (`bordeus_common`) invece che in uno dei due, con
l'altro che lo importa in prestito.

Gli embedding sono raggruppati per **Sub-ATO** (`sub_ato_id`), non per
comune: un gestore può servire più aree con contenuti diversi (es.
Quendoz, Valle d'Aosta, gestisce sia il Sub-ATO C sia il D con pagine
guida separate) — l'area è la chiave giusta, il comune eredita quella
del Sub-ATO a cui appartiene, non ha un gestore "suo".

La knowledge base RAG (chunk con embedding) non passa da qui: la scrive
`langchain-postgres`, che gestisce da sola le proprie tabelle
(`langchain_pg_collection`/`langchain_pg_embedding`) — vedi
`bordeus_ingest.vectorstore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psycopg

# Risale da common/src/bordeus_common/db.py alla radice del repo, poi
# scende in migrations/. Se common/ viene spostata fuori dal repo
# bordeus, va aggiornato (o reso configurabile) questo percorso.
_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def connect(database_url: str) -> psycopg.Connection:
    """Connessione che applica le migrazioni condivise non ancora
    presenti (tutte, in ordine — non solo la prima). Va usata all'avvio
    di un processo (una volta sola); per le query di una singola
    richiesta (es. un messaggio Telegram) usare invece `connect_light`,
    che salta il controllo schema."""
    conn = psycopg.connect(database_url, autocommit=True)
    _ensure_schema(conn)
    return conn


def connect_light(database_url: str) -> psycopg.Connection:
    """Connessione senza controllo schema — per query di una singola
    richiesta in un processo lungo (es. il bot), dopo che `connect` è già
    stata chiamata una volta all'avvio."""
    return psycopg.connect(database_url, autocommit=True)


def _ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    if not _MIGRATIONS_DIR.exists():
        raise FileNotFoundError(
            f"Cartella migrations non trovata in {_MIGRATIONS_DIR}. "
            "common/ deve restare dentro il repo bordeus, come cartella "
            "sorella di migrations/, oppure va reso configurabile questo "
            "percorso."
        )

    # In ordine alfabetico (quindi numerico, dato il prefisso a 4 cifre):
    # 0001 prima di 0002, ecc. Ogni file già applicato (tracciato per
    # nome in schema_migrations) viene saltato, non ri-eseguito.
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        _apply_migration(conn, path)


def _apply_migration(conn: psycopg.Connection, path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE filename = %s)",
            (path.name,),
        )
        (already_applied,) = cur.fetchone()
        if already_applied:
            return

        cur.execute(path.read_text())
        cur.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)",
            (path.name,),
        )


# --- sub_ato ----------------------------------------------------------------


@dataclass
class SubAto:
    id: str
    nome: str
    gestore: str


def upsert_sub_ato(conn: psycopg.Connection, sub_ato_id: str, nome: str, gestore: str) -> None:
    """Crea o aggiorna una riga di `sub_ato` (`ON CONFLICT (id) DO
    UPDATE`). Usata dalla pipeline di ingestion."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sub_ato (id, nome, gestore)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                nome    = EXCLUDED.nome,
                gestore = EXCLUDED.gestore
            """,
            (sub_ato_id, nome, gestore),
        )


def get_sub_ato(conn: psycopg.Connection, sub_ato_id: str) -> SubAto | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id, nome, gestore FROM sub_ato WHERE id = %s", (sub_ato_id,))
        row = cur.fetchone()
    if row is None:
        return None
    id_value, nome, gestore = row
    return SubAto(id=id_value, nome=nome, gestore=gestore)


# --- comuni -------------------------------------------------------------


@dataclass
class Comune:
    id: str
    nome: str
    sub_ato_id: str | None


def comune_exists(conn: psycopg.Connection, comune_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS(SELECT 1 FROM comuni WHERE id = %s)", (comune_id,))
        (exists,) = cur.fetchone()
        return exists


def get_comune(conn: psycopg.Connection, comune_id: str) -> Comune | None:
    """Cerca un comune per id esatto (non per nome/euristica — quella
    è resolve_comune_by_name). Usata dal bot per risalire dal comune già
    confermato dall'utente al suo Sub-ATO, es. per scegliere la
    collection giusta in fase di retrieval."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, nome, sub_ato_id FROM comuni WHERE id = %s", (comune_id,))
        row = cur.fetchone()
    if row is None:
        return None
    found_id, nome, sub_ato_id = row
    return Comune(id=found_id, nome=nome, sub_ato_id=sub_ato_id)


def upsert_comune(conn: psycopg.Connection, comune_id: str, nome: str, sub_ato_id: str) -> None:
    """Crea o aggiorna una riga di `comuni` (`ON CONFLICT (id) DO
    UPDATE`): ri-ingerire lo stesso Sub-ATO per un comune già noto
    aggiorna nome/sub_ato_id invece di fallire o duplicare. Usata dalla
    pipeline di ingestion (`bordeus_ingest.pipeline.run_sub_ato`)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO comuni (id, nome, sub_ato_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                nome         = EXCLUDED.nome,
                sub_ato_id = EXCLUDED.sub_ato_id
            """,
            (comune_id, nome, sub_ato_id),
        )


# Normalizzazione tollerante ad accenti/maiuscole per il confronto dei
# nomi comune (da reverse geocoding o digitati dall'utente).
_ACCENTS = str.maketrans("àáèéìíòóùú", "aaeeiioouu")


def _normalize(s: str) -> str:
    return s.strip().lower().translate(_ACCENTS)


def resolve_comune_by_name(conn: psycopg.Connection, name: str) -> Comune | None:
    """Cerca un comune supportato il cui nome corrisponda (in modo
    tollerante ad accenti/maiuscole) al nome fornito. Usata dal bot in
    fase di onboarding."""
    target = _normalize(name)
    with conn.cursor() as cur:
        cur.execute("SELECT id, nome, sub_ato_id FROM comuni")
        for found_id, nome, sub_ato_id in cur.fetchall():
            if _normalize(nome) == target:
                return Comune(id=found_id, nome=nome, sub_ato_id=sub_ato_id)
    return None


# --- users ----------------------------------------------------------------


@dataclass
class UserProfile:
    chat_id: int
    comune_id: str | None
    onboarded: bool
    # Comune risolto ma non ancora confermato esplicitamente dall'utente
    # (vedi telegram_bot.py: dopo aver riconosciuto un comune, il bot
    # chiede conferma del Sub-ATO/gestore prima di completare
    # l'onboarding). None quando non c'è nessuna conferma in sospeso.
    pending_comune_id: str | None = None


def get_user_profile(conn: psycopg.Connection, chat_id: int) -> UserProfile | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chat_id, comune_id, onboarded, pending_comune_id "
            "FROM users WHERE chat_id = %s",
            (chat_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    chat_id_value, comune_id, onboarded, pending_comune_id = row
    return UserProfile(
        chat_id=chat_id_value,
        comune_id=comune_id,
        onboarded=onboarded,
        pending_comune_id=pending_comune_id,
    )


def save_user_profile(conn: psycopg.Connection, profile: UserProfile) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (chat_id, comune_id, onboarded, pending_comune_id, updated_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (chat_id) DO UPDATE SET
                comune_id         = EXCLUDED.comune_id,
                onboarded           = EXCLUDED.onboarded,
                pending_comune_id = EXCLUDED.pending_comune_id,
                updated_at          = now()
            """,
            (profile.chat_id, profile.comune_id, profile.onboarded, profile.pending_comune_id),
        )
