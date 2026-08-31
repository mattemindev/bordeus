"""Accesso alla tabella `raccolta_date`: date di raccolta porta a porta.

Condiviso tra `bordeus_ingest` (scrive: `replace_calendario`) e
`bordeus_bot` (legge: `prossima_raccolta`, `categorie_disponibili`) —
stesso motivo per cui `db.py` vive qui e non in uno dei due pacchetti.

Il calendario NON passa più dal vector store. Trovare "la prossima
raccolta dell'organico dopo oggi" è un lookup esatto su un insieme
ordinato di date: come chunk di testo nel RAG diventa un elenco di ~50
date che il modello deve leggere e su cui deve fare aritmetica, cosa in
cui gli LLM sbagliano in modo silenzioso (una data già passata, un
errore di un giorno). Qui è una query indicizzata, e il modello riceve
una sola data già corretta tramite tool calling (vedi
`bordeus_bot.calendario`). Vedi `migrations/0003_raccolta_date.sql`.

## Frazioni (hamlet)

Un calendario vale tipicamente per più comuni; una frazione può però
averne uno diverso dal proprio comune per motivi logistici. La
convenzione è la stringa vuota (non NULL) per "vale per l'intero
comune", coerente con `comune_id` vuoto = "condiviso dall'area" nei
metadata dei chunk.

La risoluzione è a **due livelli con fallback**
(`_hamlet_effettivo`): se esiste anche una sola riga per quella
frazione, la frazione ha un calendario proprio e si usa quello per
intero; altrimenti si ricade sul calendario del comune. Il controllo è
volutamente per frazione e non per (frazione, categoria): un override
reale è sempre un calendario completo, non una singola categoria
spostata di giorno. Se una frazione avesse un override parziale,
mescolare le due fonti per categoria darebbe una risposta più difficile
da verificare per chi cura i dati che non un override dichiarato
interamente.

## Materiale vs categoria

Lo schema di raccolta varia per comune E per frazione. Caso reale:
Bard, Donnas e Hône raccolgono carta e cartone insieme, ma le frazioni
di Bard e Donnas li raccolgono separati, in due giorni diversi.

Il vocabolario è però condiviso dall'intera area: non può nominare la
categoria locale, perché quale sia dipende da chi sta chiedendo. Registra
quindi il **materiale** (carta, cartone, vetro...), che è intrinseco
all'oggetto e uguale ovunque; la traduzione materiale -> categoria
locale avviene qui, con `risolvi_categoria`, sulla base della tabella
`raccolta_materiale` popolata dall'ingestion.

Il punto non è solo di pulizia: senza questa separazione la scelta fra
"carta" e "cartone" ricadrebbe sul modello, che ha in mano solo la
descrizione dell'oggetto e, sbagliando, restituirebbe comunque una data
plausibile. Vedi `migrations/0004_raccolta_materiale.sql`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import psycopg

# Convenzione condivisa: "" = la riga vale per l'intero comune, non per
# una frazione specifica. Vedi il docstring del modulo per perché non
# NULL.
COMUNE_INTERO = ""


@dataclass(frozen=True)
class RigaCalendario:
    """Una data di raccolta, come esce dal parsing di un Markdown di
    calendario e come entra in `raccolta_date`."""

    categoria: str
    data: date
    colore: str = ""


@dataclass(frozen=True)
class ProssimaRaccolta:
    categoria: str
    data: date
    colore: str
    # Pubblicazione del gestore da cui viene questa data: serve al bot
    # per citarla nella risposta. Vuote se l'area non dichiara [[fonti]].
    # Frazione da cui la riga proviene: "" se la risposta arriva dal
    # calendario dell'intero comune (nessun override per la frazione
    # dell'utente, o utente senza frazione). Esposto perché il
    # chiamante possa dirlo in modo trasparente, non solo per debug.
    hamlet: str = ""
    # Pubblicazione del gestore da cui viene questa data: serve al bot
    # per citarla nella risposta. Vuote se l'area non dichiara [[fonti]].
    fonte_nome: str = ""
    fonte_url: str = ""


def normalizza_categoria(categoria: str) -> str:
    """MAIUSCOLO e spazi normalizzati: la stessa categoria arriva dal
    vocabolario come "Imb. Plast. e Metalli" e dall'intestazione del
    calendario come "IMB. PLAST. E METALLI". Normalizzare in scrittura
    E in lettura è ciò che tiene collegate le due fonti — un
    disallineamento qui non produce un errore, produce silenziosamente
    "nessuna data trovata", che è molto peggio.
    """
    return " ".join(categoria.split()).upper()


@dataclass(frozen=True)
class RisoluzioneCategoria:
    """Esito della traduzione materiale -> categoria locale."""

    # Nome della categoria di raccolta in questo comune/frazione, oppure
    # None se il materiale NON è raccolto porta a porta qui.
    categoria: str | None
    # Frazione effettivamente usata: quella dell'utente se ha uno schema
    # proprio, altrimenti "" (comune intero).
    hamlet: str
    # False = per questa coppia comune/frazione non è dichiarata nessuna
    # mappatura, e si è ricaduti sull'identità. Distingue "non
    # configurato" da "deliberatamente non raccolto porta a porta": nel
    # primo caso `categoria is None` non si verifica mai, nel secondo sì
    # ed è un'informazione da dare all'utente, non un errore.
    mappatura_dichiarata: bool


# --- scrittura (ingestion) --------------------------------------------------


def replace_calendario(
    conn: psycopg.Connection,
    comune_id: str,
    hamlet: str,
    source: str,
    righe: list[RigaCalendario],
    fonte_nome: str = "",
    fonte_url: str = "",
) -> int:
    """Sostituisce le righe provenienti da `source` per questa coppia
    comune/frazione, poi reinserisce quelle passate.

    DELETE-per-source + INSERT invece di un upsert riga per riga: un
    file di calendario rappresenta un semestre intero, e una correzione
    tipica non è "aggiungi una data" ma "questo elenco era sbagliato,
    eccone uno nuovo". Con il solo upsert una data rimossa dal file
    resterebbe in tabella per sempre, invisibile finché il tool non la
    restituisce come prossima raccolta.

    L'`ON CONFLICT DO UPDATE` serve comunque per il caso in cui due file
    diversi (es. due semestri che si sovrappongono di qualche giorno)
    dichiarino la stessa data: vince l'ultimo ingerito invece di far
    fallire l'intera transazione su una violazione di vincolo.
    """
    categoria_per_riga = [
        (normalizza_categoria(r.categoria), r.data, r.colore or "") for r in righe
    ]

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM raccolta_date "
            "WHERE comune_id = %s AND hamlet = %s AND source = %s",
            (comune_id, hamlet, source),
        )
        if categoria_per_riga:
            cur.executemany(
                """
                INSERT INTO raccolta_date
                    (comune_id, hamlet, categoria, colore, data, source,
                     fonte_nome, fonte_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (comune_id, hamlet, categoria, data) DO UPDATE SET
                    colore = EXCLUDED.colore,
                    source = EXCLUDED.source,
                    fonte_nome = EXCLUDED.fonte_nome,
                    fonte_url = EXCLUDED.fonte_url
                """,
                [
                    (
                        comune_id,
                        hamlet,
                        categoria,
                        colore,
                        data,
                        source,
                        fonte_nome,
                        fonte_url,
                    )
                    for categoria, data, colore in categoria_per_riga
                ],
            )

    return len(categoria_per_riga)


def replace_materiali(
    conn: psycopg.Connection,
    comune_id: str,
    hamlet: str,
    source: str,
    mappatura: dict[str, str | None],
) -> int:
    """Sostituisce la mappatura materiale -> categoria per questa coppia
    comune/frazione. Un valore None significa "materiale non raccolto
    porta a porta qui"."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM raccolta_materiale WHERE comune_id = %s AND hamlet = %s",
            (comune_id, hamlet),
        )
        if mappatura:
            cur.executemany(
                """
                INSERT INTO raccolta_materiale
                    (comune_id, hamlet, materiale, categoria, source)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        comune_id,
                        hamlet,
                        materiale.strip().lower(),
                        normalizza_categoria(categoria) if categoria else None,
                        source,
                    )
                    for materiale, categoria in mappatura.items()
                ],
            )
    return len(mappatura)


def upsert_frazione(
    conn: psycopg.Connection, comune_id: str, hamlet: str, nome: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO frazioni (comune_id, hamlet, nome)
            VALUES (%s, %s, %s)
            ON CONFLICT (comune_id, hamlet) DO UPDATE SET nome = EXCLUDED.nome
            """,
            (comune_id, hamlet, nome),
        )


# --- lettura (bot) ----------------------------------------------------------


def _hamlet_effettivo(conn: psycopg.Connection, comune_id: str, hamlet: str) -> str:
    """Frazione da usare nel filtro: quella dell'utente se ha un
    calendario proprio, altrimenti "" (calendario dell'intero comune).
    Vedi il docstring del modulo per perché il controllo è per frazione
    e non per (frazione, categoria)."""
    if not hamlet:
        return COMUNE_INTERO
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM raccolta_date "
            "WHERE comune_id = %s AND hamlet = %s)",
            (comune_id, hamlet),
        )
        (esiste,) = cur.fetchone()
    return hamlet if esiste else COMUNE_INTERO


def risolvi_categoria(
    conn: psycopg.Connection,
    comune_id: str,
    materiale: str,
    hamlet: str = COMUNE_INTERO,
) -> RisoluzioneCategoria:
    """Traduce un materiale canonico nella categoria di raccolta usata
    in questo comune/frazione.

    La frazione effettiva è calcolata con lo stesso `_hamlet_effettivo`
    usato per le date: mappatura e calendario devono venire sempre dalla
    stessa fonte, altrimenti si potrebbe risolvere "cartone" con lo
    schema diviso della frazione e poi cercarne le date nel calendario
    del comune, che quella categoria non ce l'ha.

    Nessuna riga per la coppia comune/frazione = nessuna mappatura
    dichiarata: si ricade sull'identità (il materiale usato com'è come
    nome di categoria), che è il comportamento precedente
    all'introduzione della tabella. Le aree con un solo schema di
    raccolta non devono dichiarare niente.
    """
    hamlet_usato = _hamlet_effettivo(conn, comune_id, hamlet)
    materiale_norm = materiale.strip().lower()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT categoria FROM raccolta_materiale "
            "WHERE comune_id = %s AND hamlet = %s AND materiale = %s",
            (comune_id, hamlet_usato, materiale_norm),
        )
        riga = cur.fetchone()
        if riga is not None:
            return RisoluzioneCategoria(
                categoria=riga[0], hamlet=hamlet_usato, mappatura_dichiarata=True
            )

        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM raccolta_materiale "
            "WHERE comune_id = %s AND hamlet = %s)",
            (comune_id, hamlet_usato),
        )
        (dichiarata,) = cur.fetchone()

    if not dichiarata:
        # Identità: nessuna mappatura per quest'area, il materiale vale
        # direttamente come nome di categoria.
        return RisoluzioneCategoria(
            categoria=normalizza_categoria(materiale),
            hamlet=hamlet_usato,
            mappatura_dichiarata=False,
        )

    # Mappatura dichiarata ma questo materiale non c'è: non è un errore,
    # è un materiale che il chiamante non ha previsto. Trattato come
    # sconosciuto (categoria None) — sta al chiamante distinguerlo dal
    # caso "dichiarato esplicitamente non raccolto porta a porta"
    # guardando materiali_disponibili().
    return RisoluzioneCategoria(
        categoria=None, hamlet=hamlet_usato, mappatura_dichiarata=True
    )


def materiali_disponibili(
    conn: psycopg.Connection, comune_id: str, hamlet: str = COMUNE_INTERO
) -> dict[str, str | None]:
    """Materiali dichiarati per questo comune/frazione, con la categoria
    a cui portano (None = non raccolto porta a porta qui). Vuoto se
    nessuna mappatura è dichiarata."""
    hamlet_usato = _hamlet_effettivo(conn, comune_id, hamlet)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT materiale, categoria FROM raccolta_materiale "
            "WHERE comune_id = %s AND hamlet = %s ORDER BY materiale",
            (comune_id, hamlet_usato),
        )
        return dict(cur.fetchall())


def categorie_disponibili(
    conn: psycopg.Connection, comune_id: str, hamlet: str = COMUNE_INTERO
) -> list[str]:
    """Categorie realmente presenti nel calendario di questo comune
    (o della sua frazione, se ha un override).

    Lette dal database invece di essere una costante nel codice: le
    categorie sono definite dal gestore, cambiano da area ad area
    ("RUR" qui, "Secco residuo" altrove), e una lista fissa nel codice
    sarebbe una seconda fonte di verità da tenere allineata a mano. Il
    tool le usa per validare l'argomento del modello e per elencare le
    alternative quando sbaglia."""
    hamlet_usato = _hamlet_effettivo(conn, comune_id, hamlet)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT categoria FROM raccolta_date "
            "WHERE comune_id = %s AND hamlet = %s ORDER BY categoria",
            (comune_id, hamlet_usato),
        )
        return [row[0] for row in cur.fetchall()]


def fonte_calendario(
    conn: psycopg.Connection, comune_id: str, hamlet: str = COMUNE_INTERO
) -> tuple[str, str] | None:
    """(nome, url) della pubblicazione da cui vengono i calendari di
    questa destinazione, o None se non dichiarata."""
    hamlet_usato = _hamlet_effettivo(conn, comune_id, hamlet)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fonte_nome, fonte_url FROM raccolta_date "
            "WHERE comune_id = %s AND hamlet = %s AND fonte_nome <> '' LIMIT 1",
            (comune_id, hamlet_usato),
        )
        riga = cur.fetchone()
    return (riga[0], riga[1] or "") if riga else None


def prossima_raccolta(
    conn: psycopg.Connection,
    comune_id: str,
    categoria: str,
    hamlet: str = COMUNE_INTERO,
    oggi: date | None = None,
) -> ProssimaRaccolta | None:
    """Prima data di raccolta di `categoria` a partire da oggi incluso,
    o None se il calendario ingerito non ne copre nessuna.

    `>= oggi`, non `> oggi`: se la raccolta è oggi stesso quella È la
    risposta utile, e nasconderla per saltare a quella dopo darebbe una
    data sbagliata a chi chiede la mattina del giorno di raccolta. Sta
    al chiamante distinguere "oggi" da "fra tre giorni" nel testo che
    mostra (vedi `bordeus_bot.calendario`).

    None è un risultato legittimo, non un errore: significa quasi sempre
    che il calendario ingerito si ferma prima della data richiesta e
    serve caricare il semestre successivo.
    """
    oggi = oggi or date.today()  # noqa: DTZ011 — solo la data: un calendario di raccolta non ha orario né fuso
    hamlet_usato = _hamlet_effettivo(conn, comune_id, hamlet)
    categoria_norm = normalizza_categoria(categoria)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT categoria, data, colore, fonte_nome, fonte_url
            FROM raccolta_date
            WHERE comune_id = %s AND hamlet = %s AND categoria = %s AND data >= %s
            ORDER BY data ASC LIMIT 1
            """,
            (comune_id, hamlet_usato, categoria_norm, oggi),
        )
        riga = cur.fetchone()

    if riga is None:
        return None
    categoria_trovata, data, colore, fonte_nome, fonte_url = riga
    return ProssimaRaccolta(
        categoria=categoria_trovata,
        data=data,
        colore=colore,
        hamlet=hamlet_usato,
        fonte_nome=fonte_nome or "",
        fonte_url=fonte_url or "",
    )
