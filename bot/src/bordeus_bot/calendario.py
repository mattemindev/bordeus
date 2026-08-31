"""Il tool che il modello chiama per sapere quando passa la raccolta.

Sostituisce il secondo passaggio RAG della versione precedente, che
recuperava i chunk di calendario e li metteva nel prompt come testo. Il
motivo del cambio è che quel passaggio chiedeva al modello la cosa in
cui è meno affidabile: leggere ~50 date da un elenco e trovare la prima
successiva a oggi. Gli errori non erano rari, e soprattutto erano
invisibili — una data plausibile ma sbagliata di una settimana sembra
una risposta corretta. Ora il confronto fra date lo fa Postgres
(`ORDER BY data ASC LIMIT 1` su un indice) e il modello riceve una sola
data già giusta, che deve solo riportare.

## Il parametro è un MATERIALE, non una categoria di raccolta

Il modello passa `materiale`: carta, cartone, vetro, organico, plastica,
metalli, indifferenziato. Termini intrinseci all'oggetto, uguali in
tutta la regione. La traduzione nel nome del flusso di raccolta locale
avviene qui dentro (`bordeus_common.calendario.risolvi_categoria`),
sulla base di quanto dichiarato nel manifest dell'area.

Il motivo è un caso reale, non un'astrazione: Bard, Donnas e Hône
raccolgono carta e cartone **insieme**, ma le frazioni di Bard e Donnas
(Crous, Albard, Les Pians, Bondon) li raccolgono **separati**, in due
giorni diversi. Se il parametro fosse la categoria, il modello dovrebbe
sapere quale schema vale per l'utente che ha davanti e, per le frazioni,
decidere lui se una scatola sia "carta" o "cartone" — con in mano solo
la descrizione dell'oggetto, e restituendo comunque una data plausibile
quando sbaglia. Con il materiale, quella decisione la prendono i dati.

È lo stesso principio già applicato a `comune_id`: ciò che varia
localmente è dato, non qualcosa che il modello debba indovinare.

## `comune_id` non è un parametro del modello

Comune e frazione arrivano dal profilo utente già confermato in fase di
onboarding e sono legati alla funzione tramite chiusura, quindi non
compaiono nello schema che il modello vede. Se fossero parametri,
un'allucinazione o un'istruzione infilata nel messaggio dell'utente
potrebbero far rispondere con il calendario di un comune diverso —
sbagliato in un modo che l'utente non ha modo di riconoscere. È lo
stesso principio per cui un handler HTTP prende l'identità dalla
sessione e non dal body della richiesta.
"""

from __future__ import annotations

import inspect
from datetime import date

from bordeus_common import calendario as calendario_db
from bordeus_common import db as bot_db
from bordeus_common.log import get_logger
from langchain_core.tools import tool

from .rag import FONTE_METADATA_KEY, Fonte

logger = get_logger("bordeus_bot")

TOOL_NAME = "trova_prossima_raccolta"

# Materiali canonici: termini intrinseci all'oggetto, non nomi di bidoni.
# Sono l'insieme chiuso che il modello vede nella descrizione del tool, e
# restano gli stessi per ogni utente — a differenza delle categorie di
# raccolta, che cambiano da comune a comune e da frazione a frazione.
# La corrispondenza materiale -> categoria locale è dichiarata nel
# manifest dell'area e vive in `raccolta_materiale`.
MATERIALI_CANONICI = (
    "organico",
    "carta",
    "cartone",
    "plastica",
    "metalli",
    "vetro",
    "indifferenziato",
)


def descrivi_prossima_raccolta(
    database_url: str,
    comune_id: str,
    materiale: str,
    hamlet: str = "",
    oggi: date | None = None,
) -> str:
    """Testo che il tool restituisce al modello.

    Una frase in linguaggio naturale e non la sola data grezza: se il
    modello la riporta pari pari nella risposta finale, resta comunque
    corretta e completa. Il testo è in italiano anche per un utente
    tedesco — è contesto interno, e il system prompt impone comunque la
    lingua della risposta finale (vedi `rag.py`).
    """
    oggi = oggi or date.today()  # noqa: DTZ011 — solo la data: nessun orario né fuso in gioco
    conn = bot_db.connect_light(database_url)
    try:
        risoluzione = calendario_db.risolvi_categoria(
            conn, comune_id, materiale, hamlet=hamlet
        )

        logger.trace(  # type: ignore[attr-defined]
            "risoluzione materiale=%r comune=%s hamlet=%r -> categoria=%r "
            "(hamlet effettivo=%r, mappatura dichiarata=%s)",
            materiale,
            comune_id,
            hamlet,
            risoluzione.categoria,
            risoluzione.hamlet,
            risoluzione.mappatura_dichiarata,
        )

        if risoluzione.categoria is None:
            noti = calendario_db.materiali_disponibili(conn, comune_id, hamlet=hamlet)
            if materiale.strip().lower() in noti:
                # Dichiarato esplicitamente come non raccolto porta a
                # porta: è un'informazione utile, non un fallimento.
                return (
                    f"Il materiale {materiale!r} non è raccolto porta a porta in "
                    "questo comune. Rispondi indicando il conferimento "
                    "all'Ecocentro o il canale specifico previsto, senza citare "
                    "nessun giorno di raccolta."
                )
            return (
                f"Materiale {materiale!r} non riconosciuto. I materiali raccolti "
                f"porta a porta qui sono: {', '.join(sorted(k for k, v in noti.items() if v))}. "
                "Richiama lo strumento con uno di questi, se pertinente."
            )

        risultato = calendario_db.prossima_raccolta(
            conn, comune_id, risoluzione.categoria, hamlet=hamlet, oggi=oggi
        )
        if risultato is None:
            categorie = calendario_db.categorie_disponibili(
                conn, comune_id, hamlet=hamlet
            )
    finally:
        conn.close()

    if risultato is not None:
        quando = (
            "oggi stesso"
            if risultato.data == oggi
            else f"il {risultato.data.strftime('%d/%m/%Y')}"
        )
        contenitore = f" ({risultato.colore})" if risultato.colore else ""
        return (
            f"La prossima raccolta di {risultato.categoria}{contenitore} "
            f"è prevista {quando}."
        )

    if not categorie:
        # Nessuna riga per questo comune: il calendario non è mai stato
        # ingerito, non è che il materiale sia sbagliato. Dirlo al
        # modello evita che riprovi con altri materiali a vuoto.
        logger.warning(
            "nessun calendario in raccolta_date per comune=%s hamlet=%r",
            comune_id,
            hamlet,
        )
        return (
            "Nessun calendario di raccolta disponibile per questo comune. "
            "Rispondi indicando la modalità di conferimento senza citare "
            "nessun giorno di raccolta."
        )

    # Categoria risolta correttamente, ma nessuna data futura: il
    # calendario ingerito si ferma prima. È il caso limite di fine
    # semestre. Se invece la categoria risolta non fosse fra quelle
    # presenti, sarebbe un errore di mappatura nel manifest — che
    # `bordeus_ingest.calendario.valida_mappatura` avrebbe però già
    # bloccato in fase di ingestion.
    logger.warning(
        "calendario esaurito per comune=%s materiale=%s categoria=%s (oggi=%s): "
        "serve ingerire il semestre successivo",
        comune_id,
        materiale,
        risoluzione.categoria,
        oggi,
    )
    return (
        f"Il calendario disponibile per {risoluzione.categoria} non copre date "
        f"successive a oggi ({oggi.strftime('%d/%m/%Y')}). Rispondi indicando "
        "la modalità di conferimento e di' che il calendario aggiornato non è "
        "ancora disponibile — non stimare né inventare una data."
    )


def make_tool(database_url: str, comune_id: str, hamlet: str = ""):
    """Fabbrica del tool: `comune_id` e `hamlet` sono fissati qui, dal
    profilo utente, e restano nella chiusura. Il modello vede solo
    `materiale`.

    Un tool per richiesta, non uno condiviso nel `Service`: il comune
    cambia da utente a utente, e legare il comune a un oggetto
    riutilizzato fra conversazioni sarebbe il modo più diretto per
    rispondere a un utente con il calendario di un altro.

    L'elenco dei materiali viene interpolato nella docstring — che è la
    descrizione che il modello legge. Quando l'area dichiara una
    mappatura si usano i materiali realmente previsti lì; altrimenti si
    ricade su `MATERIALI_CANONICI`.
    """
    conn = bot_db.connect_light(database_url)
    try:
        noti = calendario_db.materiali_disponibili(conn, comune_id, hamlet=hamlet)
        fonte = calendario_db.fonte_calendario(conn, comune_id, hamlet=hamlet)
    finally:
        conn.close()

    raccolti = sorted(k for k, v in noti.items() if v)
    elenco = ", ".join(raccolti or MATERIALI_CANONICI)

    @tool(TOOL_NAME)
    def trova_prossima_raccolta(materiale: str) -> str:
        """Trova la data della prossima raccolta porta a porta per un
        materiale, nel comune dell'utente corrente.

        Usa SEMPRE questo strumento quando devi indicare un giorno di
        raccolta: non calcolare né inventare mai una data da solo.

        Passa il MATERIALE di cui è fatto l'oggetto, non il nome del
        bidone: comuni e frazioni diversi raggruppano i materiali in modo
        diverso, e la corrispondenza la risolve lo strumento. Distingui
        "carta" (giornali, quaderni, buste) da "cartone" (scatole,
        imballaggi, cassette della frutta) anche quando ti sembrano lo
        stesso flusso: in alcune frazioni sono raccolti in giorni diversi.

        Args:
            materiale: il materiale da cercare. Valori validi per questo
                utente: MATERIALI_VALIDI.
        """
        logger.info(
            "tool %s invocato: comune=%s hamlet=%r materiale=%r",
            TOOL_NAME,
            comune_id,
            hamlet,
            materiale,
        )
        return descrivi_prossima_raccolta(
            database_url, comune_id, materiale, hamlet=hamlet
        )

    # La docstring è già stata trasformata in descrizione al momento del
    # decoratore: la sostituzione va fatta sull'oggetto tool, non sulla
    # funzione (dove non avrebbe più effetto).
    #
    # cleandoc toglie l'indentazione ereditata dal corpo della funzione:
    # senza, la descrizione arriva al modello con quattro spazi davanti a
    # ogni riga tranne la prima. Non cambia il senso, ma è testo che il
    # modello legge, ed è anche ciò che si vede nei log TRACE.
    trova_prossima_raccolta.description = inspect.cleandoc(
        trova_prossima_raccolta.description
    ).replace("MATERIALI_VALIDI", elenco)

    # La pubblicazione da cui vengono queste date, agganciata allo
    # strumento: `rag.answer_question` la cita solo se il modello lo ha
    # davvero invocato. Non passa mai dal prompt — un URL generato dal
    # modello sarebbe un URL inventato.
    #
    # Va nel campo `metadata`, non in un attributo qualsiasi: gli
    # strumenti di LangChain sono modelli Pydantic e rifiutano
    # l'assegnazione di campi non dichiarati (`"StructuredTool" object
    # has no field "fonte"`). `description` sopra funziona perché è un
    # campo vero; `metadata` è quello previsto per i dati applicativi.
    if fonte and fonte[0]:
        trova_prossima_raccolta.metadata = {
            **(trova_prossima_raccolta.metadata or {}),
            FONTE_METADATA_KEY: Fonte(nome=fonte[0], url=fonte[1]),
        }
    return trova_prossima_raccolta
