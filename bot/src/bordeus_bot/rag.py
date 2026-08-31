"""RAG + tool calling: retrieval sul vector store dell'area Sub-ATO
(scritto dall'ingestion, riusato via `bordeus_common.vectorstore`) e
generazione della risposta con un modello Ollama.

## Un passaggio di retrieval, non due

La versione precedente ne faceva due: uno sul vocabolario e uno sul
calendario, incatenato al primo. Il secondo non c'è più, perché il
calendario non è più nel vector store: le date vivono in
`raccolta_date` e il modello le ottiene chiamando uno strumento (vedi
`calendario.py`).

Il cambio non è solo di implementazione. Il vecchio secondo passaggio
metteva nel prompt un chunk con ~50 date e chiedeva implicitamente al
modello di trovare la prima successiva a oggi — un confronto fra date
fatto leggendo testo, in cui i modelli sbagliano in modo plausibile e
quindi difficile da notare. Ora quel confronto è una query SQL
indicizzata e il modello riceve una sola data, già corretta.

Resta un solo passaggio di similarità, sul vocabolario/guide, dove la
ricerca semantica è lo strumento giusto: l'utente descrive un oggetto
con parole sue ("tazza da caffè scheggiata") e va trovata la voce
giusta anche se il vocabolario la chiama diversamente.

## Perché non `vectorstore.as_retriever()`

Fissa i `search_kwargs` (incluso il filtro) alla costruzione, ma qui il
filtro dipende dal comune di chi fa la domanda — comuni diversi della
stessa area condividono lo stesso vector store cachato (vedi
`telegram_bot.Service`) ma hanno filtri diversi. Chiamiamo
`similarity_search(..., filter=...)` direttamente per ogni domanda.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bordeus_common.log import blocco, get_logger
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector

from . import i18n

logger = get_logger("bordeus_bot")

# microsoft/harrier-oss-v1-0.6b è decoder-only con last-token pooling: le
# QUERY vanno prefissate con un'istruzione in linguaggio naturale per
# ottenere buoni risultati, i DOCUMENTI no (confermato dalla FAQ della
# model card). Passata a bordeus_common.embed.get_embeddings(
# query_instruction=...), che la applica SOLO a embed_query — l'ingestion,
# che embedda solo documenti, non la usa mai. Resta in inglese
# indipendentemente dalla lingua dell'utente: è un'istruzione per il
# modello di embedding, non testo che l'utente vede — vedi invece
# {lingua} nel SYSTEM_PROMPT_TEMPLATE per la lingua della RISPOSTA.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about waste disposal and recycling, "
    "retrieve relevant passages that answer the question\nQuery: "
)

# Numero massimo di giri di tool calling per una singola domanda. Serve
# a chiudere il caso in cui il modello continui a richiamare lo strumento
# (es. sbagliando categoria e riprovando all'infinito): senza un limite,
# una singola domanda potrebbe tenere occupato un thread finché non
# scade il timeout di Telegram. Due giri bastano per il caso reale
# "categoria sbagliata al primo tentativo, corretta al secondo".
MAX_GIRI_TOOL = 3

# Chunk di vocabolario da recuperare. Da quando ogni voce del vocabolario
# è un chunk a sé (vedi bordeus_ingest.chunk), `k` è il numero di
# OGGETTI proposti al modello, non di blocchi da ~7 voci: 6 era poco
# generoso, 8 lascia margine per le varianti della stessa voce
# ("Ceramica piccole quantità" / "grosse quantità") senza allungare
# troppo il prompt.
DEFAULT_K_VOCABOLARIO = 8

# Chiave con cui uno strumento dichiara, in `tool.metadata`, la
# pubblicazione da cui vengono i suoi dati. Definita qui e non in
# calendario.py perché è il contratto fra i due: qualunque strumento che
# la valorizzi vede la propria fonte citata in fondo alla risposta.
FONTE_METADATA_KEY = "fonte"

# Il blocco "Calendario di raccolta" non c'è più: al suo posto
# un'istruzione a usare lo strumento. È la differenza fra dare al modello
# i dati grezzi da interpretare e dargli il risultato già calcolato.
#
# L'istruzione "rispondi esclusivamente in base al contesto" è esplicita
# per un motivo preciso: un LLM ha comunque una conoscenza generica sullo
# smaltimento rifiuti "tipico" (es. "la plastica di solito va nel giallo"),
# ma qui le regole cambiano da comune a comune e da gestore a gestore —
# una risposta plausibile ma generica può essere sbagliata per l'utente
# specifico. Dire solo "se non sai, ammettilo" non basta: un modello può
# "sapere" una risposta generica senza riconoscerla come un'ipotesi.
#
# {lingua} è altrettanto esplicito: il contesto recuperato è quasi sempre
# in italiano, ma la Valle d'Aosta accoglie molti turisti. "Rispondi
# SEMPRE in X" invece di "traduci se serve", che il modello potrebbe
# interpretare come facoltativo.
SYSTEM_PROMPT_TEMPLATE = """\
Sei un assistente cordiale e competente che rappresenta un'azienda di gestione rifiuti.
Stai chattando con un utente sulle corrette modalità di smaltimento dei rifiuti.

Rispondi ESCLUSIVAMENTE in base al contesto fornito qui sotto ("Modalità di smaltimento") e ai risultati degli strumenti che hai a disposizione.
Non usare conoscenza generale sullo smaltimento dei rifiuti che potresti già avere, anche se pensi di sapere la risposta: le regole cambiano da comune a comune e da gestore a gestore, un'informazione generica può essere sbagliata per l'utente specifico che ti sta scrivendo.
Se il contesto non contiene l'informazione richiesta, dillo esplicitamente invece di rispondere con conoscenza generica.

Rispondi SEMPRE in {lingua}, indipendentemente dalla lingua del contesto sottostante.

Usa il contesto "Modalità di smaltimento" per capire in quale categoria di conferimento rientra l'oggetto. Presta sempre attenzione alle quantità indicate (es. singolare/plurale): possono cambiare la modalità corretta (piccole quantità nel bidone, grandi quantità all'ecocentro).

Le modalità possibili sono: raccolta "Porta a Porta" (organico, carta e cartoni, imballaggi in plastica e metalli, vetro, RUR/indifferenziato), conferimento all'Ecocentro, canali specifici (es. farmacie per i medicinali), o ritiro ingombranti su chiamata.

Se la modalità corretta è la raccolta "Porta a Porta", chiama SEMPRE lo strumento {tool_name} per conoscere il giorno di passaggio, e riportalo nella risposta anche se l'utente non l'ha chiesto esplicitamente.
Allo strumento passa il MATERIALE dell'oggetto (carta, cartone, vetro, organico, plastica, metalli, indifferenziato), non il nome del bidone: comuni e frazioni diversi raggruppano i materiali in flussi di raccolta diversi, e la corrispondenza la risolve lo strumento. Lo strumento ti risponde con il nome del flusso corretto per questo utente: usa QUEL nome nella risposta, non quello che compare nel contesto qui sotto, che è condiviso da tutta l'area e può usare un raggruppamento diverso.
Non calcolare, stimare o inventare MAI una data di raccolta da solo: se lo strumento non restituisce una data, dillo apertamente.
Se invece l'oggetto non è raccolto porta a porta (es. va all'Ecocentro), non chiamare lo strumento e non menzionare nessun giorno.

Sii conciso. Indica la categoria di conferimento corretta e, quando pertinente, il giorno di raccolta.

Modalità di smaltimento:
{vocabolario_context}
"""


@dataclass(frozen=True)
class Fonte:
    """Pubblicazione del gestore da cui viene una parte della risposta."""

    nome: str
    url: str = ""

    def __str__(self) -> str:
        return f"{self.nome} — {self.url}" if self.url else self.nome


@dataclass
class Risposta:
    """Testo generato più le fonti da cui il contesto proviene.

    Le fonti sono raccolte **in Python**, dai metadata dei chunk
    recuperati e dagli strumenti effettivamente invocati — non chieste al
    modello. Chiedergliele significherebbe farsi restituire un URL
    generato token per token: un modello che ha davanti un link tende a
    riprodurlo con una cifra diversa o a inventarne uno plausibile, e
    una fonte sbagliata è peggio di nessuna fonte, perché sposta la
    fiducia su qualcosa che non si può verificare.
    """

    testo: str
    fonti: list[Fonte] = field(default_factory=list)

    def __str__(self) -> str:
        # Comodità per i notebook e per il codice che tratta la risposta
        # come una stringa: `str(risposta)` resta il solo testo.
        return self.testo


def _fonti_dai_documenti(docs) -> list[Fonte]:
    """Fonti distinte dei chunk recuperati, nell'ordine di comparsa.

    Distinte perché il vocabolario di un'area viene di solito da una
    sola pubblicazione: elencarla otto volte, una per chunk, sarebbe
    rumore."""
    viste: dict[tuple[str, str], Fonte] = {}
    for doc in docs:
        nome = doc.metadata.get("fonte_nome")
        if not nome:
            continue
        url = doc.metadata.get("fonte_url", "")
        viste.setdefault((nome, url), Fonte(nome=nome, url=url))
    return list(viste.values())


def comune_filter(comune_id: str) -> dict:
    """Filtro per il retrieval: contenuto condiviso dall'area
    (`comune_id` vuoto nel metadata dei chunk) PIÙ contenuto specifico
    del comune dato — mai quello di un comune diverso della stessa area.

    Verificato che `langchain-postgres` supporti davvero questo `$or`
    (letto il sorgente e testato contro Postgres reale, non assunto)."""
    return {"$or": [{"comune_id": ""}, {"comune_id": comune_id}]}


def _combine_filters(*filters: dict) -> dict:
    """Combina più filtri con `$and`. Verificato che
    `langchain-postgres` supporti l'annidamento `$and` con dentro un
    `$or` (non assunto)."""
    non_empty = [f for f in filters if f]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def vocabolario_filter(comune_id: str) -> dict:
    """Contenuto su cui cercare la descrizione dell'oggetto.

    L'esclusione di `tipo="calendario"` resta anche se l'ingestion non
    scrive più calendari nel vector store: una collection ingerita con
    la versione precedente contiene ancora quei chunk, e `add_chunks`
    aggiorna i chunk esistenti senza cancellare quelli che non produce
    più (vedi il limite noto in `bordeus_common.vectorstore.
    stable_chunk_id`). Senza questo filtro, un elenco di date del
    semestre scorso potrebbe tornare fra i risultati di una ricerca sul
    vocabolario e finire nel prompt — esattamente la situazione che il
    tool calling serve a evitare."""
    return _combine_filters(
        {"tipo": {"$nin": ["calendario", "calendari"]}}, comune_filter(comune_id)
    )


def build_question(object_description: str) -> str:
    """Frase completa da mandare all'LLM per la generazione finale,
    costruita a partire dalla sola descrizione dell'oggetto — quella
    usata da sola, senza wrapping, per il retrieval. Esposta a parte
    perché `bot/notebooks/rag_eval.ipynb` reimplementa retrieval e
    generazione a mano per poter scambiare system prompt diversi, e deve
    costruire esattamente la stessa domanda usata in produzione."""
    return f"Come smaltisco questo oggetto: {object_description}?"


def retrieve_vocabolario(
    vectorstore: PGVector,
    comune_id: str,
    object_description: str,
    k: int = DEFAULT_K_VOCABOLARIO,
) -> list:
    """Restituisce i Document, non il testo già unito: il chiamante ha
    bisogno anche dei metadata (la provenienza) e non solo del
    contenuto. Il testo si ottiene con `contesto_da_documenti`."""
    filtro = vocabolario_filter(comune_id)
    docs = vectorstore.similarity_search(object_description, k=k, filter=filtro)

    logger.trace(  # type: ignore[attr-defined]
        "retrieval vocabolario: query=%r k=%d filtro=%s -> %d chunk%s",
        object_description,
        k,
        filtro,
        len(docs),
        blocco(
            "chunk recuperati",
            "\n\n".join(
                f"[{i}] fonte={doc.metadata.get('source')} "
                f"tipo={doc.metadata.get('tipo')} "
                f"comune={doc.metadata.get('comune_id')!r}\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
            or "(nessun chunk)",
        ),
    )
    return docs


def contesto_da_documenti(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _esegui_tool_calls(messages: list, risposta, tools_by_name: dict) -> None:
    """Esegue le chiamate richieste dal modello e accoda i risultati.
    Muta `messages` in-place — il ciclo chiamante ha bisogno della
    cronologia completa (system, human, ai con le tool_calls, tool
    results) per il giro successivo: un modello che riceve un
    ToolMessage senza la propria richiesta davanti rifiuta la
    conversazione come malformata."""
    for chiamata in risposta.tool_calls:
        strumento = tools_by_name.get(chiamata["name"])
        if strumento is None:
            logger.warning("il modello ha chiesto uno strumento sconosciuto: %s", chiamata["name"])
            risultato = (
                f"Strumento {chiamata['name']!r} inesistente. Strumenti "
                f"disponibili: {', '.join(tools_by_name)}."
            )
        else:
            logger.trace(  # type: ignore[attr-defined]
                "tool %s <- argomenti dal modello: %s",
                chiamata["name"],
                chiamata["args"],
            )
            try:
                risultato = strumento.invoke(chiamata["args"])
            except Exception as exc:
                # Un tool che fallisce non deve far fallire la risposta:
                # il modello può ancora dire come si smaltisce l'oggetto,
                # solo senza il giorno di raccolta. Restituire l'errore
                # come contenuto (invece di risollevarlo) glielo lascia
                # gestire come informazione mancante.
                logger.error("tool %s fallito: %s", chiamata["name"], exc)
                risultato = (
                    "Lo strumento non è al momento disponibile. Rispondi "
                    "senza indicare nessun giorno di raccolta."
                )
        logger.trace(  # type: ignore[attr-defined]
            "tool %s -> %s", chiamata["name"], risultato
        )
        messages.append(ToolMessage(content=risultato, tool_call_id=chiamata["id"]))


def answer_question(
    llm: ChatOllama,
    vectorstore: PGVector,
    comune_id: str,
    object_description: str,
    tools: list | None = None,
    language_code: str | None = None,
    k_vocabolario: int = DEFAULT_K_VOCABOLARIO,
    system_prompt_template: str | None = None,
) -> Risposta:
    """Recupera il contesto di vocabolario e genera la risposta, dando al
    modello gli strumenti con cui ottenere il giorno di raccolta.

    `object_description` è SOLO la descrizione dell'oggetto (es.
    "bottiglia di plastica"), già estratta da `identify.py` — non
    l'intera domanda dell'utente. Usarla così com'è per il retrieval è
    più preciso: il vector store fa match sul contenuto
    (materiale/categoria), non su come l'utente ha formulato la domanda,
    che per una ricerca per similarità è solo rumore. La domanda completa
    per la generazione (dove invece la formulazione naturale aiuta) viene
    costruita qui dentro con `build_question`.

    `tools` vuoto o assente = generazione senza tool calling. Il resto
    funziona comunque: la risposta indicherà la modalità di conferimento
    senza il giorno. Serve a un modello che non supporti il tool calling
    (verificato: `gemma3` non lo supporta in Ollama, nemmeno nella
    variante 27b — `gemma4` sì).

    `system_prompt_template` sostituisce `SYSTEM_PROMPT_TEMPLATE` a
    parità di tutto il resto. Esiste per `bot/notebooks/rag_eval.ipynb`,
    che confronta formulazioni diverse del prompt: senza questo
    parametro il notebook doveva reimplementare la generazione a mano, e
    quella copia — priva di tool calling — finiva per valutare un
    percorso che non è più quello del bot. Deve accettare gli stessi
    placeholder: {lingua}, {tool_name}, {vocabolario_context}.
    """
    docs = retrieve_vocabolario(
        vectorstore, comune_id, object_description, k=k_vocabolario
    )
    vocabolario_context = contesto_da_documenti(docs)
    fonti = _fonti_dai_documenti(docs)

    tools = tools or []
    tools_by_name = {t.name: t for t in tools}

    template = system_prompt_template or SYSTEM_PROMPT_TEMPLATE
    system_prompt = template.format(
        lingua=i18n.language_name(language_code),
        tool_name=", ".join(tools_by_name) or "(nessuno strumento disponibile)",
        vocabolario_context=vocabolario_context or "Nessuna informazione trovata.",
    )
    domanda = build_question(object_description)

    # Il prompt assemblato per intero, com'è arrivato al modello: è
    # l'unica informazione che permette di capire a posteriori perché ha
    # risposto in un certo modo, e non è ricostruibile dagli altri log
    # (dipende da lingua, contesto recuperato e strumenti disponibili
    # insieme). A TRACE e non a DEBUG perché sono migliaia di caratteri
    # per ogni messaggio — vedi bordeus_common.log.
    logger.trace(  # type: ignore[attr-defined]
        "generazione: comune=%s lingua=%s strumenti=%s template=%s%s%s",
        comune_id,
        i18n.language_name(language_code),
        ", ".join(tools_by_name) or "nessuno",
        "personalizzato" if system_prompt_template else "SYSTEM_PROMPT_TEMPLATE",
        blocco("system prompt", system_prompt),
        blocco("messaggio utente", domanda),
    )
    for nome, strumento in tools_by_name.items():
        logger.trace(  # type: ignore[attr-defined]
            "strumento offerto al modello: %s%s",
            nome,
            blocco(f"descrizione di {nome}", strumento.description),
        )

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[{"type": "text", "text": domanda}]),
    ]

    if not tools:
        risposta = llm.invoke(messages).content
        logger.trace(  # type: ignore[attr-defined]
            "risposta finale (senza strumenti)%s", blocco("risposta", str(risposta))
        )
        return Risposta(testo=risposta, fonti=fonti)

    llm_con_tool = llm.bind_tools(tools)
    fonti_strumenti: list[Fonte] = []

    for giro in range(MAX_GIRI_TOOL):
        risposta = llm_con_tool.invoke(messages)
        messages.append(risposta)

        if not getattr(risposta, "tool_calls", None):
            logger.trace(  # type: ignore[attr-defined]
                "risposta finale al giro %d%s",
                giro + 1,
                blocco("risposta", str(risposta.content)),
            )
            return Risposta(testo=risposta.content, fonti=fonti + fonti_strumenti)

        logger.info(
            "giro %d: il modello ha richiesto %d chiamate a strumenti",
            giro + 1,
            len(risposta.tool_calls),
        )
        # La fonte di uno strumento va citata solo se lo strumento è
        # stato davvero invocato: un calendario che il modello non ha
        # consultato non ha contribuito alla risposta.
        for chiamata in risposta.tool_calls:
            strumento = tools_by_name.get(chiamata["name"])
            fonte = (getattr(strumento, "metadata", None) or {}).get(FONTE_METADATA_KEY)
            if fonte is not None and fonte not in fonti_strumenti:
                fonti_strumenti.append(fonte)
        _esegui_tool_calls(messages, risposta, tools_by_name)

    # Limite raggiunto: chiediamo una risposta finale senza strumenti,
    # invece di restituire l'ultimo messaggio (che è una richiesta di
    # tool, quindi con `content` vuoto — l'utente vedrebbe una risposta
    # vuota senza capire perché).
    logger.warning(
        "raggiunto il limite di %d giri di tool calling: genero la risposta "
        "finale senza strumenti",
        MAX_GIRI_TOOL,
    )
    return Risposta(testo=llm.invoke(messages).content, fonti=fonti + fonti_strumenti)
