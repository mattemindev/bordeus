"""RAG: retrieval sul vector store dell'area Sub-ATO (stesso
`langchain-postgres` scritto dall'ingestion, riusato via
`bordeus_common.vectorstore`) + generazione della risposta con un
modello Ollama.

Non usiamo `vectorstore.as_retriever()`: fissa i `search_kwargs` (incluso
un eventuale filtro) alla costruzione, ma qui il filtro dipende dal
comune specifico dell'utente che fa la domanda — comuni diversi della
stessa area (quindi stesso vector store cachato, vedi
`telegram_bot.Service`) hanno filtri diversi. Chiamiamo
`similarity_search(..., filter=...)` direttamente per ogni domanda.

## Retrieval a due passaggi (vocabolario + calendario)

La knowledge base ha due tipi di contenuto con ruoli diversi:
- **vocabolario/guide**: dato un oggetto, dice in quale bidone/modalità
  va smaltito (es. "bottiglia di plastica -> bidone giallo").
- **calendario**: dato un materiale, dice in che giorno passa il
  porta a porta — spesso specifico di un comune (vedi `comune_filter`).

Una singola ricerca per similarità sulla descrizione dell'oggetto
recupera bene il vocabolario, ma non garantisce di recuperare anche il
calendario giusto: sono formulati in modo diverso ("bottiglia di
plastica rossa da 1.5L" vs "plastica: lunedì"), due intenti semantici
diversi che una sola query non copre in modo affidabile. `answer_question`
fa quindi due ricerche in sequenza:

1. **Vocabolario** (`vocabolario_filter`, esclude esplicitamente
   `tipo="calendario"` invece di includere positivamente "guide": la
   classificazione euristica dell'ingestion — vedi
   `bordeus_ingest.classify` — potrebbe categorizzare il vocabolario
   come "guide" o come "altro" a seconda del contenuto, ma il
   calendario è sempre "calendario" con certezza, quindi escluderlo è
   il filtro più affidabile) — la descrizione dell'oggetto (non la
   domanda intera dell'utente, vedi `vision.py`) cerca qui.
2. **Calendario** (`calendario_filter`, `tipo="calendario"` combinato
   con `comune_filter`) — usa come query il TESTO recuperato al
   passaggio 1, non la domanda originale: il vocabolario di solito
   nomina il materiale/bidone, un match semantico migliore contro il
   calendario di quanto lo sia la descrizione originale, spesso più
   verbosa o specifica dell'oggetto.

Un solo giro di ricerca extra su Postgres (economico, locale), non una
chiamata Ollama in più — la latenza aggiuntiva è trascurabile rispetto
alla generazione della risposta.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector

from . import i18n

# microsoft/harrier-oss-v1-0.6b è decoder-only con last-token pooling:
# le QUERY vanno prefissate con un'istruzione in linguaggio naturale per
# ottenere buoni risultati, i DOCUMENTI no (confermato dalla FAQ della
# model card — "there is no need to add instructions to the document
# side"). Istruzione su misura per il nostro dominio, sul modello di
# quella d'esempio ("web search query") nella documentazione ufficiale.
# Passata a bordeus_common.embed.get_embeddings(query_instruction=...),
# che la applica SOLO a embed_query, mai a embed_documents — l'ingestion
# (che embedda solo documenti) non la usa mai. Usata per ENTRAMBI i
# passaggi di retrieval (vocabolario e calendario): in entrambi i casi
# stiamo interrogando il vector store con una query in linguaggio
# naturale, non embeddando un documento. Resta in inglese
# indipendentemente dalla lingua dell'utente: è un'istruzione per il
# modello di embedding, non testo che l'utente vede — vedi invece
# {lingua} nel SYSTEM_PROMPT_TEMPLATE per la lingua della RISPOSTA.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about waste disposal and recycling, "
    "retrieve relevant passages that answer the question\nQuery: "
)

# Tre placeholder (non due): oltre ai due contesti separati, {lingua} —
# il contesto recuperato è quasi sempre in italiano (le guide dei
# gestori sono in italiano), ma la Valle d'Aosta accoglie molti turisti
# che potrebbero non capirlo: l'istruzione dice esplicitamente al
# modello di rispondere nella lingua dell'utente indipendentemente dalla
# lingua del contesto, non di limitarsi a "tradurre se serve" (più
# debole, il modello potrebbe interpretarlo come facoltativo).
#
# L'istruzione "rispondi esclusivamente in base al contesto" è
# altrettanto esplicita, per lo stesso motivo: un LLM ha comunque una
# conoscenza generica sullo smaltimento rifiuti "tipico" (es. "la
# plastica di solito va nel bidone giallo"), ma qui le regole cambiano
# da comune a comune e da gestore a gestore — una risposta plausibile
# ma generica può essere sbagliata per l'utente specifico. Dire solo
# "se non sai, ammettilo" non basta: un modello può "sapere" una
# risposta generica senza riconoscerla come un'ipotesi, va detto
# esplicitamente di ignorare quella conoscenza generica a favore del
# contesto recuperato.
SYSTEM_PROMPT_TEMPLATE = """\
Sei un assistente cordiale e competente che rappresenta un'azienda di gestione rifiuti.
Stai chattando con un utente sulle corrette modalità di smaltimento dei rifiuti.

Rispondi ESCLUSIVAMENTE in base al contesto fornito qui sotto ("Modalità di smaltimento" e "Calendario di raccolta").
Non usare conoscenza generale sullo smaltimento dei rifiuti che potresti già avere, anche se pensi di sapere la risposta: le regole cambiano da comune a comune e da gestore a gestore, un'informazione generica può essere sbagliata per l'utente specifico che ti sta scrivendo.
Se il contesto non contiene l'informazione richiesta, dillo esplicitamente invece di rispondere con conoscenza generica.

Rispondi SEMPRE in {lingua}, indipendentemente dalla lingua del contesto sottostante.

Usa il contesto "Modalità di smaltimento" per dire in quale bidone o con quale modalità va l'oggetto. Presta sempre attenzione alle quantità indicate (es. parola singolare/plurale), possono essere fondamentali per determinare le modalità di smaltimento (es. piccole/medie/grandi quantità).

Usa il contesto "Calendario di raccolta" per indicare ANCHE il giorno di passaggio del porta a porta, anche se l'utente non lo ha chiesto esplicitamente — è un'informazione utile che va sempre data quando disponibile.
Se l'oggetto non è raccolto porta a porta (es. va all'ecocentro) o il contesto del calendario non è pertinente al materiale della domanda, non menzionare nessun giorno: non inventare un'informazione che non hai.

Sii conciso. Spiega brevemente la corretta procedura di smaltimento dell'oggetto, o degli oggetti.

Modalità di smaltimento:
{vocabolario_context}

Calendario di raccolta:
{calendario_context}
"""


def comune_filter(comune_id: str) -> dict:
    """Filtro per il retrieval: contenuto condiviso dall'area
    (`comune_id` vuoto nel metadata dei chunk) PIÙ contenuto specifico
    del comune dato — mai quello di un comune diverso nella stessa area.
    Necessario perché non tutto il contenuto di un'area è davvero
    condiviso tra i comuni che la compongono: il calendario di raccolta
    porta a porta, ad esempio, può variare da un comune all'altro per
    motivi logistici anche sotto lo stesso gestore (caso reale:
    TeknoService Italia, Sub-ATO E — Donnas e Pont-Saint-Martin, comuni
    confinanti, hanno calendari diversi). Vedi
    `bordeus_ingest.pipeline` per come i chunk vengono taggati in fase
    di ingestion (`--comune-url`), e `bordeus_common.vectorstore` per
    la scelta di una sola collection per area invece di una collection
    separata per comune da interrogare e unire.

    Verificato che `langchain-postgres` supporti davvero questo `$or`
    (letto il sorgente e testato contro Postgres reale, non assunto)."""
    return {"$or": [{"comune_id": ""}, {"comune_id": comune_id}]}


def _combine_filters(*filters: dict) -> dict:
    """Combina più filtri con `$and` — utile quando serve applicare sia
    il filtro per comune (`comune_filter`) sia un vincolo aggiuntivo su
    `tipo` (vocabolario vs calendario). Verificato che `langchain-postgres`
    supporti l'annidamento `$and` con dentro un `$or` (non assunto)."""
    non_empty = [f for f in filters if f]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def vocabolario_filter(comune_id: str) -> dict:
    """Contenuto NON calendario (vocabolario, guide, moduli, servizi):
    dove cerchiamo la descrizione dell'oggetto. Esclude esplicitamente
    `tipo="calendario"` invece di includere positivamente "guide" — vedi
    il docstring del modulo per il perché."""
    return _combine_filters({"tipo": {"$ne": "calendario"}}, comune_filter(comune_id))


def calendario_filter(comune_id: str) -> dict:
    """Solo contenuto di calendario, con lo stesso filtro per comune
    (contenuto condiviso dell'area + specifico del comune, mai quello di
    un comune vicino)."""
    return _combine_filters({"tipo": "calendario"}, comune_filter(comune_id))


def build_question(object_description: str) -> str:
    """Frase completa da mandare all'LLM per la generazione finale,
    costruita a partire dalla sola descrizione dell'oggetto — quella
    usata da sola, senza wrapping, per il retrieval (vedi
    `answer_question`). Esposta a parte (non solo interna ad
    `answer_question`) perché `bot/notebooks/rag_eval.ipynb`
    reimplementa retrieval e generazione a mano per poter scambiare
    system prompt diversi, e deve costruire esattamente la stessa
    domanda usata in produzione, non una leggermente diversa per caso."""
    return f"Come smaltisco questo oggetto: {object_description}?"


def answer_question(
    llm: ChatOllama,
    vectorstore: PGVector,
    comune_id: str,
    object_description: str,
    language_code: str | None = None,
    k_vocabolario: int = 6,
    k_calendario: int = 4,
) -> str:
    """Recupera il contesto in due passaggi (vocabolario poi calendario,
    vedi il docstring del modulo) e genera una risposta con il modello
    Ollama, usando entrambi come contesto — così la risposta include
    sempre anche il giorno di passaggio quando pertinente, non solo la
    modalità di smaltimento.

    `object_description` è SOLO la descrizione dell'oggetto (es.
    "bottiglia di plastica"), già estratta da `vision.py`
    (`identify_object_from_photo`/`identify_object_from_text`) — non
    l'intera domanda dell'utente. Usarla così com'è per il retrieval,
    invece di una domanda completa tipo "Dove butto una bottiglia di
    plastica?", è più preciso: il vector store fa match sul contenuto
    (materiale/categoria dell'oggetto), non su come l'utente ha
    formulato la domanda — la formulazione (saluti, cortesie, la
    struttura stessa della domanda) è solo rumore per la ricerca per
    similarità. La domanda completa per la generazione finale (dove
    invece la formulazione naturale aiuta l'LLM) viene costruita qui
    dentro con `build_question`, non passata dal chiamante: un solo
    punto dove quella frase è definita.

    `language_code` (il language_code del client Telegram dell'utente,
    vedi `i18n.py`) determina la lingua della risposta, non del
    contesto recuperato — la conversione avviene nella stessa chiamata
    di generazione, non con una traduzione separata a parte."""
    vocabolario_docs = vectorstore.similarity_search(
        object_description, k=k_vocabolario, filter=vocabolario_filter(comune_id)
    )
    vocabolario_context = "\n\n".join(doc.page_content for doc in vocabolario_docs)

    # Query 2: il testo recuperato al passaggio 1 come query, non la
    # descrizione dell'oggetto — vedi il docstring del modulo. Se il
    # passaggio 1 non ha trovato nulla, non ha senso cercare nel
    # calendario con una query vuota.
    calendario_context = ""
    if vocabolario_context:
        calendario_docs = vectorstore.similarity_search(
            vocabolario_context, k=k_calendario, filter=calendario_filter(comune_id)
        )
        calendario_context = "\n\n".join(doc.page_content for doc in calendario_docs)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lingua=i18n.language_name(language_code),
        vocabolario_context=vocabolario_context or "Nessuna informazione trovata.",
        calendario_context=calendario_context
        or "Nessuna informazione di calendario trovata.",
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[{"type": "text", "text": build_question(object_description)}]
        ),
    ]
    response = llm.invoke(messages)
    return response.content
