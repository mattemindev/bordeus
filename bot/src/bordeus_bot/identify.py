"""Identificazione dell'oggetto di cui l'utente sta parlando — da una
foto (`identify_object_from_photo`) o da un messaggio di testo
(`identify_object_from_text`) — tramite un modello servito da Ollama
(`langchain_ollama.ChatOllama`).

Entrambe le funzioni restituiscono una descrizione BREVE dell'oggetto
(es. "bottiglia di plastica, trasparente"), non l'intera frase
dell'utente: usata così com'è per interrogare il vector store (vedi
`rag.answer_question`), non la domanda intera formulata dall'utente
("Dove butto questo?", "Come smaltisco una bottiglia di plastica?") —
quella formulazione aggiunge solo rumore alla ricerca per similarità,
il vector store deve fare match sul contenuto (materiale/categoria
dell'oggetto), non su come l'utente ha scritto la domanda.

Entrambe possono anche restituire None, quando non è stato possibile
identificare un oggetto specifico — un messaggio di testo fuori tema
(un saluto, una domanda generica) o una foto che non permette di
identificare con sufficiente chiarezza un singolo oggetto principale
(sfocata, più oggetti ugualmente in primo piano senza un soggetto
chiaro, nessun oggetto riconoscibile). Il chiamante decide come
comportarsi in quel caso — tipicamente chiedendo all'utente di
riformulare o di mandare un nuovo input, invece di fare una ricerca RAG
su una descrizione vuota o inventata.
"""

from __future__ import annotations

from dataclasses import dataclass

from bordeus_common.log import blocco, get_logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

logger = get_logger("bordeus_bot")

# Valore sentinella richiesto al modello quando non è possibile
# identificare un oggetto specifico (vedi il docstring del modulo):
# confrontato case-insensitive contro la risposta grezza, non un
# parsing più elaborato — sufficiente per questo scopo, e più robusto
# di aspettarsi un output vuoto o un formato strutturato (JSON, ecc.)
# da un modello che potrebbe non rispettarlo sempre alla lettera.
# Condiviso tra i due prompt (foto e testo): stesso significato in
# entrambi i contesti, non serve un valore diverso per ciascuno.
_NO_OBJECT_SENTINEL = "NESSUNO"


@dataclass(frozen=True)
class Oggetto:
    """Le due forme dello stesso oggetto, prodotte da una sola chiamata.

    `descrizione` è **sempre in italiano** ed è ciò che va al retrieval:
    il vocabolario è in italiano, e anche con un modello di embedding
    multilingua cercare nella stessa lingua dei documenti resta il caso
    migliore. `etichetta` è la stessa cosa nella lingua dell'utente, ed
    esiste solo per il messaggio "Ho capito: ...".

    Tenerle separate evita il compromesso fra le due: far rispondere
    l'identificazione nella lingua dell'utente peggiorerebbe il
    retrieval, e tradurre a parte costerebbe una seconda chiamata al
    modello per una riga di cortesia.
    """

    descrizione: str
    etichetta: str


def _parse_oggetto(grezzo: str) -> Oggetto | None:
    """Legge le due righe `IT:` / `LOC:` prodotte dai prompt.

    Tollerante di proposito: se il modello restituisce una riga sola (o
    ignora le etichette), quella vale per entrambe le forme. Un formato
    non rispettato alla lettera non deve far fallire l'identificazione,
    che è comunque riuscita."""
    testo = grezzo.strip()
    if not testo or testo.upper() == _NO_OBJECT_SENTINEL:
        return None

    descrizione = etichetta = ""
    for riga in testo.splitlines():
        pulita = riga.strip()
        if pulita.upper().startswith("IT:"):
            descrizione = pulita[3:].strip()
        elif pulita.upper().startswith("LOC:"):
            etichetta = pulita[4:].strip()

    if not descrizione:
        # Nessuna etichetta riconosciuta: prendiamo la prima riga non
        # vuota e la usiamo per entrambe.
        descrizione = next((r.strip() for r in testo.splitlines() if r.strip()), "")
    if not descrizione or descrizione.upper() == _NO_OBJECT_SENTINEL:
        return None

    return Oggetto(descrizione=descrizione, etichetta=etichetta or descrizione)

PHOTO_SYSTEM_PROMPT_TEMPLATE = f"""\
Identifica l'unico oggetto principale nell'immagine: il soggetto in primo piano della fotografia.

Restituisci il nome dell'oggetto e il materiale di cui è fatto, in modo BREVE: al massimo 6 parole.
Includi solo ciò che serve a capire come si smaltisce — materiale, e la forma dell'oggetto se cambia il conferimento (es. "bicchiere in vetro", "vaschetta in plastica").
NON descrivere colore, decorazioni, texture, finitura, riflessi o altri dettagli estetici: non cambiano dove va buttato l'oggetto.

Non menzionare oggetti sullo sfondo, oggetti secondari, persone, scenario o contesto. Non fornire spiegazioni, punteggi di confidenza, introduzioni o altro testo aggiuntivo.

Se l'immagine non permette di identificare con sufficiente chiarezza un singolo oggetto principale (è sfocata, mostra più oggetti ugualmente in primo piano senza un soggetto chiaro, o non contiene nessun oggetto riconoscibile), restituisci esattamente la parola {_NO_OBJECT_SENTINEL}, senza altro testo.

Formato di output, esattamente due righe e nient'altro:
IT: [oggetto e materiale, in italiano]
LOC: [la stessa cosa, in {{lingua}}]
"""

PHOTO_QUESTION = "Qual è l'oggetto principale in questa immagine?"


def identify_object_from_photo(
    llm: ChatOllama, image_base64: str, mime_type: str, lingua: str = "italiano"
) -> Oggetto | None:
    """Identifica l'oggetto principale nella foto, in italiano (per il
    retrieval) e nella lingua dell'utente (per il messaggio di attesa) —
    oppure None se la foto non permette di identificarne uno con
    sufficiente chiarezza (vedi il docstring del modulo)."""
    system_prompt = PHOTO_SYSTEM_PROMPT_TEMPLATE.format(lingua=lingua)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "image", "base64": image_base64, "mime_type": mime_type},
                {"type": "text", "text": PHOTO_QUESTION},
            ]
        ),
    ]
    logger.trace(  # type: ignore[attr-defined]
        "identificazione da foto: mime=%s, %d byte di base64%s",
        mime_type,
        len(image_base64),
        blocco("system prompt (foto)", system_prompt),
    )
    response = llm.invoke(messages)
    result = response.content.strip()
    logger.trace("identificazione da foto -> %r", result)  # type: ignore[attr-defined]
    return _parse_oggetto(result)


TEXT_SYSTEM_PROMPT_TEMPLATE = f"""\
Identifica l'oggetto di cui l'utente sta parlando nel messaggio seguente: l'oggetto che vuole smaltire, non il resto della frase.

Restituisci il nome dell'oggetto e il materiale, in modo BREVE: al massimo 6 parole. Includi solo i dettagli presenti nel messaggio che cambiano il modo di smaltirlo (materiale, quantità). Non inventare dettagli che il messaggio non fornisce, e non aggiungere descrizioni estetiche.

Non menzionare saluti, formule di cortesia, o la formulazione della domanda stessa. Non fornire spiegazioni, introduzioni, punteggi di confidenza o altro testo aggiuntivo.

Se il messaggio non menziona nessun oggetto da smaltire (es. un saluto, una domanda generica, un messaggio non pertinente allo smaltimento rifiuti), restituisci esattamente la parola {_NO_OBJECT_SENTINEL}, senza altro testo.

Formato di output, esattamente due righe e nient'altro:
IT: [oggetto e materiale, in italiano, anche se il messaggio è in un'altra lingua]
LOC: [la stessa cosa, in {{lingua}}]
"""


def identify_object_from_text(
    llm: ChatOllama, text: str, lingua: str = "italiano"
) -> Oggetto | None:
    """Estrae dal messaggio dell'utente solo la descrizione
    dell'oggetto da smaltire (es. da "dove butto una bottiglia di
    plastica?" -> "bottiglia di plastica"), scartando il resto della
    frase — stesso principio di `identify_object_from_photo`, ma per
    l'input testuale, dove il "rumore" da scartare è la formulazione
    della domanda invece dello sfondo della foto.

    Restituisce None se il messaggio non sembra menzionare nessun
    oggetto specifico da smaltire — il chiamante decide come comportarsi
    in quel caso (es. chiedere di riformulare, invece di fare una
    ricerca RAG su una descrizione vuota o su tutta la frase originale)."""
    system_prompt = TEXT_SYSTEM_PROMPT_TEMPLATE.format(lingua=lingua)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text),
    ]
    logger.trace(  # type: ignore[attr-defined]
        "identificazione da testo%s%s",
        blocco("system prompt (testo)", system_prompt),
        blocco("messaggio utente", text),
    )
    response = llm.invoke(messages)
    result = response.content.strip()
    logger.trace("identificazione da testo -> %r", result)  # type: ignore[attr-defined]
    return _parse_oggetto(result)
