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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Valore sentinella richiesto al modello quando non è possibile
# identificare un oggetto specifico (vedi il docstring del modulo):
# confrontato case-insensitive contro la risposta grezza, non un
# parsing più elaborato — sufficiente per questo scopo, e più robusto
# di aspettarsi un output vuoto o un formato strutturato (JSON, ecc.)
# da un modello che potrebbe non rispettarlo sempre alla lettera.
# Condiviso tra i due prompt (foto e testo): stesso significato in
# entrambi i contesti, non serve un valore diverso per ciascuno.
_NO_OBJECT_SENTINEL = "NESSUNO"

PHOTO_SYSTEM_PROMPT = f"""\
Identifica l'unico oggetto principale nell'immagine: il soggetto in primo piano della fotografia.

Restituisci solo il nome dell'oggetto, seguito da dettagli descrittivi concisi sulle sue caratteristiche visibili, come materiale, colore, tipo, quantità, forma, texture, finitura o tratti distintivi.

Non menzionare oggetti sullo sfondo, oggetti secondari, persone, scenario o contesto. Non fornire spiegazioni, punteggi di confidenza, introduzioni, etichette o altro testo aggiuntivo.

Se l'immagine non permette di identificare con sufficiente chiarezza un singolo oggetto principale (è sfocata, mostra più oggetti ugualmente in primo piano senza un soggetto chiaro, o non contiene nessun oggetto riconoscibile), restituisci esattamente la parola {_NO_OBJECT_SENTINEL}, senza altro testo.

Rispondi in lingua italiana.

Formato di output:
[nome oggetto], [caratteristiche visibili principali]
"""

PHOTO_QUESTION = "Qual è l'oggetto principale in questa immagine?"


def identify_object_from_photo(
    llm: ChatOllama, image_base64: str, mime_type: str
) -> str | None:
    """Restituisce una breve descrizione (in italiano) dell'oggetto
    principale nella foto, nel formato "[nome], [caratteristiche]" —
    oppure None se la foto non permette di identificarne uno con
    sufficiente chiarezza (vedi il docstring del modulo)."""
    messages = [
        SystemMessage(content=PHOTO_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": PHOTO_QUESTION},
                {"type": "image", "base64": image_base64, "mime_type": mime_type},
            ]
        ),
    ]
    response = llm.invoke(messages)
    result = response.content.strip()
    return None if result.upper() == _NO_OBJECT_SENTINEL else result


TEXT_SYSTEM_PROMPT = f"""\
Identifica l'oggetto di cui l'utente sta parlando nel messaggio seguente: l'oggetto che vuole smaltire, non il resto della frase.

Restituisci solo il nome dell'oggetto, seguito da eventuali dettagli identificativi utili (materiale, colore, tipo, quantità) presenti nel messaggio. Non inventare dettagli che il messaggio non fornisce.

Non menzionare saluti, formule di cortesia, o la formulazione della domanda stessa. Non fornire spiegazioni, introduzioni, punteggi di confidenza o altro testo aggiuntivo.

Se il messaggio non menziona nessun oggetto da smaltire (es. un saluto, una domanda generica, un messaggio non pertinente allo smaltimento rifiuti), restituisci esattamente la parola {_NO_OBJECT_SENTINEL}, senza altro testo.

Rispondi in lingua italiana anche se il messaggio dell'utente è scritto in un'altra lingua.

Formato di output:
[nome oggetto], [dettagli utili]
"""


def identify_object_from_text(llm: ChatOllama, text: str) -> str | None:
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
    messages = [
        SystemMessage(content=TEXT_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ]
    response = llm.invoke(messages)
    result = response.content.strip()
    return None if result.upper() == _NO_OBJECT_SENTINEL else result
