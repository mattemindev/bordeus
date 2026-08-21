"""Messaggi multilingua del bot — italiano, francese, inglese, spagnolo,
tedesco.

La lingua è presa da `update.effective_user.language_code` —
l'impostazione del client Telegram dell'utente, non rilevata dal testo
del messaggio. Più affidabile di un rilevamento automatico su testi
brevi (una descrizione di due parole non basta a un language detector),
e funziona anche per le foto, dove non c'è testo da analizzare.

Pensato soprattutto per i turisti: la Valle d'Aosta ne accoglie molti,
spesso più a lungo dei residenti con l'italiano come lingua madre — un
turista tedesco o inglese deve poter usare il bot come un residente
italiano, comprese le risposte generate dall'LLM (vedi `rag.py`,
`language_name()` + l'istruzione nel system prompt), non solo i
messaggi statici di questo modulo.
"""

from __future__ import annotations

# Il bot è pensato per la Valle d'Aosta: l'italiano è il fallback
# naturale se il client Telegram non dichiara una lingua, o dichiara una
# lingua che non abbiamo tradotto qui sotto.
DEFAULT_LANGUAGE = "it"

SUPPORTED_LANGUAGES = ("it", "fr", "en", "es", "de")

# Nomi delle lingue in italiano, usati per istruire l'LLM (vedi rag.py,
# language_name()) a rispondere nella lingua dell'utente — il system
# prompt del bot è scritto in italiano, un nome in italiano ("rispondi
# in tedesco") è più coerente con quel contesto di un codice ISO nudo,
# anche se i modelli moderni capirebbero comunque "rispondi in 'de'".
_LANGUAGE_NAMES = {
    "it": "italiano",
    "fr": "francese",
    "en": "inglese",
    "es": "spagnolo",
    "de": "tedesco",
}

_MESSAGES: dict[str, dict[str, str]] = {
    "location_button": {
        "it": "📍 Condividi posizione",
        "fr": "📍 Partager la position",
        "en": "📍 Share location",
        "es": "📍 Compartir ubicación",
        "de": "📍 Standort teilen",
    },
    "confirm_yes_button": {
        "it": "✅ Sì, è corretto",
        "fr": "✅ Oui, c'est correct",
        "en": "✅ Yes, that's correct",
        "es": "✅ Sí, es correcto",
        "de": "✅ Ja, das stimmt",
    },
    "confirm_no_button": {
        "it": "❌ No, riprova",
        "fr": "❌ Non, réessayer",
        "en": "❌ No, try again",
        "es": "❌ No, inténtalo de nuevo",
        "de": "❌ Nein, nochmal versuchen",
    },
    "start_prompt": {
        "it": "Ciao! Per dirti come smaltire un oggetto devo sapere il tuo comune.\n\nTocca il bottone qui sotto per condividere la posizione, oppure scrivimi direttamente il nome del comune.",
        "fr": "Bonjour ! Pour te dire comment jeter un objet, j'ai besoin de connaître ta commune.\n\nAppuie sur le bouton ci-dessous pour partager ta position, ou écris-moi directement le nom de ta commune.",
        "en": "Hi! To tell you how to dispose of something, I need to know your comune (municipality).\n\nTap the button below to share your location, or just type the name of your comune.",
        "es": "¡Hola! Para decirte cómo desechar un objeto, necesito saber tu municipio.\n\nToca el botón de abajo para compartir tu ubicación, o escríbeme directamente el nombre de tu municipio.",
        "de": "Hallo! Um dir zu sagen, wie du etwas entsorgen kannst, muss ich deine Gemeinde kennen.\n\nTippe unten auf die Schaltfläche, um deinen Standort zu teilen, oder schreib mir einfach den Namen deiner Gemeinde.",
    },
    "need_start": {
        "it": "Prima di iniziare, manda /start.",
        "fr": "Avant de commencer, envoie /start.",
        "en": "Before we start, send /start.",
        "es": "Antes de empezar, envía /start.",
        "de": "Bevor wir loslegen, sende /start.",
    },
    "need_location_or_name": {
        "it": "Condividi la posizione o scrivimi il nome del tuo comune per iniziare.",
        "fr": "Partage ta position ou écris-moi le nom de ta commune pour commencer.",
        "en": "Share your location or type the name of your comune to get started.",
        "es": "Comparte tu ubicación o escríbeme el nombre de tu municipio para empezar.",
        "de": "Teile deinen Standort oder schreib mir den Namen deiner Gemeinde, um loszulegen.",
    },
    "need_object": {
        "it": "Mandami una descrizione testuale o una foto dell'oggetto da buttare.",
        "fr": "Envoie-moi une description ou une photo de l'objet à jeter.",
        "en": "Send me a text description or a photo of the item you want to dispose of.",
        "es": "Envíame una descripción o una foto del objeto que quieres desechar.",
        "de": "Schick mir eine Beschreibung oder ein Foto des Gegenstands, den du entsorgen möchtest.",
    },
    "object_not_recognized_text": {
        "it": 'Non ho capito quale oggetto vuoi buttare. Prova a descriverlo in modo più specifico (es. "bottiglia di plastica") o mandami una foto.',
        "fr": "Je n'ai pas compris quel objet tu veux jeter. Essaie de le décrire plus précisément (ex. « bouteille en plastique ») ou envoie-moi une photo.",
        "en": 'I couldn\'t tell which item you want to dispose of. Try describing it more specifically (e.g. "plastic bottle") or send me a photo.',
        "es": 'No he entendido qué objeto quieres desechar. Intenta describirlo con más detalle (p. ej. "botella de plástico") o envíame una foto.',
        "de": 'Ich habe nicht verstanden, welchen Gegenstand du entsorgen möchtest. Versuch, ihn genauer zu beschreiben (z. B. "Plastikflasche") oder schick mir ein Foto.',
    },
    "object_not_recognized_photo": {
        "it": "Non sono riuscito a riconoscere con chiarezza un oggetto specifico in questa foto. Prova a scattarne un'altra più a fuoco o da un'altra angolazione, oppure descrivimi l'oggetto a parole.",
        "fr": "Je n'ai pas réussi à reconnaître clairement un objet précis sur cette photo. Essaie d'en prendre une autre, plus nette ou sous un autre angle, ou décris-moi l'objet avec des mots.",
        "en": "I couldn't clearly recognize a specific item in this photo. Try taking another one that's sharper or from a different angle, or describe the item in words.",
        "es": "No he podido reconocer con claridad un objeto específico en esta foto. Intenta tomar otra más nítida o desde otro ángulo, o descríbeme el objeto con palabras.",
        "de": "Ich konnte in diesem Foto keinen bestimmten Gegenstand klar erkennen. Versuch ein schärferes Foto oder eines aus einem anderen Winkel, oder beschreib mir den Gegenstand in Worten.",
    },
    "geocode_failed": {
        "it": "Non sono riuscito a determinare il comune dalla posizione. Prova a scrivermi il nome del comune direttamente.",
        "fr": "Je n'ai pas réussi à déterminer la commune à partir de la position. Essaie de m'écrire directement le nom de la commune.",
        "en": "I couldn't determine your comune from the location. Try typing the name of your comune directly.",
        "es": "No he podido determinar el municipio a partir de la ubicación. Intenta escribirme directamente el nombre del municipio.",
        "de": "Ich konnte die Gemeinde anhand des Standorts nicht ermitteln. Versuch, mir den Namen der Gemeinde direkt zu schreiben.",
    },
    "generic_error": {
        "it": "Si è verificato un errore, riprova tra poco.",
        "fr": "Une erreur s'est produite, réessaie dans un instant.",
        "en": "Something went wrong, please try again in a moment.",
        "es": "Se ha producido un error, inténtalo de nuevo en un momento.",
        "de": "Es ist ein Fehler aufgetreten, versuch es gleich noch einmal.",
    },
    "comune_not_supported": {
        "it": '"{nome}" non è tra i comuni che al momento supporto (questo è ancora un proof of concept). Riprova con un altro comune, oppure scrivimi il nome esatto.',
        "fr": "« {nome} » ne fait pas partie des communes actuellement prises en charge (c'est encore un prototype). Réessaie avec une autre commune, ou écris-moi le nom exact.",
        "en": '"{nome}" isn\'t among the comuni I currently support (this is still a proof of concept). Try another comune, or type the exact name.',
        "es": '"{nome}" no está entre los municipios que actualmente admito (esto sigue siendo una prueba de concepto). Inténtalo con otro municipio, o escríbeme el nombre exacto.',
        "de": '"{nome}" gehört nicht zu den Gemeinden, die ich derzeit unterstütze (das ist noch ein Proof of Concept). Versuch es mit einer anderen Gemeinde oder schreib mir den genauen Namen.',
    },
    "confirm_prompt_area_gestore": {
        "it": "Ho trovato: {comune}, che fa parte del {area}, gestito da {gestore}.\n\nÈ corretto?",
        "fr": "J'ai trouvé : {comune}, qui fait partie de {area}, gérée par {gestore}.\n\nEst-ce correct ?",
        "en": "I found: {comune}, part of {area}, managed by {gestore}.\n\nIs this correct?",
        "es": "He encontrado: {comune}, que forma parte de {area}, gestionado por {gestore}.\n\n¿Es correcto?",
        "de": "Ich habe gefunden: {comune}, Teil von {area}, verwaltet von {gestore}.\n\nIst das richtig?",
    },
    "confirm_prompt_area_only": {
        "it": "Ho trovato: {comune}, che fa parte del {area}.\n\nÈ corretto?",
        "fr": "J'ai trouvé : {comune}, qui fait partie de {area}.\n\nEst-ce correct ?",
        "en": "I found: {comune}, part of {area}.\n\nIs this correct?",
        "es": "He encontrado: {comune}, que forma parte de {area}.\n\n¿Es correcto?",
        "de": "Ich habe gefunden: {comune}, Teil von {area}.\n\nIst das richtig?",
    },
    "confirm_prompt_no_area": {
        "it": "Ho trovato: {comune}, che è supportato, anche se non ho informazioni sulla sua area di gestione.\n\nÈ corretto?",
        "fr": "J'ai trouvé : {comune}, qui est prise en charge, même si je n'ai pas d'informations sur sa zone de gestion.\n\nEst-ce correct ?",
        "en": "I found: {comune}, which is supported, though I don't have information about its management area.\n\nIs this correct?",
        "es": "He encontrado: {comune}, que está admitido, aunque no tengo información sobre su área de gestión.\n\n¿Es correcto?",
        "de": "Ich habe gefunden: {comune}, das unterstützt wird, auch wenn ich keine Informationen über sein Verwaltungsgebiet habe.\n\nIst das richtig?",
    },
    "no_pending_confirmation": {
        "it": "Nessuna conferma in sospeso. Manda /start per ricominciare.",
        "fr": "Aucune confirmation en attente. Envoie /start pour recommencer.",
        "en": "No confirmation pending. Send /start to begin again.",
        "es": "No hay ninguna confirmación pendiente. Envía /start para volver a empezar.",
        "de": "Keine ausstehende Bestätigung. Sende /start, um neu zu beginnen.",
    },
    "confirm_no_response": {
        "it": "Ok, riprova: condividi la posizione o scrivimi il nome del tuo comune.",
        "fr": "D'accord, réessaie : partage ta position ou écris-moi le nom de ta commune.",
        "en": "Okay, try again: share your location or type the name of your comune.",
        "es": "De acuerdo, inténtalo de nuevo: comparte tu ubicación o escríbeme el nombre de tu municipio.",
        "de": "Okay, versuch es noch einmal: Teile deinen Standort oder schreib mir den Namen deiner Gemeinde.",
    },
    "confirm_yes_response": {
        "it": 'Perfetto, comune impostato su {nome}. Ora mandami una foto o descrivi l\'oggetto che vuoi buttare (es. "lattina di alluminio") e ti dico come conferirlo.\n\nPer cambiare comune in futuro, manda /comune.',
        "fr": "Parfait, commune définie sur {nome}. Envoie-moi maintenant une photo ou décris l'objet que tu veux jeter (ex. « canette en aluminium ») et je te dirai comment le trier.\n\nPour changer de commune plus tard, envoie /comune.",
        "en": 'Great, comune set to {nome}. Now send me a photo or describe the item you want to dispose of (e.g. "aluminum can") and I\'ll tell you how to sort it.\n\nTo change comune later, send /comune.',
        "es": 'Perfecto, municipio configurado en {nome}. Ahora envíame una foto o describe el objeto que quieres desechar (p. ej. "lata de aluminio") y te diré cómo hacerlo.\n\nPara cambiar de municipio más adelante, envía /comune.',
        "de": 'Perfekt, Gemeinde auf {nome} eingestellt. Schick mir jetzt ein Foto oder beschreib den Gegenstand, den du entsorgen möchtest (z. B. "Aludose"), und ich sage dir, wie du ihn entsorgst.\n\nUm später die Gemeinde zu ändern, sende /comune.',
    },
    "confirmation_error": {
        "it": "Si è verificato un errore, riprova con /start.",
        "fr": "Une erreur s'est produite, réessaie avec /start.",
        "en": "Something went wrong, please try again with /start.",
        "es": "Se ha producido un error, inténtalo de nuevo con /start.",
        "de": "Es ist ein Fehler aufgetreten, versuch es erneut mit /start.",
    },
    "your_comune_fallback": {
        "it": "il tuo comune",
        "fr": "ta commune",
        "en": "your comune",
        "es": "tu municipio",
        "de": "deine Gemeinde",
    },
    "photo_download_failed": {
        "it": "Non sono riuscito a scaricare la foto, riprova.",
        "fr": "Je n'ai pas réussi à télécharger la photo, réessaie.",
        "en": "I couldn't download the photo, please try again.",
        "es": "No he podido descargar la foto, inténtalo de nuevo.",
        "de": "Ich konnte das Foto nicht herunterladen, versuch es noch einmal.",
    },
    "photo_analysis_failed": {
        "it": "Non sono riuscito ad analizzare la foto, riprova.",
        "fr": "Je n'ai pas réussi à analyser la photo, réessaie.",
        "en": "I couldn't analyze the photo, please try again.",
        "es": "No he podido analizar la foto, inténtalo de nuevo.",
        "de": "Ich konnte das Foto nicht analysieren, versuch es noch einmal.",
    },
    "waiting_text": {
        "it": "🔎 Un attimo, sto controllando le guide del tuo comune...",
        "fr": "🔎 Un instant, je consulte les guides de votre commune...",
        "en": "🔎 One moment, checking your municipality's guides...",
        "es": "🔎 Un momento, estoy revisando las guías de tu municipio...",
        "de": "🔎 Einen Moment, ich prüfe die Unterlagen deiner Gemeinde...",
    },
    "waiting_photo": {
        "it": "📷 Un attimo, sto analizzando la foto...",
        "fr": "📷 Un instant, j'analyse la photo...",
        "en": "📷 One moment, analyzing the photo...",
        "es": "📷 Un momento, estoy analizando la foto...",
        "de": "📷 Einen Moment, ich analysiere das Foto...",
    },
}


def normalize_language(language_code: str | None) -> str:
    """'en-US' -> 'en': ci interessa solo la lingua, non la variante
    regionale. Restituisce sempre uno dei valori in SUPPORTED_LANGUAGES
    (mai None o una stringa non tradotta) — chi chiama `t()`/
    `language_name()` non deve gestire un fallback separato, lo fa già
    questa funzione."""
    lang = (language_code or "").split("-")[0].lower()
    return lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def t(message_id: str, language_code: str | None, **kwargs: str) -> str:
    """Messaggio tradotto per `message_id` nella lingua del client
    Telegram dell'utente, con fallback all'italiano se la lingua non è
    tra quelle tradotte (o non dichiarata). `kwargs` vengono interpolati
    nel template con `str.format()` (es. `t("comune_not_supported", lc,
    nome="Cogne")`)."""
    lang = normalize_language(language_code)
    template = _MESSAGES[message_id][lang]
    return template.format(**kwargs) if kwargs else template


def language_name(language_code: str | None) -> str:
    """Nome della lingua in italiano (es. "tedesco"), da inserire
    nell'istruzione al modello — vedi `rag.py`, dove il system prompt
    (scritto in italiano) chiede di rispondere in questa lingua."""
    return _LANGUAGE_NAMES[normalize_language(language_code)]
