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

from bordeus_common.log import get_logger

logger = get_logger("bordeus_bot")

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
    # Usata dal gestore d'errore di handle_confirmation. Mancava: `t()`
    # sollevava KeyError proprio mentre stava gestendo un errore, quindi
    # l'utente non vedeva niente e nel log compariva "anche il messaggio
    # di errore è fallito" senza il motivo vero.
    "confirmation_error": {
        "it": "Qualcosa è andato storto nel salvare il comune. Riprova con /comune.",
        "fr": "Un problème est survenu lors de l'enregistrement de la commune. Réessaie avec /comune.",
        "en": "Something went wrong while saving your municipality. Try again with /comune.",
        "es": "Algo salió mal al guardar el municipio. Inténtalo de nuevo con /comune.",
        "de": "Beim Speichern der Gemeinde ist etwas schiefgelaufen. Versuch es erneut mit /comune.",
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
    # Solo la conferma, nient'altro: le istruzioni su cosa fare adesso
    # stanno in "ready_message"/"group_ready", che è il messaggio subito
    # successivo (deve esistere comunque, per togliere la tastiera della
    # posizione — vedi telegram_bot.handle_confirmation). Averle in
    # entrambi significava mandare due volte di fila la stessa cosa.
    "confirm_yes_response": {
        "it": "✅ Comune impostato: {nome}.",
        "fr": "✅ Commune définie : {nome}.",
        "en": "✅ Municipality set: {nome}.",
        "es": "✅ Municipio configurado: {nome}.",
        "de": "✅ Gemeinde eingestellt: {nome}.",
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
    # --- messaggio di attesa, aggiornato a fasi -----------------------
    # Descrivono cosa succede per l'utente, non nel programma: niente
    # "retrieval", "embedding" o nomi di modelli. Vedi ui.Progress.
    "waiting_text": {
        "it": "📖 Sto leggendo il tuo messaggio...",
        "fr": "📖 Je lis ton message...",
        "en": "📖 Reading your message...",
        "es": "📖 Estoy leyendo tu mensaje...",
        "de": "📖 Ich lese deine Nachricht...",
    },
    "waiting_photo": {
        "it": "📷 Sto guardando la foto...",
        "fr": "📷 Je regarde la photo...",
        "en": "📷 Looking at the photo...",
        "es": "📷 Estoy mirando la foto...",
        "de": "📷 Ich schaue mir das Foto an...",
    },
    # Mostra l'oggetto riconosciuto: è l'informazione più utile durante
    # l'attesa, perché è anche il punto in cui il bot può sbagliare — se
    # ha capito male, l'utente lo vede subito invece che dopo una
    # risposta inutile.
    "waiting_identified": {
        "it": "🔎 Ho capito: {oggetto}\nCerco come si smaltisce a {comune}...",
        "fr": "🔎 J'ai compris : {oggetto}\nJe cherche comment le jeter à {comune}...",
        "en": "🔎 Got it: {oggetto}\nLooking up how to dispose of it in {comune}...",
        "es": "🔎 Entendido: {oggetto}\nBuscando cómo desecharlo en {comune}...",
        "de": "🔎 Verstanden: {oggetto}\nIch suche, wie man das in {comune} entsorgt...",
    },
    "waiting_identified_no_comune": {
        "it": "🔎 Ho capito: {oggetto}\nCerco come si smaltisce...",
        "fr": "🔎 J'ai compris : {oggetto}\nJe cherche comment le jeter...",
        "en": "🔎 Got it: {oggetto}\nLooking up how to dispose of it...",
        "es": "🔎 Entendido: {oggetto}\nBuscando cómo desecharlo...",
        "de": "🔎 Verstanden: {oggetto}\nIch suche, wie man das entsorgt...",
    },
    # --- fine onboarding: toglie la tastiera della posizione ----------
    "ready_message": {
        "it": "Tutto pronto. 👋\n\nMandami la foto di un oggetto, oppure scrivimi cos'è (es. \"una tazzina rotta\"), e ti dico dove va e quando passa la raccolta.\n\nPer cambiare comune: /comune",
        "fr": "Tout est prêt. 👋\n\nEnvoie-moi la photo d'un objet, ou écris-moi ce que c'est (ex. « une tasse cassée »), et je te dis où le jeter et quand a lieu la collecte.\n\nPour changer de commune : /comune",
        "en": "All set. 👋\n\nSend me a photo of an object, or just describe it (e.g. \"a broken mug\"), and I'll tell you where it goes and when it's collected.\n\nTo change municipality: /comune",
        "es": "Todo listo. 👋\n\nMándame la foto de un objeto, o escríbeme qué es (p. ej. «una taza rota»), y te digo dónde va y cuándo pasa la recogida.\n\nPara cambiar de municipio: /comune",
        "de": "Alles bereit. 👋\n\nSchick mir ein Foto eines Gegenstands oder beschreib ihn (z. B. „eine kaputte Tasse\"), und ich sage dir, wohin er gehört und wann die Abholung ist.\n\nGemeinde ändern: /comune",
    },
    # --- aiuto e uso nei gruppi ---------------------------------------
    "help_text": {
        "it": "Ti dico come smaltire un oggetto nel tuo comune, in Valle d'Aosta.\n\n• Mandami una **foto** dell'oggetto\n• Oppure scrivimi cos'è: \"una tazzina rotta\"\n\nComandi:\n/comune — imposta o cambia il comune\n/rifiuti <oggetto> — chiedi in un gruppo\n/help — questo messaggio\n\nIn un gruppo, scrivimi menzionandomi oppure usa /rifiuti.\nIn qualsiasi chat puoi scrivere @{bot} seguito dall'oggetto.",
        "fr": "Je te dis comment jeter un objet dans ta commune, en Vallée d'Aoste.\n\n• Envoie-moi une **photo** de l'objet\n• Ou écris ce que c'est : « une tasse cassée »\n\nCommandes :\n/comune — définir ou changer de commune\n/rifiuti <objet> — demander dans un groupe\n/help — ce message\n\nDans un groupe, mentionne-moi ou utilise /rifiuti.\nDans n'importe quelle discussion, tape @{bot} suivi de l'objet.",
        "en": "I tell you how to dispose of something in your municipality, in Aosta Valley.\n\n• Send me a **photo** of the object\n• Or describe it: \"a broken mug\"\n\nCommands:\n/comune — set or change your municipality\n/rifiuti <object> — ask in a group\n/help — this message\n\nIn a group, mention me or use /rifiuti.\nIn any chat, type @{bot} followed by the object.",
        "es": "Te digo cómo desechar un objeto en tu municipio, en el Valle de Aosta.\n\n• Mándame una **foto** del objeto\n• O escribe qué es: «una taza rota»\n\nComandos:\n/comune — fijar o cambiar de municipio\n/rifiuti <objeto> — preguntar en un grupo\n/help — este mensaje\n\nEn un grupo, menciónarme o usa /rifiuti.\nEn cualquier chat, escribe @{bot} seguido del objeto.",
        "de": "Ich sage dir, wie du etwas in deiner Gemeinde im Aostatal entsorgst.\n\n• Schick mir ein **Foto** des Gegenstands\n• Oder beschreib ihn: „eine kaputte Tasse\"\n\nBefehle:\n/comune — Gemeinde festlegen oder ändern\n/rifiuti <Gegenstand> — in einer Gruppe fragen\n/help — diese Nachricht\n\nErwähne mich in einer Gruppe oder nutze /rifiuti.\nIn jedem Chat: @{bot} gefolgt vom Gegenstand.",
    },
    "group_start_prompt": {
        "it": "Ciao! Per rispondere in questo gruppo devo sapere di quale comune si parla.\n\nScrivimi il nome del comune (es. \"Donnas\").",
        "fr": "Bonjour ! Pour répondre dans ce groupe, j'ai besoin de savoir de quelle commune il s'agit.\n\nÉcris-moi le nom de la commune (ex. « Donnas »).",
        "en": "Hi! To answer in this group I need to know which municipality it refers to.\n\nType the name of the comune (e.g. \"Donnas\").",
        "es": "¡Hola! Para responder en este grupo necesito saber de qué municipio se trata.\n\nEscríbeme el nombre del municipio (p. ej. «Donnas»).",
        "de": "Hallo! Um in dieser Gruppe zu antworten, muss ich wissen, um welche Gemeinde es geht.\n\nSchreib mir den Namen der Gemeinde (z. B. „Donnas\").",
    },
    # Non ripete il nome del comune: l'ha appena detto la conferma. Dice
    # invece l'unica cosa che in un gruppo non è ovvia — che il bot
    # risponde solo se interpellato.
    "group_ready": {
        "it": "In un gruppo rispondo solo se mi interpelli: menzionami, oppure usa /rifiuti <oggetto>.",
        "fr": "Dans un groupe, je réponds seulement si tu m'interpelles : mentionne-moi, ou utilise /rifiuti <objet>.",
        "en": "In a group I only reply when addressed: mention me, or use /rifiuti <object>.",
        "es": "En un grupo solo respondo si me interpelas: menciónarme, o usa /rifiuti <objeto>.",
        "de": "In einer Gruppe antworte ich nur, wenn du mich ansprichst: erwähne mich oder nutze /rifiuti <Gegenstand>.",
    },
    "rifiuti_needs_argument": {
        "it": "Scrivi cosa vuoi smaltire dopo il comando, per esempio:\n/rifiuti una tazzina rotta",
        "fr": "Écris ce que tu veux jeter après la commande, par exemple :\n/rifiuti une tasse cassée",
        "en": "Write what you want to dispose of after the command, for example:\n/rifiuti a broken mug",
        "es": "Escribe qué quieres desechar después del comando, por ejemplo:\n/rifiuti una taza rota",
        "de": "Schreib nach dem Befehl, was du entsorgen willst, zum Beispiel:\n/rifiuti eine kaputte Tasse",
    },
    # --- modalità inline (@bot ... da qualsiasi chat) ------------------
    "inline_hint_title": {
        "it": "Scrivi cosa vuoi smaltire",
        "fr": "Écris ce que tu veux jeter",
        "en": "Type what you want to dispose of",
        "es": "Escribe qué quieres desechar",
        "de": "Schreib, was du entsorgen willst",
    },
    "inline_hint_description": {
        "it": "Per esempio: una tazzina rotta, una bottiglia di plastica...",
        "fr": "Par exemple : une tasse cassée, une bouteille en plastique...",
        "en": "For example: a broken mug, a plastic bottle...",
        "es": "Por ejemplo: una taza rota, una botella de plástico...",
        "de": "Zum Beispiel: eine kaputte Tasse, eine Plastikflasche...",
    },
    "inline_not_onboarded_title": {
        "it": "Prima devi dirmi il tuo comune",
        "fr": "Dis-moi d'abord ta commune",
        "en": "First tell me your municipality",
        "es": "Primero dime tu municipio",
        "de": "Sag mir zuerst deine Gemeinde",
    },
    "inline_not_onboarded_description": {
        "it": "Apri la chat con me e manda /start",
        "fr": "Ouvre la discussion avec moi et envoie /start",
        "en": "Open the chat with me and send /start",
        "es": "Abre el chat conmigo y envía /start",
        "de": "Öffne den Chat mit mir und sende /start",
    },
    "inline_answer_title": {
        "it": "Come smaltire: {oggetto}",
        "fr": "Comment jeter : {oggetto}",
        "en": "How to dispose of: {oggetto}",
        "es": "Cómo desechar: {oggetto}",
        "de": "Entsorgung von: {oggetto}",
    },
    "inline_not_recognized_title": {
        "it": "Non ho riconosciuto un oggetto",
        "fr": "Je n'ai pas reconnu d'objet",
        "en": "I didn't recognise an object",
        "es": "No he reconocido ningún objeto",
        "de": "Ich habe keinen Gegenstand erkannt",
    },
    # --- fonte e correzione dell'oggetto ------------------------------
    "fonte_label": {
        "it": "Fonte:",
        "fr": "Source :",
        "en": "Source:",
        "es": "Fuente:",
        "de": "Quelle:",
    },
    "fonti_label": {
        "it": "Fonti:",
        "fr": "Sources :",
        "en": "Sources:",
        "es": "Fuentes:",
        "de": "Quellen:",
    },
    "correggi_button": {
        "it": "✏️ Non è questo",
        "fr": "✏️ Ce n'est pas ça",
        "en": "✏️ That's not it",
        "es": "✏️ No es esto",
        "de": "✏️ Das ist es nicht",
    },
    "correggi_prompt": {
        "it": "Scrivimi tu cos'è, con parole tue — userò esattamente quello che scrivi, senza provare a interpretarlo.",
        "fr": "Dis-moi toi ce que c'est, avec tes mots — j'utiliserai exactement ce que tu écris, sans essayer de l'interpréter.",
        "en": "Tell me what it is in your own words — I'll use exactly what you write, without trying to interpret it.",
        "es": "Dime tú qué es, con tus palabras — usaré exactamente lo que escribas, sin intentar interpretarlo.",
        "de": "Sag mir mit deinen Worten, was es ist — ich nehme genau das, was du schreibst, ohne es zu deuten.",
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
    voce = _MESSAGES.get(message_id)
    if voce is None:
        # Non solleva: `t()` viene chiamata anche dentro i gestori
        # d'errore, e un KeyError lì sostituisce un problema
        # diagnosticabile con un silenzio (l'utente non riceve niente e
        # il log mostra solo il fallimento del messaggio d'errore). Meglio
        # un testo generico e una riga di log che dica quale chiave manca.
        logger.error("chiave di traduzione mancante: %r", message_id)
        voce = _MESSAGES["generic_error"]
        kwargs = {}
    template = voce.get(lang) or voce[DEFAULT_LANGUAGE]
    return template.format(**kwargs) if kwargs else template


def language_name(language_code: str | None) -> str:
    """Nome della lingua in italiano (es. "tedesco"), da inserire
    nell'istruzione al modello — vedi `rag.py`, dove il system prompt
    (scritto in italiano) chiede di rispondere in questa lingua."""
    return _LANGUAGE_NAMES[normalize_language(language_code)]
