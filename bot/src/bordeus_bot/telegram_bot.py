"""Handler del bot Telegram: onboarding (posizione o nome comune, con
conferma esplicita del Sub-ATO/gestore risolto), descrizione testuale o
foto di un oggetto -> risposta RAG.

## Chat private e gruppi

Il profilo è sempre legato al `chat_id`, quindi un gruppo ha il proprio
comune, indipendente da quello dei suoi membri: è la cosa giusta per un
gruppo di paese, dove la domanda "dove butto questo?" ha una sola
risposta valida per tutti. Nessuna modifica allo schema: un gruppo è
semplicemente un'altra riga in `users`.

Due differenze di comportamento nei gruppi:

- **Il bot risponde solo se interpellato** — menzione, risposta a un suo
  messaggio, o `/rifiuti`. Un bot che commenta ogni messaggio di un
  gruppo viene rimosso dal gruppo.
- **Niente tastiera della posizione.** Una reply keyboard in un gruppo
  comparirebbe a tutti i membri, e la posizione di chi tocca il bottone
  non è necessariamente quella di cui parla il gruppo. Lì l'onboarding
  passa dal nome del comune.

## La tastiera della posizione compare solo quando serve

`ReplyKeyboardMarkup` resta attaccata alla chat finché non viene tolta
esplicitamente: `one_time_keyboard` la fa solo collassare, non
sparire. Lasciarla significa che il bottone "condividi posizione"
resta nel menu degli allegati per sempre, suggerendo un'azione che dopo
l'onboarding non serve più e che, se usata, farebbe ripartire la scelta
del comune.

Viene quindi mostrata solo durante l'onboarding e su `/comune`, e
rimossa con `ReplyKeyboardRemove` appena il comune è confermato. La
rimozione non può viaggiare su `edit_message_text` (le reply keyboard
non si possono allegare a una modifica), quindi arriva con il messaggio
di "tutto pronto" subito dopo la conferma.

Le chiamate bloccanti (Postgres, geocoding, LLM/vision via Ollama) girano
in un thread separato (`asyncio.to_thread`) per non bloccare l'event
loop di python-telegram-bot — non è async "vero" fino in fondo (nessuna
di queste librerie ha un client asyncio nativo), ma evita che una
richiesta lenta blocchi tutte le altre conversazioni in corso.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import html
import logging

from bordeus_common import db as bot_db
from bordeus_common.vectorstore import get_vectorstore
from langchain_ollama import ChatOllama
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ChatType
from telegram.ext import ContextTypes

from . import calendario, geocode, i18n, identify, rag, ui
from .config import Config

logger = logging.getLogger("bordeus_bot")

_CONFIRM_YES = "confirm_comune_yes"
_CONFIRM_NO = "confirm_comune_no"
_CORREGGI = "correggi_oggetto"


def is_group(update: Update) -> bool:
    chat = update.effective_chat
    return chat is not None and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)


def addressed_to_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """In un gruppo, il bot risponde solo se interpellato: menzione
    esplicita, oppure risposta a un suo messaggio.

    In chat privata è sempre interpellato, quindi la funzione è vera
    per costruzione. I comandi non passano di qui: hanno i loro handler,
    e funzionano anche con la modalità privacy attiva (che è il
    default di BotFather e impedisce al bot di vedere i messaggi
    normali del gruppo — vedi bot/README.md)."""
    if not is_group(update):
        return True

    message = update.message
    if message is None:
        return False

    risposta_al_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )
    if risposta_al_bot:
        return True

    username = (context.bot.username or "").lower()
    testo = (message.text or message.caption or "").lower()
    return bool(username) and f"@{username}" in testo


def strip_mention(testo: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Toglie la menzione dal testo prima di usarlo come descrizione
    dell'oggetto: "@bordeus_bot una tazzina rotta" deve arrivare
    all'identificazione come "una tazzina rotta", altrimenti il nome del
    bot diventa parte di ciò che si sta cercando di smaltire."""
    username = context.bot.username
    if not username:
        return testo.strip()
    return testo.replace(f"@{username}", "").replace(f"@{username.lower()}", "").strip()


def _language_code(update: Update) -> str | None:
    """`language_code` del client Telegram dell'utente (impostazione
    del client, non rilevata dal testo — vedi `i18n.py`), o None se
    `effective_user` non è disponibile (caso raro, es. alcuni tipi di
    update senza un mittente diretto)."""
    return update.effective_user.language_code if update.effective_user else None


def _location_keyboard(language_code: str | None) -> ReplyKeyboardMarkup:
    """Bottone nativo Telegram che condivide la posizione GPS in un tap,
    senza che l'utente debba scrivere nulla.

    Mostrato SOLO durante l'onboarding e su /comune, e solo in chat
    privata: vedi il docstring del modulo."""
    return ReplyKeyboardMarkup(
        [
            [
                KeyboardButton(
                    i18n.t("location_button", language_code), request_location=True
                )
            ]
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def _correzione_keyboard(language_code: str | None) -> InlineKeyboardMarkup:
    """Bottone sotto la risposta per dire "hai capito l'oggetto
    sbagliato".

    L'identificazione è il punto in cui il bot sbaglia più spesso, ed è
    anche quello in cui l'utente se ne accorge subito, perché il
    messaggio di attesa gli ha appena mostrato cosa aveva capito. Senza
    il bottone, correggere significa riscrivere tutto sperando di essere
    più fortunati; con il bottone, la descrizione la fornisce
    direttamente l'utente e il passaggio di identificazione viene
    saltato del tutto — quindi il secondo tentativo non può sbagliare
    allo stesso modo."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t("correggi_button", language_code), callback_data=_CORREGGI
                )
            ]
        ]
    )


def _con_fonti(risposta: rag.Risposta, language_code: str | None) -> str:
    """Risposta convertita in HTML, con le fonti elencate in fondo.

    Una fonte per riga, con l'URL agganciato al nome invece che stampato
    per esteso: un link a un PDF del gestore occupa due righe di testo e
    rende illeggibile la coda del messaggio, mentre il nome della
    pubblicazione è l'informazione che serve davvero.

    Composto in Python dai metadata dei chunk recuperati e dagli
    strumenti invocati, mai chiesto al modello: un URL generato token per
    token è un URL che prima o poi viene inventato, e una fonte sbagliata
    è peggio di nessuna fonte."""
    corpo = ui.to_html(risposta.testo)
    if not risposta.fonti:
        return corpo

    chiave = "fonti_label" if len(risposta.fonti) > 1 else "fonte_label"
    righe = [i18n.t(chiave, language_code)]
    for fonte in risposta.fonti:
        nome = html.escape(fonte.nome, quote=False)
        if fonte.url:
            righe.append(f'• <a href="{html.escape(fonte.url)}">{nome}</a>')
        else:
            righe.append(f"• {nome}")
    return corpo + "\n\n" + "\n".join(righe)


def _confirmation_keyboard(language_code: str | None) -> InlineKeyboardMarkup:
    """Bottoni inline (non una reply keyboard: restano legati al
    messaggio specifico, spariscono/si disabilitano dopo il tap invece
    di restare visibili sulla tastiera)."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t("confirm_yes_button", language_code),
                    callback_data=_CONFIRM_YES,
                ),
                InlineKeyboardButton(
                    i18n.t("confirm_no_button", language_code),
                    callback_data=_CONFIRM_NO,
                ),
            ]
        ]
    )


class Service:
    """Stato condiviso tra gli handler: configurazione, LLM, vector
    store per area Sub-ATO (cache — costruito alla prima richiesta di
    quell'area, poi riusato; più comuni condividono la stessa area,
    quindi anche lo stesso vector store cachato). Il filtro per comune
    specifico (contenuto condiviso + specifico dell'utente, mai quello
    di un comune vicino) si applica per singola query, non qui — vedi
    rag.comune_filter.

    Il tool del calendario NON è cachato qui, a differenza del vector
    store: ha il comune dell'utente legato nella chiusura (vedi
    `calendario.make_tool`), quindi un tool riusato fra conversazioni
    risponderebbe a un utente con il calendario di un altro. Il vector
    store si può condividere perché è per area e il filtro per comune si
    applica alla singola query; il tool no, e la differenza è
    deliberata."""

    def __init__(self, config: Config, embeddings) -> None:
        self.config = config
        self.embeddings = embeddings
        self.llm = ChatOllama(
            model=config.ollama_model, base_url=config.ollama_base_url, temperature=0.2
        )
        self._vectorstores: dict[str, object] = {}
        self._nomi_comune: dict[str, str] = {}

        # Il modello gira in locale su una sola GPU: le richieste
        # concorrenti non vanno più veloci, si contendono la stessa VRAM
        # e rallentano tutte. Il semaforo le mette in fila invece di
        # lasciarle accavallare — conta ora che il bot può stare in un
        # gruppo, dove più persone scrivono insieme.
        self.llm_slots = asyncio.Semaphore(config.max_richieste_parallele)

        # Applica le migration condivise una volta sola all'avvio, non
        # ad ogni messaggio (le query per-richiesta usano connect_light).
        bot_db.connect(config.database_url).close()

    def nome_comune(self, comune_id: str | None) -> str | None:
        """Nome leggibile del comune, cachato in memoria. Serve solo ai
        messaggi ("cerco come si smaltisce a Donnas"): è un dato che per
        un profilo non cambia mai, e una query a Postgres per ogni
        messaggio di attesa sarebbe sprecata. Chiamata bloccante alla
        prima richiesta di un comune: va dentro un `asyncio.to_thread`
        come il resto."""
        if not comune_id:
            return None
        if comune_id not in self._nomi_comune:
            conn = bot_db.connect_light(self.config.database_url)
            try:
                comune = bot_db.get_comune(conn, comune_id)
            finally:
                conn.close()
            if comune is None:
                return None
            self._nomi_comune[comune_id] = comune.nome
        return self._nomi_comune[comune_id]

    def vectorstore_for_comune(self, comune_id: str):
        """Risolve il comune -> area Sub-ATO -> vector store dell'area
        (cachato per area, non per comune: chiamata bloccante, va
        invocata dentro un `asyncio.to_thread`, non direttamente
        nell'event loop). Il chiamante applica poi il filtro per comune
        alla query specifica (`rag.answer_question`, non qui) — comuni
        diversi della stessa area condividono lo stesso vector store
        cachato ma hanno filtri diversi."""
        conn = bot_db.connect_light(self.config.database_url)
        try:
            comune = bot_db.get_comune(conn, comune_id)
        finally:
            conn.close()

        if comune is None or comune.sub_ato_id is None:
            raise ValueError(
                f"il comune {comune_id!r} non ha un'area Sub-ATO assegnata "
                "(dato mancante o non ancora ingerito)"
            )

        area_id = comune.sub_ato_id
        if area_id not in self._vectorstores:
            self._vectorstores[area_id] = get_vectorstore(
                self.config.database_url, area_id, self.embeddings
            )
        return self._vectorstores[area_id]

    def tools_for_comune(self, comune_id: str, hamlet: str = "") -> list:
        """Costruisce gli strumenti per QUESTA richiesta, legati al
        comune dell'utente. Chiamata bloccante (interroga Postgres per
        le categorie disponibili), va invocata dentro un
        `asyncio.to_thread`.

        `hamlet` resta vuoto finché l'onboarding non risolve la frazione
        dell'utente: `users` non ha ancora una colonna per la frazione, e
        `bordeus_common.calendario` ricade da sola sul calendario del
        comune intero quando la frazione non è indicata. Il parametro
        c'è già perché il giorno in cui l'onboarding la risolverà (via
        reverse geocoding OSM, che restituisce l'hamlet nella stessa
        risposta già usata per il comune) non ci sia altro da cambiare
        qui."""
        return [
            calendario.make_tool(self.config.database_url, comune_id, hamlet=hamlet)
        ]


def _service(context: ContextTypes.DEFAULT_TYPE) -> Service:
    return context.application.bot_data["service"]


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start e /comune: riportano sempre l'utente alla scelta del
    comune (azzerando anche un'eventuale conferma in sospeso), così è
    anche il modo per "cambiare comune" senza bisogno di comandi
    dedicati."""
    chat_id = update.effective_chat.id
    service = _service(context)

    def _reset() -> None:
        conn = bot_db.connect_light(service.config.database_url)
        try:
            bot_db.save_user_profile(
                conn,
                bot_db.UserProfile(
                    chat_id=chat_id,
                    comune_id=None,
                    onboarded=False,
                    pending_comune_id=None,
                ),
            )
        finally:
            conn.close()

    await asyncio.to_thread(_reset)

    language_code = _language_code(update)

    if is_group(update):
        # Niente tastiera della posizione in un gruppo: comparirebbe a
        # tutti i membri, e la posizione di chi tocca il bottone non è
        # necessariamente quella di cui parla il gruppo.
        await update.message.reply_text(i18n.t("group_start_prompt", language_code))
        return

    await update.message.reply_text(
        i18n.t("start_prompt", language_code),
        reply_markup=_location_keyboard(language_code),
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        i18n.t("help_text", _language_code(update), bot=context.bot.username or "bot")
    )


async def handle_rifiuti(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/rifiuti <oggetto>: la via esplicita per chiedere in un gruppo.

    Esiste perché la modalità privacy di BotFather è attiva per
    default e impedisce al bot di vedere i messaggi normali di un
    gruppo: i comandi passano comunque, le menzioni no. Con un comando
    il bot funziona in un gruppo senza dover chiedere di disattivare la
    privacy — che è una modifica invasiva (il bot vedrebbe tutto)."""
    testo = " ".join(context.args).strip() if context.args else ""
    language_code = _language_code(update)

    if not testo:
        await update.message.reply_text(
            i18n.t("rifiuti_needs_argument", language_code)
        )
        return

    profile = await _load_profile(update, context)
    if profile is None or not profile.onboarded:
        await update.message.reply_text(
            i18n.t("group_start_prompt" if is_group(update) else "need_start", language_code)
        )
        return

    await _handle_text(update, context, profile, testo)


async def _load_profile(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bot_db.UserProfile | None:
    chat_id = update.effective_chat.id
    service = _service(context)

    def _load() -> bot_db.UserProfile | None:
        conn = bot_db.connect_light(service.config.database_url)
        try:
            return bot_db.get_user_profile(conn, chat_id)
        finally:
            conn.close()

    return await asyncio.to_thread(_load)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Router unico per posizione/testo/foto, in base allo stato di
    onboarding del profilo: finché il comune non è confermato, ogni
    messaggio serve a determinarlo (o a ri-tentare, se una risoluzione
    precedente non è stata confermata)."""
    if update.message is None:
        return

    # In un gruppo si risponde solo se interpellati: un bot che commenta
    # ogni messaggio viene rimosso dal gruppo.
    if not addressed_to_bot(update, context):
        return

    profile = await _load_profile(update, context)
    language_code = _language_code(update)

    if profile is None:
        await update.message.reply_text(i18n.t("need_start", language_code))
        return

    if not profile.onboarded:
        if update.message.location is not None:
            await _handle_location_onboarding(update, context)
        elif update.message.text:
            await _onboard_by_comune_name(
                update, context, strip_mention(update.message.text, context)
            )
        else:
            await update.message.reply_text(
                i18n.t("need_location_or_name", language_code)
            )
        return

    if update.message.location is not None:
        # Già onboardato ma manda di nuovo la posizione: la trattiamo
        # come richiesta implicita di ricontrollare/cambiare comune.
        await _handle_location_onboarding(update, context)
    elif update.message.photo:
        await _handle_photo(update, context, profile)
    elif update.message.text:
        await _handle_text(
            update, context, profile, strip_mention(update.message.text, context)
        )
    else:
        await update.message.reply_text(i18n.t("need_object", language_code))


async def _handle_location_onboarding(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    loc = update.message.location

    def _geocode() -> str:
        return geocode.reverse_geocode(
            loc.latitude,
            loc.longitude,
        )

    try:
        nome = await asyncio.to_thread(_geocode)
    except Exception as exc:
        logger.warning("reverse geocoding fallito: %s", exc)
        await update.message.reply_text(
            i18n.t("geocode_failed", _language_code(update))
        )
        return

    await _onboard_by_comune_name(update, context, nome)


async def _onboard_by_comune_name(
    update: Update, context: ContextTypes.DEFAULT_TYPE, nome: str
) -> None:
    """Risolve il nome in un comune supportato e, se trovato, chiede
    conferma esplicita (Sub-ATO + gestore) invece di completare subito
    l'onboarding — l'utente conferma con un tap sul bottone, gestito da
    `handle_confirmation`."""
    chat_id = update.effective_chat.id
    service = _service(context)

    def _resolve_and_set_pending() -> tuple[bot_db.Comune | None, bot_db.SubAto | None]:
        conn = bot_db.connect_light(service.config.database_url)
        try:
            comune = bot_db.resolve_comune_by_name(conn, nome)
            if comune is None:
                return None, None

            sub_ato = (
                bot_db.get_sub_ato(conn, comune.sub_ato_id)
                if comune.sub_ato_id
                else None
            )

            bot_db.save_user_profile(
                conn,
                bot_db.UserProfile(
                    chat_id=chat_id,
                    comune_id=None,
                    onboarded=False,
                    pending_comune_id=comune.id,
                ),
            )
            return comune, sub_ato
        finally:
            conn.close()

    try:
        comune, sub_ato = await asyncio.to_thread(_resolve_and_set_pending)
    except Exception as exc:
        logger.error("risoluzione comune fallita: %s", exc)
        await update.message.reply_text(i18n.t("generic_error", _language_code(update)))
        return

    language_code = _language_code(update)

    if comune is None:
        await update.message.reply_text(
            i18n.t("comune_not_supported", language_code, nome=nome)
        )
        return

    if sub_ato is not None and sub_ato.gestore:
        text = i18n.t(
            "confirm_prompt_area_gestore",
            language_code,
            comune=comune.nome,
            area=sub_ato.nome,
            gestore=sub_ato.gestore,
        )
    elif sub_ato is not None:
        text = i18n.t(
            "confirm_prompt_area_only",
            language_code,
            comune=comune.nome,
            area=sub_ato.nome,
        )
    else:
        # Comune registrato ma senza un'area assegnata (dato mancante):
        # capita raro, ma non deve bloccare la conferma — solo essere
        # meno specifica.
        text = i18n.t("confirm_prompt_no_area", language_code, comune=comune.nome)

    await update.message.reply_text(
        text, reply_markup=_confirmation_keyboard(language_code)
    )


async def handle_confirmation(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Gestisce il tap sui bottoni Sì/No della conferma di onboarding.

    A differenza degli altri handler, l'intero corpo è avvolto in un
    solo try/except: prima di questa correzione, qualunque eccezione
    (es. un problema di connessione a Postgres) risaliva in silenzio —
    nessun messaggio all'utente, bottone che sembra non fare nulla."""
    query = update.callback_query
    try:
        await (
            query.answer()
        )  # obbligatorio: toglie lo stato di "caricamento" dal bottone lato client
    except Exception as exc:
        # Non interrompiamo per questo: il caso più comune è un timeout
        # innocuo della UI del bottone, la conferma vera e propria può
        # comunque procedere.
        logger.warning("query.answer() fallita (proseguo comunque): %s", exc)

    chat_id = query.message.chat_id
    service = _service(context)
    language_code = _language_code(update)

    try:

        def _load_profile() -> bot_db.UserProfile | None:
            conn = bot_db.connect_light(service.config.database_url)
            try:
                return bot_db.get_user_profile(conn, chat_id)
            finally:
                conn.close()

        profile = await asyncio.to_thread(_load_profile)

        if profile is None or not profile.pending_comune_id:
            await query.edit_message_text(
                i18n.t("no_pending_confirmation", language_code)
            )
            return

        if query.data == _CONFIRM_NO:

            def _clear_pending() -> None:
                conn = bot_db.connect_light(service.config.database_url)
                try:
                    bot_db.save_user_profile(
                        conn,
                        bot_db.UserProfile(
                            chat_id=chat_id,
                            comune_id=profile.comune_id,
                            onboarded=False,
                            pending_comune_id=None,
                        ),
                    )
                finally:
                    conn.close()

            await asyncio.to_thread(_clear_pending)
            await query.edit_message_text(i18n.t("confirm_no_response", language_code))
            return

        # _CONFIRM_YES
        def _confirm() -> bot_db.Comune | None:
            conn = bot_db.connect_light(service.config.database_url)
            try:
                comune = bot_db.get_comune(conn, profile.pending_comune_id)
                bot_db.save_user_profile(
                    conn,
                    bot_db.UserProfile(
                        chat_id=chat_id,
                        comune_id=profile.pending_comune_id,
                        onboarded=True,
                        pending_comune_id=None,
                    ),
                )
                return comune
            finally:
                conn.close()

        comune = await asyncio.to_thread(_confirm)
        logger.info(
            "utente onboardato: chat_id=%s comune=%s",
            chat_id,
            comune.id if comune else "?",
        )

        nome = comune.nome if comune else i18n.t("your_comune_fallback", language_code)
        await query.edit_message_text(
            i18n.t("confirm_yes_response", language_code, nome=nome)
        )

        # La tastiera della posizione va tolta ORA che il comune è
        # confermato: `one_time_keyboard` la fa solo collassare, non
        # sparire, quindi il bottone "condividi posizione" resterebbe
        # per sempre nel menu degli allegati — suggerendo un'azione che
        # non serve più e che, se usata, farebbe ripartire la scelta del
        # comune.
        #
        # Deve viaggiare su un messaggio nuovo: una reply keyboard (e la
        # sua rimozione) non si può allegare a `edit_message_text`. Ne
        # approfittiamo per dire cosa fare adesso, che è comunque il
        # momento giusto in cui dirlo.
        in_gruppo = query.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
        try:
            if in_gruppo:
                await query.message.reply_text(i18n.t("group_ready", language_code))
            else:
                await query.message.reply_text(
                    i18n.t("ready_message", language_code),
                    reply_markup=ReplyKeyboardRemove(),
                )
        except Exception as exc:
            # Il comune è già salvato: se questo messaggio non parte,
            # l'utente ha comunque un bot funzionante, solo con la
            # tastiera ancora visibile.
            logger.warning("messaggio di 'tutto pronto' fallito: %s", exc)
    except Exception as exc:
        logger.error("gestione conferma onboarding fallita: %s", exc)
        try:
            await query.edit_message_text(i18n.t("confirmation_error", language_code))
        except Exception as edit_exc:
            # Se anche questo fallisce (es. messaggio già cancellato o
            # troppo vecchio per essere modificato), non c'è altro da
            # fare qui: l'errore originale è già stato loggato sopra.
            logger.error("anche il messaggio di errore è fallito: %s", edit_exc)


async def handle_correzione(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Tap su "Non è questo": il prossimo messaggio di testo di questo
    utente viene usato come descrizione dell'oggetto così com'è.

    Lo stato sta in `user_data` (per utente, non per chat): in un gruppo
    più persone possono avere una correzione in sospeso
    contemporaneamente, e legarla alla chat farebbe scambiare le
    correzioni fra membri diversi. È volutamente in memoria e non su
    Postgres — se il bot riparte, al massimo un utente riscrive la
    frase."""
    query = update.callback_query
    with contextlib.suppress(Exception):
        await query.answer()

    language_code = _language_code(update)
    context.user_data["correzione_in_attesa"] = True

    # Il bottone viene tolto dal messaggio: è stato usato, lasciarlo
    # farebbe pensare che si possa correggere più volte la stessa
    # risposta.
    with contextlib.suppress(Exception):
        await query.edit_message_reply_markup(reply_markup=None)

    await query.message.reply_text(i18n.t("correggi_prompt", language_code))


async def _handle_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    profile: bot_db.UserProfile,
    text: str,
) -> None:
    service = _service(context)
    language_code = _language_code(update)

    # Dopo un tap su "Non è questo", il testo dell'utente È la
    # descrizione: saltare l'identificazione è il punto del bottone,
    # altrimenti il secondo tentativo potrebbe sbagliare esattamente
    # come il primo.
    correzione = bool(context.user_data.pop("correzione_in_attesa", False))

    async with ui.Progress(
        update.message, i18n.t("waiting_text", language_code)
    ) as progress:
        if correzione:
            # La frase dell'utente vale sia come query sia come
            # etichetta: l'ha scritta lui, non c'è niente da tradurre.
            testo_pulito = text.strip()
            oggetto = identify.Oggetto(
                descrizione=testo_pulito, etichetta=testo_pulito
            )
            logger.info("descrizione fornita dall'utente: %s", testo_pulito)
        else:
            try:
                oggetto = await asyncio.to_thread(
                    identify.identify_object_from_text,
                    service.llm,
                    text,
                    i18n.language_name(language_code),
                )
            except Exception as exc:
                logger.error("identificazione da testo fallita: %s", exc)
                await progress.fail(i18n.t("generic_error", language_code))
                return

            if oggetto is None:
                await progress.fail(
                    i18n.t("object_not_recognized_text", language_code)
                )
                return

        await progress.update(
            _testo_identificato(service, profile, oggetto.etichetta, language_code)
        )

        try:
            answer = await _rispondi(
                service, profile, oggetto.descrizione, language_code
            )
        except Exception as exc:
            logger.error("query RAG fallita: %s", exc)
            await progress.fail(i18n.t("generic_error", language_code))
            return

        await progress.done(
            _con_fonti(answer, language_code),
            reply_markup=_correzione_keyboard(language_code),
        )


def _testo_identificato(
    service: Service,
    profile: bot_db.UserProfile,
    etichetta: str,
    language_code: str | None,
) -> str:
    """Fase centrale del messaggio di attesa: mostra cosa il bot ha
    capito, nella lingua dell'utente (`Oggetto.etichetta`, non la
    descrizione italiana che va al retrieval). Il nome del comune arriva da una cache in memoria — è un
    dato che non cambia mai per un profilo, e una query a Postgres solo
    per abbellire un messaggio di attesa sarebbe sprecata."""
    nome = service.nome_comune(profile.comune_id)
    if nome:
        return i18n.t(
            "waiting_identified", language_code, oggetto=etichetta, comune=nome
        )
    return i18n.t("waiting_identified_no_comune", language_code, oggetto=etichetta)


async def _rispondi(
    service: Service,
    profile: bot_db.UserProfile,
    descrizione: str,
    language_code: str | None,
) -> rag.Risposta:
    """Retrieval + generazione + eventuale tool calling, fuori
    dall'event loop e dietro il semaforo: vedi `Service.llm_slots`.

    L'attesa per lo slot avviene mentre l'utente vede già il messaggio
    "ho capito: X", quindi una coda di qualche secondo si legge come
    elaborazione in corso, non come bot bloccato."""

    def _lavora() -> rag.Risposta:
        vectorstore = service.vectorstore_for_comune(profile.comune_id)
        return rag.answer_question(
            service.llm,
            vectorstore,
            profile.comune_id,
            descrizione,
            tools=service.tools_for_comune(profile.comune_id),
            language_code=language_code,
        )

    async with service.llm_slots:
        return await asyncio.to_thread(_lavora)


async def _handle_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE, profile: bot_db.UserProfile
) -> None:
    service = _service(context)
    # Telegram invia più risoluzioni della stessa foto: l'ultima è la
    # più grande.
    largest = update.message.photo[-1]
    language_code = _language_code(update)

    async with ui.Progress(
        update.message, i18n.t("waiting_photo", language_code)
    ) as progress:
        try:
            file = await largest.get_file()
            image_bytes = await file.download_as_bytearray()
        except Exception as exc:
            logger.error("download foto fallito: %s", exc)
            await progress.fail(i18n.t("photo_download_failed", language_code))
            return

        image_base64 = base64.b64encode(bytes(image_bytes)).decode("ascii")
        # Telegram converte sempre in JPEG le immagini inviate come "photo".
        mime_type = "image/jpeg"

        try:
            oggetto = await asyncio.to_thread(
                identify.identify_object_from_photo,
                service.llm,
                image_base64,
                mime_type,
                i18n.language_name(language_code),
            )
        except Exception as exc:
            logger.error("identificazione da foto fallita: %s", exc)
            await progress.fail(i18n.t("photo_analysis_failed", language_code))
            return

        if oggetto is None:
            await progress.fail(
                i18n.t("object_not_recognized_photo", language_code)
            )
            return

        logger.info(
            "oggetto identificato: %r (etichetta: %r)",
            oggetto.descrizione,
            oggetto.etichetta,
        )
        await progress.update(
            _testo_identificato(service, profile, oggetto.etichetta, language_code)
        )

        try:
            answer = await _rispondi(
                service, profile, oggetto.descrizione, language_code
            )
        except Exception as exc:
            logger.error("risposta fallita: %s", exc)
            await progress.fail(i18n.t("photo_analysis_failed", language_code))
            return

        await progress.done(
            _con_fonti(answer, language_code),
            reply_markup=_correzione_keyboard(language_code),
        )
