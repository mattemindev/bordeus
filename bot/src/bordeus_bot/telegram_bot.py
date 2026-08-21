"""Handler del bot Telegram: onboarding (posizione o nome comune, con
conferma esplicita del Sub-ATO/gestore risolto), descrizione testuale o
foto di un oggetto -> risposta RAG.

Le chiamate bloccanti (Postgres, geocoding, LLM/vision via Ollama) girano
in un thread separato (`asyncio.to_thread`) per non bloccare l'event
loop di python-telegram-bot — non è async "vero" fino in fondo (nessuna
di queste librerie ha un client asyncio nativo), ma evita che una
richiesta lenta blocchi tutte le altre conversazioni in corso.
"""

from __future__ import annotations

import asyncio
import base64
import logging

from bordeus_common import db as bot_db
from bordeus_common.vectorstore import get_vectorstore
from langchain_ollama import ChatOllama
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import ContextTypes

from . import geocode, i18n, identify, rag
from .config import Config

logger = logging.getLogger("bordeus_bot")

_CONFIRM_YES = "confirm_comune_yes"
_CONFIRM_NO = "confirm_comune_no"


def _language_code(update: Update) -> str | None:
    """`language_code` del client Telegram dell'utente (impostazione
    del client, non rilevata dal testo — vedi `i18n.py`), o None se
    `effective_user` non è disponibile (caso raro, es. alcuni tipi di
    update senza un mittente diretto)."""
    return update.effective_user.language_code if update.effective_user else None


def _location_keyboard(language_code: str | None) -> ReplyKeyboardMarkup:
    """Bottone nativo Telegram che condivide la posizione GPS in un tap,
    senza che l'utente debba scrivere nulla."""
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
    rag.comune_filter."""

    def __init__(self, config: Config, embeddings) -> None:
        self.config = config
        self.embeddings = embeddings
        self.llm = ChatOllama(
            model=config.ollama_model, base_url=config.ollama_base_url, temperature=0.2
        )
        self._vectorstores: dict[str, object] = {}

        # Applica le migration condivise una volta sola all'avvio, non
        # ad ogni messaggio (le query per-richiesta usano connect_light).
        bot_db.connect(config.database_url).close()

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
    await update.message.reply_text(
        i18n.t("start_prompt", language_code),
        reply_markup=_location_keyboard(language_code),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Router unico per posizione/testo/foto, in base allo stato di
    onboarding del profilo: finché il comune non è confermato, ogni
    messaggio serve a determinarlo (o a ri-tentare, se una risoluzione
    precedente non è stata confermata)."""
    if update.message is None:
        return
    chat_id = update.effective_chat.id
    service = _service(context)

    def _load_profile() -> bot_db.UserProfile | None:
        conn = bot_db.connect_light(service.config.database_url)
        try:
            return bot_db.get_user_profile(conn, chat_id)
        finally:
            conn.close()

    profile = await asyncio.to_thread(_load_profile)
    language_code = _language_code(update)

    if profile is None:
        await update.message.reply_text(i18n.t("need_start", language_code))
        return

    if not profile.onboarded:
        if update.message.location is not None:
            await _handle_location_onboarding(update, context)
        elif update.message.text:
            await _onboard_by_comune_name(update, context, update.message.text)
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
        await _handle_text(update, context, profile, update.message.text)
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
    except Exception as exc:
        logger.error("gestione conferma onboarding fallita: %s", exc)
        try:
            await query.edit_message_text(i18n.t("confirmation_error", language_code))
        except Exception as edit_exc:
            # Se anche questo fallisce (es. messaggio già cancellato o
            # troppo vecchio per essere modificato), non c'è altro da
            # fare qui: l'errore originale è già stato loggato sopra.
            logger.error("anche il messaggio di errore è fallito: %s", edit_exc)


async def _handle_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    profile: bot_db.UserProfile,
    text: str,
) -> None:
    service = _service(context)
    language_code = _language_code(update)

    await update.message.reply_text(i18n.t("waiting_text", language_code))

    def _answer() -> str | None:
        description = identify.identify_object_from_text(service.llm, text)
        if description is None:
            return None
        vectorstore = service.vectorstore_for_comune(profile.comune_id)
        return rag.answer_question(
            service.llm,
            vectorstore,
            profile.comune_id,
            description,
            language_code=language_code,
        )

    try:
        answer = await asyncio.to_thread(_answer)
    except Exception as exc:
        logger.error("query RAG fallita: %s", exc)
        await update.message.reply_text(i18n.t("generic_error", language_code))
        return

    if answer is None:
        await update.message.reply_text(
            i18n.t("object_not_recognized_text", language_code)
        )
        return

    await update.message.reply_text(answer)


async def _handle_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE, profile: bot_db.UserProfile
) -> None:
    service = _service(context)
    # Telegram invia più risoluzioni della stessa foto: l'ultima è la
    # più grande.
    largest = update.message.photo[-1]
    language_code = _language_code(update)

    await update.message.reply_text(i18n.t("waiting_photo", language_code))

    try:
        file = await largest.get_file()
        image_bytes = await file.download_as_bytearray()
    except Exception as exc:
        logger.error("download foto fallito: %s", exc)
        await update.message.reply_text(i18n.t("photo_download_failed", language_code))
        return

    image_base64 = base64.b64encode(bytes(image_bytes)).decode("ascii")
    mime_type = (
        "image/jpeg"  # Telegram converte sempre le foto inviate come "photo" in JPEG
    )

    def _identify_and_answer() -> str | None:
        description = identify.identify_object_from_photo(
            service.llm, image_base64, mime_type
        )
        if description is None:
            return None
        logger.info("oggetto identificato: %s", description)
        vectorstore = service.vectorstore_for_comune(profile.comune_id)
        return rag.answer_question(
            service.llm,
            vectorstore,
            profile.comune_id,
            description,
            language_code=language_code,
        )

    try:
        answer = await asyncio.to_thread(_identify_and_answer)
    except Exception as exc:
        logger.error("identificazione/risposta fallita: %s", exc)
        await update.message.reply_text(i18n.t("photo_analysis_failed", language_code))
        return

    if answer is None:
        await update.message.reply_text(
            i18n.t("object_not_recognized_photo", language_code)
        )
        return

    await update.message.reply_text(answer)
