"""Modalità inline: usare il bot da **qualsiasi** chat, scrivendo
`@nomebot una tazzina rotta` senza aggiungerlo a nulla.

È la risposta di Telegram al "vorrei chiederlo mentre parlo con
qualcun altro": funziona in una conversazione privata con un'altra
persona, in un gruppo dove il bot non è stato aggiunto, in un canale.
Il risultato viene inviato come un normale messaggio dell'utente, quindi
l'altra persona vede la risposta senza dover interagire col bot.

## Quale comune?

Una query inline non ha una chat propria: `chat_id` non esiste. Ha però
`from_user.id`, l'identificativo di chi scrive — che in una chat privata
**coincide con il `chat_id`** di quella conversazione. Il profilo si
cerca quindi con quell'id, e un utente che ha fatto l'onboarding nella
chat privata col bot si ritrova già configurato ovunque, senza doverlo
rifare.

Chi non l'ha mai fatto riceve un unico risultato che spiega di aprire la
chat col bot e mandare `/start`: la modalità inline non ha modo di
condurre un onboarding (niente bottoni, niente posizione), quindi
l'unica cosa onesta è indirizzare lì.

## Il vincolo che conta: il tempo

Telegram si aspetta una risposta a una query inline in pochi secondi, e
il client smette di attendere ben prima che una generazione completa
finisca (identificazione + retrieval + eventuale tool: nell'ordine dei
dieci secondi con un modello locale).

La modalità inline funziona quindi **a conferma**, non a battitura: il
risultato compare solo quando l'utente smette di scrivere abbastanza a
lungo, e la generazione parte una volta sola grazie alla cache di
Telegram (`cache_time` + `is_personal`). Non si prova a rispondere a
ogni carattere digitato — sarebbe una generazione buttata per ogni
tasto premuto, su una GPU che ne regge poche in parallelo.

`auto_pagination` non serve: si restituisce sempre un solo risultato.
"""

from __future__ import annotations

import asyncio

from bordeus_common import db as bot_db
from bordeus_common.log import get_logger
from telegram import (
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
)
from telegram.ext import ContextTypes

from . import i18n, identify, rag

logger = get_logger("bordeus_bot")

# Query più corta di così è quasi sempre una parola a metà: rispondere
# vorrebbe dire generare per ogni tasto premuto.
MIN_QUERY_LEN = 4

# Secondi per cui Telegram tiene il risultato. `is_personal=True` lo
# rende per-utente: due persone diverse con lo stesso testo hanno comuni
# diversi e non devono condividere la risposta.
CACHE_SECONDS = 300


def _articolo(
    id_risultato: str, titolo: str, descrizione: str, messaggio: str
) -> InlineQueryResultArticle:
    return InlineQueryResultArticle(
        id=id_risultato,
        title=titolo,
        description=descrizione,
        input_message_content=InputTextMessageContent(messaggio),
    )


async def handle_inline_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    inline_query = update.inline_query
    if inline_query is None:
        return

    testo = (inline_query.query or "").strip()
    language_code = (
        inline_query.from_user.language_code if inline_query.from_user else None
    )
    service = context.bot_data["service"]

    if len(testo) < MIN_QUERY_LEN:
        await inline_query.answer(
            [
                _articolo(
                    "hint",
                    i18n.t("inline_hint_title", language_code),
                    i18n.t("inline_hint_description", language_code),
                    i18n.t("inline_hint_description", language_code),
                )
            ],
            cache_time=CACHE_SECONDS,
            is_personal=False,
        )
        return

    # `from_user.id` coincide con il chat_id della chat privata fra
    # utente e bot: chi ha fatto l'onboarding lì è già configurato qui.
    user_id = inline_query.from_user.id

    def _load() -> bot_db.UserProfile | None:
        conn = bot_db.connect_light(service.config.database_url)
        try:
            return bot_db.get_user_profile(conn, user_id)
        finally:
            conn.close()

    profile = await asyncio.to_thread(_load)

    if profile is None or not profile.onboarded or not profile.comune_id:
        await inline_query.answer(
            [
                _articolo(
                    "onboarding",
                    i18n.t("inline_not_onboarded_title", language_code),
                    i18n.t("inline_not_onboarded_description", language_code),
                    i18n.t("inline_not_onboarded_description", language_code),
                )
            ],
            cache_time=CACHE_SECONDS,
            is_personal=True,
        )
        return

    try:
        oggetto = await asyncio.to_thread(
            identify.identify_object_from_text,
            service.llm,
            testo,
            i18n.language_name(language_code),
        )
        if oggetto is None:
            await inline_query.answer(
                [
                    _articolo(
                        "sconosciuto",
                        i18n.t("inline_not_recognized_title", language_code),
                        i18n.t("object_not_recognized_text", language_code),
                        i18n.t("object_not_recognized_text", language_code),
                    )
                ],
                cache_time=CACHE_SECONDS,
                is_personal=True,
            )
            return

        def _lavora():
            vectorstore = service.vectorstore_for_comune(profile.comune_id)
            return rag.answer_question(
                service.llm,
                vectorstore,
                profile.comune_id,
                oggetto.descrizione,
                tools=service.tools_for_comune(profile.comune_id),
                language_code=language_code,
            )

        async with service.llm_slots:
            risposta = await asyncio.to_thread(_lavora)
        # Testo semplice: un risultato inline viene inviato come
        # messaggio dell'utente, e l'anteprima nell'elenco dei risultati
        # non renderizza markup.
        testo = risposta.testo
        if risposta.fonti:
            chiave = "fonti_label" if len(risposta.fonti) > 1 else "fonte_label"
            righe = [i18n.t(chiave, language_code)]
            righe += [f"• {f}" for f in risposta.fonti]
            testo = testo + "\n\n" + "\n".join(righe)
    except Exception as exc:
        # Una query inline non ha una chat in cui scusarsi: l'unica cosa
        # possibile è non offrire risultati (il client mostra "nessun
        # risultato") e lasciare traccia nei log.
        logger.error("query inline fallita: %s", exc)
        return

    await inline_query.answer(
        [
            _articolo(
                "risposta",
                i18n.t(
                    "inline_answer_title", language_code, oggetto=oggetto.etichetta
                ),
                risposta.testo.split("\n")[0],
                testo,
            )
        ],
        cache_time=CACHE_SECONDS,
        is_personal=True,
    )
