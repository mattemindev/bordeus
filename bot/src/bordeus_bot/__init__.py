"""bordeus_bot: bot Telegram di bordeus.

Onboarding del comune (posizione o nome), identificazione dell'oggetto
da una foto (vision, via Ollama), risposta RAG sulle guide di
smaltimento del comune (retrieval su Postgres/pgvector via
`langchain-postgres`, generazione via Ollama). Riusa `Embeddings` e
`PGVector` da `bordeus_common` (dipendenza di workspace — vedi
../pyproject.toml), lo stesso codice usato dalla pipeline di ingestion:
un bot non ha bisogno di dipendere da `bordeus_ingest` (fetch/loaders/
chunking sono concetti solo dell'ingestion), solo del sottoinsieme
condiviso.
"""

from __future__ import annotations

from bordeus_common.log import TRACE, get_logger, setup_logging

# Gli import pesanti (telegram.ext, bordeus_common.embed -> torch, e i
# moduli del bot che ne dipendono) stanno DENTRO `main()` e
# `_pubblica_comandi()`, non qui.
#
# A livello di modulo obbligherebbero chiunque importi un sottomodulo a
# caricare l'intera catena: `bordeus_bot.i18n` è un dizionario di
# stringhe, e importarlo non deve tirarsi dietro torch e il vector
# store. È anche ciò che permette ai test di girare con la sola
# python-telegram-bot, quindi in pochi secondi su un runner senza GPU
# (vedi bot/tests/conftest.py).

__all__ = ["main"]


logger = get_logger("bordeus_bot")


async def _pubblica_comandi(application) -> None:
    """Registra i comandi nel menu di Telegram, con elenchi diversi per
    chat private e gruppi.

    Senza questo l'unico modo di scoprire /rifiuti è leggere la
    documentazione: nei gruppi il menu dei comandi è di fatto l'unica
    superficie di scoperta che il bot ha, visto che non può commentare
    da solo. /start e /comune non compaiono nell'elenco dei gruppi
    perché lì il comune si imposta una volta e non è un'azione da
    proporre a ogni membro."""

    from telegram import (
        BotCommand,
        BotCommandScopeAllGroupChats,
        BotCommandScopeAllPrivateChats,
    )

    privati = [
        BotCommand("comune", "Imposta o cambia il tuo comune"),
        BotCommand("help", "Come si usa"),
    ]
    gruppi = [
        BotCommand("rifiuti", "Chiedi come smaltire un oggetto"),
        BotCommand("comune", "Imposta il comune di questo gruppo"),
        BotCommand("help", "Come si usa"),
    ]
    try:
        await application.bot.set_my_commands(
            privati, scope=BotCommandScopeAllPrivateChats()
        )
        await application.bot.set_my_commands(
            gruppi, scope=BotCommandScopeAllGroupChats()
        )
    except Exception as exc:
        # Puramente cosmetico: il bot funziona anche senza il menu.
        logger.warning("pubblicazione dei comandi fallita: %s", exc)


def main() -> None:
    # Import locali: vedi la nota in cima al modulo. Tenerli qui evita
    # che importare `bordeus_bot.i18n` carichi torch e il vector store.
    from bordeus_common.embed import get_embeddings
    from dotenv import load_dotenv
    from telegram.ext import (
        ApplicationBuilder,
        CallbackQueryHandler,
        CommandHandler,
        InlineQueryHandler,
        MessageHandler,
        filters,
    )

    from . import config as config_module
    from . import inline, telegram_bot
    from .rag import QUERY_INSTRUCTION

    load_dotenv()

    # LOG_LEVEL accetta anche TRACE (5), che registra i prompt di sistema
    # completi, il contesto recuperato e le chiamate agli strumenti: è il
    # livello del "cosa ha visto esattamente il modello". Contiene il
    # contenuto dei messaggi degli utenti, quindi è per il debug locale,
    # non per un'istanza che serve persone vere.
    livello = setup_logging()
    if livello <= TRACE:
        logger.warning(
            "log a livello TRACE: i prompt di sistema e il contenuto dei "
            "messaggi degli utenti finiranno nei log. Non lasciarlo attivo "
            "in esercizio."
        )

    cfg = config_module.load()

    # Stesso modello usato in ingestion (EMBEDDING_MODEL, o il default
    # del pacchetto se non impostata) — deve restare lo stesso: gli
    # embedding devono vivere nello stesso spazio vettoriale dei chunk
    # già scritti. query_instruction attiva il prefisso istruzione solo
    # lato query (vedi rag.py).
    embeddings = get_embeddings(
        cfg.embedding_model, query_instruction=QUERY_INSTRUCTION
    )

    service = telegram_bot.Service(cfg, embeddings)

    application = ApplicationBuilder().token(cfg.telegram_token).build()
    application.bot_data["service"] = service

    application.add_handler(CommandHandler("start", telegram_bot.handle_start))
    application.add_handler(CommandHandler("comune", telegram_bot.handle_start))
    application.add_handler(CommandHandler("help", telegram_bot.handle_help))
    # /rifiuti <oggetto>: la via che funziona in un gruppo anche con la
    # modalità privacy attiva (default di BotFather), che nasconde al
    # bot i messaggi normali ma non i comandi.
    application.add_handler(CommandHandler("rifiuti", telegram_bot.handle_rifiuti))
    application.add_handler(
        CallbackQueryHandler(
            telegram_bot.handle_confirmation, pattern="^confirm_comune_"
        )
    )
    application.add_handler(
        CallbackQueryHandler(
            telegram_bot.handle_correzione, pattern="^correggi_oggetto$"
        )
    )
    application.add_handler(
        MessageHandler(
            filters.LOCATION | filters.PHOTO | (filters.TEXT & ~filters.COMMAND),
            telegram_bot.handle_message,
        )
    )
    # Modalità inline: `@nomebot una tazzina rotta` da qualunque chat,
    # anche dove il bot non è stato aggiunto. Va abilitata una volta con
    # /setinline su @BotFather, altrimenti Telegram non invia mai questi
    # update.
    application.add_handler(InlineQueryHandler(inline.handle_inline_query))

    application.post_init = _pubblica_comandi

    logger.info(
        "bot avviato (modello Ollama: %s, base_url: %s)",
        cfg.ollama_model,
        cfg.ollama_base_url,
    )
    application.run_polling()


if __name__ == "__main__":
    main()
