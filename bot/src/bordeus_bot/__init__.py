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

import logging

from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, MessageHandler, filters
from bordeus_common.embed import get_embeddings

from . import config as config_module
from . import telegram_bot
from .rag import QUERY_INSTRUCTION

__all__ = ["config_module", "telegram_bot"]


def main() -> None:
    load_dotenv()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("bordeus_bot")

    cfg = config_module.load()

    # Stesso modello usato in ingestion (EMBEDDING_MODEL, o il default
    # del pacchetto se non impostata) — deve restare lo stesso: gli
    # embedding devono vivere nello stesso spazio vettoriale dei chunk
    # già scritti. query_instruction attiva il prefisso istruzione solo
    # lato query (vedi rag.py).
    embeddings = get_embeddings(cfg.embedding_model, query_instruction=QUERY_INSTRUCTION)

    service = telegram_bot.Service(cfg, embeddings)

    application = ApplicationBuilder().token(cfg.telegram_token).build()
    application.bot_data["service"] = service

    application.add_handler(CommandHandler("start", telegram_bot.handle_start))
    application.add_handler(CommandHandler("comune", telegram_bot.handle_start))
    application.add_handler(
        CallbackQueryHandler(telegram_bot.handle_confirmation, pattern="^confirm_comune_")
    )
    application.add_handler(
        MessageHandler(
            filters.LOCATION | filters.PHOTO | (filters.TEXT & ~filters.COMMAND),
            telegram_bot.handle_message,
        )
    )

    logger.info("bot avviato (modello Ollama: %s, base_url: %s)", cfg.ollama_model, cfg.ollama_base_url)
    application.run_polling()


if __name__ == "__main__":
    main()
