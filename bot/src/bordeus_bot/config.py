"""Configurazione del bot da variabili d'ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    telegram_token: str
    database_url: str
    embedding_model: str | None  # None -> default del pacchetto (bordeus_common.embed.DEFAULT_MODEL_NAME)
    ollama_base_url: str
    ollama_model: str


def load() -> Config:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN non impostata")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL non impostata")

    return Config(
        telegram_token=token,
        database_url=database_url,
        embedding_model=os.environ.get("EMBEDDING_MODEL"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "gemma4:latest"),
    )
