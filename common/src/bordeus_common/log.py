"""Livello di log TRACE e configurazione condivisa.

`logging` si ferma a DEBUG (10). Serve un livello sotto perché i prompt
di sistema completi sono una categoria di log a sé: indispensabili per
capire *perché* il modello ha risposto in un certo modo, ma lunghi
migliaia di caratteri e ripetuti a ogni messaggio. Metterli a DEBUG
significherebbe rendere DEBUG inutilizzabile per tutto il resto — chi lo
attiva per seguire una query SQL o il flusso di onboarding si
troverebbe il terminale pieno di prompt.

TRACE (5) è quindi il livello del "cosa ha visto esattamente il modello":
system prompt assemblato, contesto recuperato con le fonti, argomenti e
risultati delle chiamate agli strumenti, risposta grezza.

⚠️ **I log TRACE contengono il contenuto dei messaggi degli utenti** (la
descrizione dell'oggetto, che deriva da quanto ha scritto o fotografato
la persona) oltre ai prompt. Va usato per il debug locale e nei
notebook di valutazione, non lasciato acceso su un'istanza che serve
utenti reali.

Uso:

    from bordeus_common.log import TRACE, setup_logging, get_logger

    setup_logging()                 # legge LOG_LEVEL dall'ambiente
    logger = get_logger("bordeus_bot")
    logger.trace("system prompt:\\n%s", prompt)
"""

from __future__ import annotations

import logging
import os

TRACE = 5
TRACE_NAME = "TRACE"

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _trace(self: logging.Logger, message: str, *args, **kwargs) -> None:
    # isEnabledFor prima di formattare: un prompt di sistema è una
    # stringa lunga e l'interpolazione degli argomenti costerebbe anche
    # quando il livello è disattivato, cioè sempre in esercizio normale.
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


def install() -> None:
    """Registra il livello TRACE e il metodo `logger.trace()`.

    Idempotente: chiamarla più volte (import multipli, reload di un
    notebook) non duplica niente."""
    if logging.getLevelName(TRACE) != TRACE_NAME:
        logging.addLevelName(TRACE, TRACE_NAME)
    if not hasattr(logging.Logger, "trace"):
        logging.Logger.trace = _trace  # type: ignore[attr-defined]


def parse_level(value: str | int | None, default: int = logging.INFO) -> int:
    """Accetta "TRACE", "DEBUG", un numero, o None."""
    if value is None or value == "":
        return default
    if isinstance(value, int):
        return value
    testo = str(value).strip().upper()
    if testo.isdigit():
        return int(testo)
    livello = logging.getLevelName(testo)
    # getLevelName restituisce la stringa "Level X" quando il nome non
    # esiste: un LOG_LEVEL scritto male non deve spegnere i log, deve
    # ricadere sul default.
    return livello if isinstance(livello, int) else default


def setup_logging(level: str | int | None = None, fmt: str = DEFAULT_FORMAT) -> int:
    """Installa TRACE e configura il root logger. `level` esplicito, o
    la variabile d'ambiente `LOG_LEVEL`, o INFO."""
    install()
    livello = parse_level(level if level is not None else os.environ.get("LOG_LEVEL"))
    logging.basicConfig(level=livello, format=fmt)
    logging.getLogger().setLevel(livello)
    return livello


def get_logger(name: str) -> logging.Logger:
    """Logger con `.trace()` garantito, anche se `setup_logging` non è
    ancora stata chiamata (import in ordine imprevedibile, notebook)."""
    install()
    return logging.getLogger(name)


def set_level(name: str, level: str | int) -> int:
    """Alza o abbassa il livello di un singolo logger senza toccare gli
    altri — nei notebook serve per accendere TRACE solo su
    `bordeus_bot` e non sulle librerie."""
    install()
    livello = parse_level(level)
    logging.getLogger(name).setLevel(livello)
    # Senza un handler sul root (o senza basicConfig) i record non
    # verrebbero emessi comunque: garantiamo che ci sia.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=livello, format=DEFAULT_FORMAT)
    return livello


def blocco(titolo: str, contenuto: str) -> str:
    """Formatta un contenuto multilinea (prompt, contesto recuperato) con
    delimitatori, così in un terminale si distingue dove finisce il
    prompt e ricominciano i log normali."""
    linea = "─" * 8
    return f"\n{linea} {titolo} {linea}\n{contenuto}\n{linea} fine {titolo} {linea}"


install()
