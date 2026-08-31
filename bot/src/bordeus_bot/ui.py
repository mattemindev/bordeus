"""Pezzi di interfaccia riusati dagli handler: messaggio di stato a
fasi, indicatore "sta scrivendo", invio di risposte lunghe.

## Il messaggio di attesa

Una richiesta impiega diversi secondi: identificazione dell'oggetto,
ricerca nelle guide del comune, eventuale lettura del calendario. Un
unico "un attimo..." lascia l'utente senza sapere se il bot è vivo, e
soprattutto senza sapere **cosa ha capito**.

Il messaggio di stato viene quindi aggiornato man mano, e la fase
centrale mostra l'oggetto riconosciuto: "Ho capito: tazzina in
ceramica". È l'informazione più utile che il bot possa dare durante
l'attesa, perché è anche il punto in cui sbaglia — se ha riconosciuto
la cosa sbagliata, l'utente lo vede subito e può riformulare senza
aspettare una risposta che sarà comunque inutile.

Le fasi sono descritte in termini di cosa succede per l'utente ("sto
cercando nelle guide del tuo comune"), non di cosa succede nel
programma: nessun riferimento a retrieval, embedding o modelli.

Alla fine lo stesso messaggio diventa la risposta, invece di
accumularne uno nuovo: la chat resta pulita.

## L'indicatore "sta scrivendo"

Telegram lo fa scadere dopo circa cinque secondi, quindi va rinnovato
finché il lavoro è in corso. `Progress` lo tiene acceso da sé con un
task in background, che viene sempre fermato nel `finally` — altrimenti
un errore lascerebbe l'indicatore acceso all'infinito.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import re
from typing import Self

from bordeus_common.log import get_logger
from telegram import LinkPreviewOptions, Message
from telegram.constants import ChatAction, ParseMode

logger = get_logger("bordeus_bot")

# Limite di Telegram per un singolo messaggio. Le risposte del bot sono
# quasi sempre brevi, ma un modello può divagare e un errore qui
# significherebbe perdere del tutto la risposta.
MAX_MESSAGE_LEN = 4096

# Le anteprime dei link sono disattivate: la fonte è spesso un PDF di
# qualche megabyte, e Telegram ne allegherebbe una scheda di download
# sotto ogni risposta — più ingombrante della risposta stessa. Il link
# resta cliccabile.
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

# `**grassetto**` e `` `codice` ``: il minimo che i modelli producono
# spontaneamente. `_corsivo_` è deliberatamente NON convertito — gli
# underscore compaiono dentro gli URL e dentro gli id (`les_pians`,
# `sub-ato-e`), e trasformarli in corsivo spezzerebbe proprio il testo
# che deve restare copiabile.
_GRASSETTO = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_CODICE = re.compile(r"`([^`\n]+)`")

# Telegram fa scadere l'azione di chat dopo ~5 secondi.
_TYPING_REFRESH_SECONDS = 4.0


def to_html(testo: str) -> str:
    """Converte il poco Markdown che i modelli usano nell'HTML che
    Telegram accetta.

    Telegram non interpreta Markdown se non glielo si chiede, quindi
    senza questo passaggio l'utente legge `**Vetro**` con gli asterischi
    in chiaro. Fra i tre formati possibili si usa HTML e non MarkdownV2:
    MarkdownV2 obbliga a fare l'escape di una quindicina di caratteri
    (`.`, `-`, `!`, `(`, `)`...) che compaiono di continuo in un testo
    normale, e un solo carattere dimenticato fa fallire l'invio
    dell'intera risposta. In HTML i caratteri da proteggere sono tre.

    L'escape viene fatto **prima** della conversione: così un `<` scritto
    dal modello diventa testo, non un tag, e non c'è modo che l'output
    del modello inietti markup.
    """
    sicuro = html.escape(testo, quote=False)
    sicuro = _CODICE.sub(r"<code>\1</code>", sicuro)
    return _GRASSETTO.sub(r"<b>\1</b>", sicuro)


def html_to_plain(testo: str) -> str:
    """Toglie i tag e ripristina le entità: il ripiego quando Telegram
    rifiuta il markup."""
    senza_tag = re.sub(r"<[^>]+>", "", testo)
    return html.unescape(senza_tag)


def split_message(text: str, limit: int = MAX_MESSAGE_LEN) -> list[str]:
    """Spezza un testo lungo in messaggi, preferendo i confini di
    paragrafo, poi di riga, poi di parola.

    Il taglio deve cadere oltre metà della finestra: un confine trovato
    subito all'inizio produrrebbe un messaggio di poche parole seguito da
    uno pieno, che si legge peggio di un taglio un po' più brutto ma
    equilibrato. Se nessun separatore soddisfa la condizione — un blocco
    di testo senza spazi, per dire — si taglia netto al limite: meglio
    una parola spezzata che un invio rifiutato.

    Si prende il **primo** tipo di separatore che va bene, non l'ultimo
    valutato: una versione precedente riassegnava `taglio` a ogni
    tentativo successivo, quindi un ottimo confine di paragrafo veniva
    sostituito dal primo spazio disponibile e i pezzi finivano a metà
    frase.
    """
    if len(text) <= limit:
        return [text]

    pezzi: list[str] = []
    rimanente = text
    while len(rimanente) > limit:
        finestra = rimanente[:limit]
        taglio = -1
        for separatore in ("\n\n", "\n", " "):
            posizione = finestra.rfind(separatore)
            if posizione > limit // 2:
                taglio = posizione
                break
        if taglio <= 0:
            taglio = limit
        pezzi.append(rimanente[:taglio].rstrip())
        rimanente = rimanente[taglio:].lstrip()
    if rimanente:
        pezzi.append(rimanente)
    return pezzi


async def send_long(message: Message, text: str) -> None:
    """Risponde spezzando il testo se supera il limite di Telegram."""
    for pezzo in split_message(text):
        await message.reply_text(pezzo)


async def _invia(inviante, testo: str, **kwargs):
    """Invia (o modifica) provando prima in HTML e ricadendo sul testo
    semplice.

    Il ripiego non è pignoleria: il testo viene da un modello, e per
    quanto lo si converta con attenzione resta possibile che produca
    qualcosa che Telegram rifiuta. Fra "risposta senza grassetto" e
    "nessuna risposta" la scelta è ovvia, ma va scritta — altrimenti il
    caso raro diventa un messaggio perso in silenzio."""
    try:
        return await inviante(
            testo,
            parse_mode=ParseMode.HTML,
            link_preview_options=NO_PREVIEW,
            **kwargs,
        )
    except Exception as exc:
        logger.warning(
            "invio in HTML rifiutato (%s): ripiego sul testo semplice", exc
        )
        return await inviante(
            html_to_plain(testo), link_preview_options=NO_PREVIEW, **kwargs
        )


class Progress:
    """Messaggio di stato aggiornabile, con indicatore "sta scrivendo".

    Uso tipico:

        async with Progress(update.message, i18n.t("waiting_text", lang)) as p:
            descrizione = await asyncio.to_thread(...)
            await p.update(i18n.t("waiting_identified", lang, oggetto=descrizione))
            risposta = await asyncio.to_thread(...)
            await p.done(risposta)

    Se il blocco esce per un'eccezione senza che `done()` sia stato
    chiamato, l'indicatore viene comunque spento: l'handler chiamante
    resta responsabile del messaggio d'errore da mostrare.
    """

    def __init__(self, message: Message, testo_iniziale: str) -> None:
        self._richiesta = message
        self._testo_iniziale = testo_iniziale
        self._stato: Message | None = None
        self._typing: asyncio.Task | None = None
        self._chiuso = False

    async def __aenter__(self) -> Self:
        self._stato = await self._richiesta.reply_text(self._testo_iniziale)
        self._typing = asyncio.create_task(self._mantieni_typing())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self._ferma_typing()
        return False

    async def _mantieni_typing(self) -> None:
        chat = self._richiesta.chat
        try:
            while True:
                with contextlib.suppress(Exception):
                    # Un fallimento qui è puramente estetico: non deve
                    # interrompere il lavoro vero né propagare.
                    await chat.send_action(ChatAction.TYPING)
                await asyncio.sleep(_TYPING_REFRESH_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _ferma_typing(self) -> None:
        if self._typing is not None:
            self._typing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._typing
            self._typing = None

    async def update(self, testo: str) -> None:
        """Aggiorna il messaggio di stato. Un fallimento non è fatale: la
        richiesta vera sta procedendo comunque, e interromperla perché
        non si è potuto aggiornare un testo di cortesia sarebbe il
        compromesso sbagliato."""
        if self._stato is None or self._chiuso:
            return
        try:
            # Messaggi di stato: testo nostro, nessun markup da
            # convertire, nessun rischio di rifiuto.
            await self._stato.edit_text(testo)
        except Exception as exc:
            logger.debug("aggiornamento del messaggio di stato fallito: %s", exc)

    async def done(self, testo: str, reply_markup=None) -> None:
        """Trasforma il messaggio di stato nella risposta finale.

        `testo` è già HTML (vedi `to_html`): l'invio ricade da solo sul
        testo semplice se Telegram lo rifiuta.

        Se la risposta supera il limite di Telegram, il primo pezzo
        sostituisce lo stato e il resto arriva come messaggi successivi.
        """
        await self._ferma_typing()
        pezzi = split_message(testo)

        # Solo l'ultimo pezzo porta la tastiera: attaccarla al primo di
        # una risposta lunga la lascerebbe a metà conversazione.
        markup_primo = reply_markup if len(pezzi) == 1 else None

        if self._stato is not None:
            try:
                await _invia(
                    self._stato.edit_text, pezzi[0], reply_markup=markup_primo
                )
            except Exception as exc:
                # Es. messaggio troppo vecchio per essere modificato:
                # meglio un messaggio in più che una risposta persa.
                logger.warning("modifica del messaggio finale fallita: %s", exc)
                await _invia(
                    self._richiesta.reply_text, pezzi[0], reply_markup=markup_primo
                )
        else:
            await _invia(
                self._richiesta.reply_text, pezzi[0], reply_markup=markup_primo
            )

        for i, pezzo in enumerate(pezzi[1:], start=1):
            ultimo = i == len(pezzi) - 1
            await _invia(
                self._richiesta.reply_text,
                pezzo,
                reply_markup=reply_markup if ultimo else None,
            )
        self._chiuso = True

    async def fail(self, testo: str) -> None:
        """Chiude mostrando un errore. Stessa forma di `done`, nome
        diverso perché il chiamante lo legga come esito."""
        await self.done(testo)
