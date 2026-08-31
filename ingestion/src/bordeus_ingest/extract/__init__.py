"""Estrattori: portano una fonte del gestore (PDF, immagine) in Markdown
sotto `knowledge/`, come **punto di partenza da rileggere e correggere a
mano**, non come output definitivo.

È la parte "semi" di semi-automatica. Il contenuto vero dei gestori vive
in tabelle e in immagini di calendari: un'estrazione completamente
automatica produce output plausibile ma con errori sparsi che nessuno
vede finché il bot non risponde male a un utente. Estrarre a mano tutto
da zero, però, è lavoro inutile per la parte meccanica (200 voci di
vocabolario con lo stesso formato).

La divisione è quindi: la macchina fa il lavoro meccanico e produce
Markdown leggibile; una persona lo rilegge, corregge quello che serve, e
solo allora `bordeus-ingest sync` lo porta in Postgres. Il Markdown è il
formato di scambio proprio perché è leggibile e correggibile a mano — un
JSON intermedio non lo sarebbe.

Gli estrattori NON scrivono mai su Postgres, e non vengono chiamati da
`sync`: sono comandi separati (`extract-vocabolario`,
`extract-calendario`) che si eseguono quando il gestore pubblica una
fonte nuova, non ad ogni ingestion.
"""

from __future__ import annotations

from . import calendario, vocabolario

__all__ = ["calendario", "vocabolario"]
