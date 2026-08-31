"""Estrazione del vocabolario (oggetto -> categoria di conferimento) dal
PDF "riciclabolario" del gestore, verso un Markdown per lettera.

Il PDF è un elenco a due colonne unite da puntini di riempimento:

    Bottiglia di plastica ................................ Plastica
    Bombola del gas ...................................... Ecocentro
                        o Enti autorizzati se di grandi dimensioni

`pdfplumber` restituisce quelle righe come testo piatto, e i puntini —
che in un PDF sono un fastidio grafico — qui sono il **separatore più
affidabile che ci sia**: sei o più punti consecutivi non compaiono mai
dentro il nome di un oggetto né dentro quello di una categoria. Da qui la
regex `_PUNTINI`, preferita a un'estrazione per coordinate (che si rompe
appena il gestore cambia il layout di una colonna) o a un parsing di
tabella (queste non sono tabelle nel PDF, sono righe di testo).

Due casi da gestire che una sola regex sbaglierebbe:

- **Righe di intestazione**: il PDF stampa la lettera corrente come
  sezione ("B", a volte seguita da una "b" decorativa). Sono righe di una
  lettera sola, senza puntini. Vanno scartate esplicitamente, altrimenti
  finiscono attaccate all'oggetto successivo.
- **Righe di continuazione**: le note sotto una voce ("o Ecocentro se in
  grandi quantità") non hanno puntini. Non sono un oggetto nuovo: sono un
  dettaglio della voce **precedente**, e vanno riattaccate a quella. Senza
  questo, la nota corromperebbe la voce che la segue, ed è una
  correzione che poi qualcuno deve trovare rileggendo il Markdown.

## Perché un file per lettera

È l'unità in cui il PDF stesso è organizzato, quindi è l'unità in cui è
più facile confrontare a occhio l'output con la fonte quando si rilegge —
che è il vero scopo di questo passaggio. Sul retrieval il
raggruppamento conta meno di quanto sembri: il chunking ripete
l'intestazione della tabella su ogni pezzo (vedi `chunk.py`), quindi ogni
chunk resta leggibile da solo anche quando una lettera occupa più chunk.

## Il vocabolario registra MATERIALI, non categorie di raccolta

`normalizza_materiale` traduce i termini del riciclabolario nei materiali
canonici del progetto (carta, cartone, vetro, organico, plastica,
metalli, indifferenziato). Non nel nome del bidone: quale bidone sia
dipende da comune e frazione, mentre il vocabolario è condiviso
dall'intera area.

Il caso che lo rende necessario è reale: Bard, Donnas e Hône raccolgono
carta e cartone insieme, ma le frazioni di Bard e Donnas li raccolgono
separati, in giorni diversi. Un vocabolario che dicesse "scatola di
cartone -> Carta e Cartoni" sarebbe sbagliato per Albard, e la scelta
fra i due flussi ricadrebbe sul modello. Dicendo invece "scatola di
cartone -> cartone", la traduzione avviene su dati
(`raccolta_materiale`, dichiarata nel manifest dell'area).

⚠️ **Il riciclabolario spesso non distingue carta da cartone**: mette
entrambi sotto "Carta". La mappa di default riflette questo, quindi le
voci di cartone (scatole, imballaggi, cassette della frutta) vanno
corrette **a mano** nel Markdown dopo l'estrazione, se l'area ha almeno
un comune o una frazione che li divide. Il volantino "modalità di
conferimento" del gestore di solito elenca esattamente quali oggetti
sono cartone: è la fonte da cui copiare.

La mappa di default è quella di TeknoService; un altro gestore ne passa
una propria (`materiali_map=...`). Non è indovinabile da un'euristica:
va guardata una volta, per gestore.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pdfplumber
import requests

logger = logging.getLogger("bordeus_ingest")

# Sei o più punti consecutivi: il separatore fra oggetto e categoria.
# Meno di sei rischierebbe di spezzare su un'ellissi o su un'abbreviazione.
_PUNTINI = re.compile(r"^(.*?)\.{6,}\s*(.+)$")

# Riga composta da una sola lettera: marcatore di sezione del PDF.
_ARTEFATTO_SEZIONE = re.compile(r"^[A-Za-z]$")

# TeknoService Italia (Sub-ATO E). Chiave = come appare nel
# riciclabolario, valore = materiale canonico.
#
# Nota "Carta" -> "carta": il riciclabolario non distingue carta da
# cartone, quindi anche le scatole finiscono qui. Vanno riclassificate a
# mano come "cartone" nel Markdown, altrimenti nelle frazioni che
# dividono i due flussi verrà indicato il giorno sbagliato. Vedi il
# docstring del modulo.
MATERIALI_TEKNOSERVICE = {
    "Umido": "organico",
    "Carta": "carta",
    "Plastica": "plastica",
    "Vetro": "vetro",
    "Indifferenziato": "indifferenziato",
}

USER_AGENT = (
    "bordeus-ingest/0.2 (+https://github.com/; proof of concept, uso non commerciale)"
)


def normalizza_materiale(voce: str, materiali_map: dict[str, str]) -> str:
    """Voce non mappata = lasciata com'è: la mappa copre solo i materiali
    raccolti porta a porta. "Ecocentro", "Enti autorizzati", "Contenitore
    indumenti usati" e simili non sono materiali e non hanno un
    corrispettivo nel calendario — restano come li scrive il PDF, e il
    modello li legge come destinazione, non come materiale da passare
    allo strumento."""
    return materiali_map.get(voce.strip(), voce.strip())


def _apri_pdf(sorgente: str | Path):
    """Accetta un percorso locale o un URL http(s). I gestori pubblicano
    questi PDF sul proprio sito e cambiano l'URL a ogni revisione: poter
    passare l'URL direttamente evita il passaggio "scarica a mano, poi
    ricordati dove l'hai messo"."""
    testo = str(sorgente)
    if testo.startswith(("http://", "https://")):
        logger.info("scarico il PDF da %s", testo)
        resp = requests.get(testo, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        return pdfplumber.open(BytesIO(resp.content))
    return pdfplumber.open(testo)


def estrai(
    sorgente: str | Path, materiali_map: dict[str, str] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Legge il PDF e restituisce (voci, note): oggetto -> materiale e
    oggetto -> nota, in ordine di comparsa."""
    materiali_map = MATERIALI_TEKNOSERVICE if materiali_map is None else materiali_map

    voci: dict[str, str] = {}
    note: dict[str, str] = {}
    ultima_voce: str | None = None

    with _apri_pdf(sorgente) as pdf:
        for pagina in pdf.pages:
            testo = pagina.extract_text() or ""
            for riga_grezza in testo.splitlines():
                riga = riga_grezza.strip()
                if not riga or _ARTEFATTO_SEZIONE.match(riga):
                    continue

                match = _PUNTINI.match(riga)
                if match:
                    oggetto = match.group(1).strip()
                    voce = match.group(2).strip()
                    if not oggetto:
                        continue
                    voci[oggetto] = normalizza_materiale(voce, materiali_map)
                    ultima_voce = oggetto
                elif ultima_voce is not None:
                    # Riga di continuazione: appartiene alla voce
                    # precedente, non a quella che verrà.
                    note[ultima_voce] = f"{note.get(ultima_voce, '')} {riga}".strip()

    logger.info("estratte %d voci (%d con note) da %s", len(voci), len(note), sorgente)
    return voci, note


def _tabella_markdown(lettera: str, voci: dict[str, str], note: dict[str, str]) -> str:
    righe = [
        f"# Vocabolario - Lettera {lettera}",
        "",
        "| Oggetto | Conferimento |",
        "| --- | --- |",
    ]
    for oggetto in sorted(voci):
        categoria = voci[oggetto]
        nota = note.get(oggetto)
        cella = f"{categoria} ({nota})" if nota else categoria
        # Una pipe dentro il testo romperebbe la colonna: l'unico
        # carattere da proteggere in una cella di tabella Markdown.
        righe.append(f"| {oggetto.replace('|', '/')} | {cella.replace('|', '/')} |")
    return "\n".join(righe) + "\n"


def scrivi_markdown(
    voci: dict[str, str], note: dict[str, str], destinazione: Path
) -> list[Path]:
    """Scrive un file per lettera iniziale in `destinazione`
    (tipicamente `knowledge/<area_id>/guide/`). Sovrascrive: la
    ri-estrazione dello stesso PDF aggiorna, non accumula.

    ⚠️ Quello che esce da qui è una **bozza**. Vanno rilette almeno le
    voci con nota (la nota è testo libero del gestore, a volte spezzato
    male dal PDF), le voci non mappate, e — se l'area ha comuni o
    frazioni che dividono carta e cartone — le voci di cartone, che il
    riciclabolario classifica come "Carta".
    """
    destinazione.mkdir(parents=True, exist_ok=True)

    per_lettera: dict[str, list[str]] = defaultdict(list)
    for oggetto in voci:
        if oggetto:
            per_lettera[oggetto[0].upper()].append(oggetto)

    scritti: list[Path] = []
    for lettera in sorted(per_lettera):
        sottoinsieme = {o: voci[o] for o in per_lettera[lettera]}
        path = destinazione / f"{lettera}.md"
        path.write_text(
            _tabella_markdown(lettera, sottoinsieme, note), encoding="utf-8"
        )
        scritti.append(path)
        logger.info("%s: %d voci", path.name, len(sottoinsieme))

    return scritti


def run(
    sorgente: str | Path,
    destinazione: Path,
    materiali_map: dict[str, str] | None = None,
) -> list[Path]:
    voci, note = estrai(sorgente, materiali_map=materiali_map)
    return scrivi_markdown(voci, note, destinazione)
