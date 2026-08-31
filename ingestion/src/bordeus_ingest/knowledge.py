"""Convenzioni della cartella `knowledge/`: dove vivono i Markdown
curati e come si deduce il loro `tipo`.

    knowledge/<area_id>/area.toml                       manifest dell'area
    knowledge/<area_id>/<tipo>/*.md                     documenti RAG condivisi dall'area
    knowledge/<area_id>/_comuni/<comune_id>/<tipo>/*.md documenti RAG di UN comune
    knowledge/<area_id>/_calendari/<comune_id>/*.md     calendari (-> Postgres, non RAG)
    knowledge/<area_id>/_calendari/<comune_id>/_frazioni/<hamlet>/*.md   override per frazione

Il `tipo` è il nome della cartella che contiene il file. Nella versione
precedente della pipeline era indovinato da un'euristica a parole chiave
(`classify.py`, ora rimosso) perché i file arrivavano da un crawl e
nessuno li aveva guardati. Ora ogni file è messo lì da chi cura i dati,
che sa già cosa contiene: la cartella È la dichiarazione, e non può
sbagliarsi come faceva l'euristica (che classificava "calendario" un
vocabolario solo perché citava le parole "giorni di raccolta").

## Cartelle con underscore

`_comuni`, `_calendari`, `_frazioni`: l'underscore iniziale segnala
"non è una cartella di tipo". La regola è generale — qualunque cartella
il cui nome inizia per underscore viene saltata dalla scoperta dei
documenti RAG — così aggiungere una cartella di servizio in futuro non
la fa finire per sbaglio nel vector store con un `tipo` inventato.

`_calendari` in particolare **deve** restare esclusa: il calendario non
è più contenuto RAG, va in `raccolta_date` (vedi
`bordeus_common.calendario`). Se finisse anche nel vector store, il
modello avrebbe davanti sia la data corretta restituita dal tool sia un
elenco di 50 date in testo libero da cui pescarne un'altra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Radice di knowledge/, sorella di src/ e notebooks/ dentro ingestion/.
KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "knowledge"

COMUNI_SUBDIR = "_comuni"
FRAZIONI_SUBDIR = "_frazioni"
CALENDARI_SUBDIR = "_calendari"

MARKDOWN_SUFFIXES = (".md", ".markdown")

# Tipi previsti oggi, per un avviso quando ne compare uno diverso — non
# un elenco chiuso: chi cura i dati può creare una cartella nuova (es.
# "tariffe") e funziona senza toccare il codice. L'avviso serve solo a
# far notare un refuso ("guie/" invece di "guide/"), che altrimenti
# passerebbe come tipo valido e romperebbe i filtri di retrieval in
# silenzio.
TIPI_NOTI = frozenset(
    {
        "guide",
        "vocabolario",
        "categorie",
        "info",
        "info_generali",
        "moduli",
        "servizi",
        "altro",
    }
)


def area_dir(area_id: str) -> Path:
    return KNOWLEDGE_ROOT / area_id


def resolve_area_dir(area: str) -> Path:
    """Accetta sia un id di area (`sub-ato-e`, risolto sotto
    `KNOWLEDGE_ROOT`) sia un percorso a una cartella — comodo per
    lavorare su una copia fuori dal repo senza dover spostare file."""
    candidato = Path(area)
    if candidato.is_dir():
        return candidato.resolve()
    return area_dir(area)


def is_service_dir(path: Path) -> bool:
    """Cartella di servizio (underscore o punto iniziale), da saltare
    nella scoperta dei documenti RAG."""
    return path.name.startswith(("_", "."))


@dataclass(frozen=True)
class CalendarioTrovato:
    """Un file di calendario e la destinazione dedotta dal suo percorso."""

    path: Path
    comune_id: str
    hamlet: str  # "" = calendario dell'intero comune
    source: str  # percorso relativo all'area: stabile, usato come chiave di ri-ingestione

    @property
    def etichetta(self) -> str:
        return f"{self.comune_id}/{self.hamlet}" if self.hamlet else self.comune_id


def discover_calendari(area_dir: Path) -> list[CalendarioTrovato]:
    """Scopre i calendari da `_calendari/`, deducendo comune e frazione
    dal percorso invece che da una dichiarazione nel manifest.

        _calendari/<comune_id>/<periodo>.md
        _calendari/<comune_id>/_frazioni/<hamlet>/<periodo>.md

    Il percorso è il legame. Chi cura i dati apre la cartella del proprio
    comune e ci trova i suoi calendari, senza dover incrociare un file di
    configurazione — che era il difetto della versione precedente, dove i
    calendari stavano in una cartella piatta e il legame con i comuni era
    dichiarato altrove.
    """
    base = area_dir / CALENDARI_SUBDIR
    if not base.is_dir():
        return []

    trovati: list[CalendarioTrovato] = []
    for comune_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if comune_dir.name.startswith("."):
            continue
        comune_id = comune_dir.name

        for file in sorted(comune_dir.glob("*")):
            if file.is_file() and file.suffix.lower() in MARKDOWN_SUFFIXES:
                trovati.append(
                    CalendarioTrovato(
                        path=file,
                        comune_id=comune_id,
                        hamlet="",
                        source=str(file.relative_to(area_dir)),
                    )
                )

        frazioni_dir = comune_dir / FRAZIONI_SUBDIR
        if not frazioni_dir.is_dir():
            continue
        for hamlet_dir in sorted(p for p in frazioni_dir.iterdir() if p.is_dir()):
            for file in sorted(hamlet_dir.glob("*")):
                if file.is_file() and file.suffix.lower() in MARKDOWN_SUFFIXES:
                    trovati.append(
                        CalendarioTrovato(
                            path=file,
                            comune_id=comune_id,
                            hamlet=hamlet_dir.name,
                            source=str(file.relative_to(area_dir)),
                        )
                    )

    return trovati


def calendario_dir(area_dir: Path, comune_id: str, hamlet: str = "") -> Path:
    """Cartella in cui va scritto il calendario di una destinazione —
    usata dall'estrattore per creare le copie nel posto giusto."""
    base = area_dir / CALENDARI_SUBDIR / comune_id
    return base / FRAZIONI_SUBDIR / hamlet if hamlet else base


def short(text: str, max_len: int = 50) -> str:
    """Accorcia una stringa per la barra di avanzamento, così non sfonda
    la larghezza del terminale."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"
