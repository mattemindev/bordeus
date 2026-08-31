"""Estrazione di un calendario di raccolta da un'immagine (o da una
pagina di PDF esportata come immagine), verso un Markdown per semestre.

I gestori pubblicano i calendari come griglie a colori: una colonna per
mese, una riga per giorno, e la categoria di rifiuto indicata dal
**colore della cella** oltre che dal testo. L'OCR classico legge il testo
e perde la struttura — che è esattamente l'informazione che serve, perché
è la posizione di una cella nella colonna a dire di che giorno si tratta.
Un modello multimodale con output strutturato legge invece la griglia
come griglia.

## Output strutturato, non testo libero

Lo schema Pydantic (`CalendarioEstratto`) passa come `response_format`:
il modello non può rispondere in prosa, deve riempire i campi. Non
garantisce che le date siano *giuste*, garantisce che siano
*processabili* — la differenza tra un errore che si vede subito
rileggendo il Markdown e un errore che va cercato dentro un paragrafo.

Restano da rileggere a mano, sempre: i modelli sbagliano volentieri i
bordi mese (l'ultima riga di una colonna) e le settimane con festivi
spostati. È il motivo per cui questo comando scrive Markdown e non
direttamente su Postgres.

## Configurazione

Endpoint OpenAI-compatibile, quindi lo stesso codice funziona con Ollama
locale (`/v1`) o con un provider remoto, scegliendo con le variabili
d'ambiente. Tutto da ambiente, niente chiavi nel codice: durante la
sperimentazione questo estrattore è nato in un notebook con la chiave
scritta inline, ed è il tipo di cosa che finisce in un commit.

    INGEST_VISION_BASE_URL   default http://localhost:11434/v1 (Ollama)
    INGEST_VISION_MODEL      default qwen3.5:9b
    INGEST_VISION_API_KEY    default "ollama" (Ollama non la verifica)
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("bordeus_ingest")

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3.5:9b"

_PROMPT = """\
Extract all waste collection dates from this Italian waste calendar image.
The calendar is laid out as a grid: each month is a column, with a header \
naming the month and year (e.g. "GIUGNO '26"), and each row is a day of that \
month paired with the waste category collected that day (e.g. "1 LUN | ORGANICO").
Each category also has its own colour (e.g. ORGANICO orange, RUR grey, \
IMB. PLAST. E METALLI yellow) — use the colour to disambiguate when the text \
is unclear, and report the bin colour or bag description when the legend gives one.
Group the dates by category. Every date must use the full DD/MM/YYYY format, \
with the month and year taken from the column header it belongs to.
Do not invent dates that are not in the image, and do not skip the last rows \
of a column.
"""


class CategoriaEstratta(BaseModel):
    categoria: str = Field(
        description=(
            "Waste category name in Italian, exactly as written in the calendar "
            "(e.g. ORGANICO, RUR, CARTA E CARTONI, VETRO, IMB. PLAST. E METALLI)"
        )
    )
    colore: str = Field(
        default="",
        description=(
            "Bin or bag description from the legend, if any "
            "(e.g. 'Bidone Marrone', 'Sacchi Gialli Semitrasparenti'). "
            "Empty string when the calendar does not give one."
        ),
    )
    date: list[str] = Field(
        description="Collection dates for this category, DD/MM/YYYY (e.g. ['01/06/2026'])"
    )


class CalendarioEstratto(BaseModel):
    periodo: str = Field(
        description="Period covered by the calendar, e.g. \"GIUGNO '26 - NOVEMBRE '26\""
    )
    categorie: list[CategoriaEstratta] = Field(
        description="Collection dates grouped by waste category"
    )


def _data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None or not mime.startswith("image/"):
        raise ValueError(
            f"{path}: tipo di immagine non riconosciuto. Se è un PDF, esportane "
            "prima la pagina come PNG/JPEG — questo estrattore lavora sulla "
            "griglia a colori, che il testo estratto da un PDF non conserva."
        )
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _parse_completion(client, **kwargs):
    """`chat.completions.parse` è uscito da `beta` in una versione
    recente dell'SDK openai. Proviamo prima la posizione attuale e
    ricadiamo su quella precedente, invece di fissare una versione
    minima solo per questo."""
    parse = getattr(getattr(client, "chat", None), "completions", None)
    if parse is not None and hasattr(parse, "parse"):
        return client.chat.completions.parse(**kwargs)
    return client.beta.chat.completions.parse(**kwargs)


def estrai(
    immagine: Path,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> CalendarioEstratto:
    # Import locale: `openai` serve solo a questo estrattore, non a
    # `sync`, che è il comando che gira più spesso.
    from openai import OpenAI

    model = model or os.environ.get("INGEST_VISION_MODEL", DEFAULT_MODEL)
    base_url = base_url or os.environ.get("INGEST_VISION_BASE_URL", DEFAULT_BASE_URL)
    api_key = api_key or os.environ.get("INGEST_VISION_API_KEY", "ollama")

    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("estrazione calendario da %s (modello=%s)", immagine.name, model)

    response = _parse_completion(
        client,
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_uri(immagine)},
                    },
                ],
            }
        ],
        response_format=CalendarioEstratto,
    )
    estratto = response.choices[0].message.parsed
    if estratto is None:
        raise RuntimeError(
            f"il modello {model!r} non ha restituito un output conforme allo "
            "schema. Non tutti i modelli supportano response_format: prova un "
            "modello diverso con INGEST_VISION_MODEL."
        )
    return estratto


def to_markdown(estratto: CalendarioEstratto) -> str:
    """Stesso formato che `bordeus_ingest.calendario.parse` sa
    rileggere: intestazione `##` per categoria con il colore fra
    parentesi, elenco puntato di date."""
    parti = [f"# Calendario Porta a Porta - {estratto.periodo}", ""]
    for categoria in estratto.categorie:
        titolo = categoria.categoria.strip().upper()
        if categoria.colore.strip():
            titolo = f"{titolo} ({categoria.colore.strip()})"
        parti.append(f"## {titolo}")
        parti.append("")
        parti.extend(f"- {d.strip()}" for d in categoria.date)
        parti.append("")
    return "\n".join(parti)


def run(
    immagine: Path,
    destinazioni: list[Path],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[Path]:
    """Estrae **una volta** dall'immagine e scrive lo stesso Markdown in
    tutte le destinazioni.

    I calendari vivono in una cartella per comune, quindi un calendario
    che vale per Bard, Donnas e Hône esiste in tre copie identiche. La
    duplicazione è il prezzo di un layout navigabile, ma copiare i file
    a mano è il modo più semplice per dimenticarne uno — e una copia
    dimenticata non dà errore, dà un comune che risponde con le date del
    semestre scorso.

    Scriverle tutte qui elimina il problema alla fonte: il momento in cui
    le copie nascono è anche l'unico in cui sono automatiche. Una sola
    chiamata al modello, che è anche la parte lenta e costosa.
    """
    estratto = estrai(immagine, model=model, base_url=base_url, api_key=api_key)
    markdown = to_markdown(estratto)

    scritti: list[Path] = []
    for destinazione in destinazioni:
        destinazione.parent.mkdir(parents=True, exist_ok=True)
        destinazione.write_text(markdown, encoding="utf-8")
        scritti.append(destinazione)

    totale = sum(len(c.date) for c in estratto.categorie)
    logger.info(
        "%d date in %d categorie (%s), scritte in %d file. RILEGGILE prima di "
        "eseguire sync: i bordi mese e le settimane con festivi sono i punti "
        "in cui il modello sbaglia più spesso.",
        totale,
        len(estratto.categorie),
        ", ".join(c.categoria for c in estratto.categorie),
        len(scritti),
    )
    return scritti
