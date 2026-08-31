"""Scoperta e caricamento dei documenti RAG da `knowledge/<area_id>/`.

Sostituisce `loaders.py` della pipeline precedente. Non c'è più niente da
"caricare" nel senso di prima: nessun HTML da ripulire, nessun PDF da cui
estrarre testo, nessun manifest.json che ricordi da quale URL venisse un
file. Ogni file qui è già Markdown curato — o scritto a mano, o prodotto
dagli estrattori in `extract/` e poi riletto e corretto da una persona.
Caricarlo è `read_text()`, il resto è dedurre i metadata dalla posizione.

Il calendario NON viene mai caricato qui, per quanto sia Markdown come
gli altri: vive in `_calendari/` (cartella di servizio, saltata) e va in
Postgres tramite `calendario.py`. Vedi `knowledge.py`.

## Metadata prodotti

- `area_id`   — collection del vector store
- `comune_id` — "" per contenuto condiviso dall'area, altrimenti il
                comune; è la chiave del filtro di retrieval del bot
- `tipo`      — nome della cartella contenitore
- `fonte_nome`/`fonte_url` — la pubblicazione del gestore da cui il file
                proviene, dichiarata in `[[fonti]]` nel manifest. Assenti
                se l'area non ne dichiara: la citazione della fonte è
                facoltativa.
- `source`    — percorso relativo alla cartella dell'area, non un URL:
                le fonti non arrivano più dalla rete, e un percorso
                relativo è ciò che serve davvero per ritrovare il file da
                correggere quando una risposta è sbagliata

`source` entra anche nel calcolo dell'id stabile del chunk
(`bordeus_common.vectorstore.stable_chunk_id`), quindi deve restare
stabile tra un'esecuzione e l'altra: relativo all'area, mai assoluto,
altrimenti spostare il repo duplicherebbe l'intera knowledge base invece
di aggiornarla.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from . import knowledge

logger = logging.getLogger("bordeus_ingest")

def _leggi(
    path: Path,
    area_dir: Path,
    area_id: str,
    tipo: str,
    comune_id: str,
    fonti=None,
):
    testo = path.read_text(encoding="utf-8").strip()
    if not testo:
        logger.warning("file vuoto, salto: %s", path)
        return None

    source = str(path.relative_to(area_dir))
    metadata = {
        "area_id": area_id,
        "comune_id": comune_id,
        "tipo": tipo,
        "source": source,
    }

    # La provenienza viaggia nei metadata del chunk, non in una tabella
    # a parte: il retrieval restituisce già il Document, quindi il bot
    # sa da cosa viene ogni pezzo di contesto senza una query in più.
    fonte = fonti(source) if fonti else None
    if fonte is not None:
        metadata["fonte_nome"] = fonte.nome
        metadata["fonte_url"] = fonte.url

    return Document(page_content=testo, metadata=metadata)


def _documenti_da_cartella_tipi(
    base: Path, area_dir: Path, area_id: str, comune_id: str, fonti=None
) -> list[Document]:
    """Legge `base/<tipo>/*.md`. `base` è la cartella dell'area (per il
    contenuto condiviso) o `_comuni/<comune_id>/` (per quello di un
    comune) — stessa struttura interna nei due casi, stessa funzione."""
    documenti: list[Document] = []
    for tipo_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if knowledge.is_service_dir(tipo_dir):
            continue
        tipo = tipo_dir.name
        if tipo not in knowledge.TIPI_NOTI:
            logger.warning(
                "tipo %r non tra quelli noti (%s) in %s — non è un errore, ma "
                "controlla che non sia un refuso: i filtri di retrieval del bot "
                "ragionano su questi nomi",
                tipo,
                ", ".join(sorted(knowledge.TIPI_NOTI)),
                tipo_dir,
            )
        for file in sorted(tipo_dir.rglob("*")):
            if file.suffix.lower() not in knowledge.MARKDOWN_SUFFIXES or not file.is_file():
                continue
            doc = _leggi(file, area_dir, area_id, tipo, comune_id, fonti)
            if doc is not None:
                documenti.append(doc)
    return documenti


def discover(area_dir: Path, area_id: str, fonti=None) -> list[Document]:
    """Tutti i documenti RAG di un'area: quelli condivisi più quelli
    specifici di ciascun comune.

    I documenti sotto `_comuni/<id>/_frazioni/` sono saltati di
    proposito: i chunk nel vector store hanno un metadata `comune_id`,
    non `hamlet`, quindi il bot non saprebbe filtrarli e un utente della
    frazione sbagliata vedrebbe contenuto non suo. Oggi le frazioni
    hanno solo calendari (che non passano dal RAG), quindi non si perde
    niente; il giorno in cui servisse contenuto RAG per frazione, la
    strada è aggiungere `hamlet` ai metadata dei chunk e al filtro,
    non allentare questa esclusione.
    """
    if not area_dir.is_dir():
        raise FileNotFoundError(f"cartella dell'area non trovata: {area_dir}")

    documenti = _documenti_da_cartella_tipi(
        area_dir, area_dir, area_id, comune_id="", fonti=fonti
    )

    comuni_dir = area_dir / knowledge.COMUNI_SUBDIR
    if comuni_dir.is_dir():
        for comune_dir in sorted(p for p in comuni_dir.iterdir() if p.is_dir()):
            documenti.extend(
                _documenti_da_cartella_tipi(
                    comune_dir, area_dir, area_id, comune_id=comune_dir.name, fonti=fonti
                )
            )

    return documenti
