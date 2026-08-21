"""bordeus_ingest: pipeline di ingestion RAG per bordeus, costruita
attorno a LangChain.

Scarica pagine HTML e i PDF/Markdown linkati in una cartella locale
knowledge/<sub_ato_id>/<categoria>/, li carica con i loader di
LangChain (`BSHTMLLoader`/`PDFPlumberLoader` per HTML/PDF, `TextLoader`
per Markdown — vedi
loaders.py), li spezza in chunk con MarkdownTextSplitter e li scrive su
Postgres/pgvector tramite l'integrazione langchain-postgres. Gli
embedding sono raggruppati per **area Sub-ATO** (`sub_ato_id`), non per
comune: un gestore può servire più aree con contenuti diversi (es.
Quendoz, Valle d'Aosta, gestisce sia il Sub-ATO C sia il D con pagine
guida separate) — l'area, non il gestore, è la chiave giusta. L'ingestion
fa l'upsert dell'area e di ciascun comune che vi appartiene, non serve
più che esistano già.

Il wrapper del modello di embedding e la scrittura sul vector store
(`embed.py`/`vectorstore.py` nella prima versione di questo pacchetto)
non sono più qui: vivono in `bordeus_common`, condivisi con il bot —
vedi `bordeus_common.embed`/`bordeus_common.vectorstore`.

Non tutto il contenuto di un'area è condiviso da tutti i comuni che la
compongono: il calendario di raccolta porta a porta, ad esempio, può
variare da un comune all'altro anche sotto lo stesso gestore (caso
reale: TeknoService Italia, Sub-ATO E, Donnas e Pont-Saint-Martin hanno
calendari diversi pur essendo comuni confinanti nella stessa area).
`--comune-url` (ripetibile) ingerisce una fonte specifica di un comune
— il contenuto resta nella stessa collection dell'area, ma taggato con
`comune_id` nel metadata dei chunk, così il bot filtra correttamente
in fase di retrieval (vedi `bordeus_common.vectorstore`, `bot/rag.py`).

Lo schema Postgres scritto qui (gestito da `langchain-postgres`) è letto
direttamente dal bot Python (`bot/src/bordeus_bot/`) tramite la stessa
integrazione — nessun secondo schema da tenere allineato.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from bordeus_common.embed import DEFAULT_MODEL_NAME, EMBEDDING_DIM, get_embeddings
from dotenv import load_dotenv

from . import viz
from .pipeline import ComuneInput, run_sub_ato

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIM",
    "ComuneInput",
    "get_embeddings",
    "run_sub_ato",
    "viz",
]


def _parse_id_nome_arg(flag: str, value: str) -> tuple[str, str]:
    """Formato atteso: 'id:Nome' (es. 'sub-ato-e:Sub-ATO E — Mont-Rose
    e Walser'). Se manca ':' usiamo il valore anche come nome, così un
    uso rapido con un solo elemento funziona comunque, senza richiedere
    sempre il formato completo."""
    raw_id, sep, nome = value.partition(":")
    raw_id = raw_id.strip()
    nome = nome.strip() if sep else raw_id
    if not raw_id:
        raise argparse.ArgumentTypeError(
            f"{flag} non valido: {value!r} (atteso 'id:Nome')"
        )
    return raw_id, nome


def _parse_comune_arg(value: str) -> ComuneInput:
    comune_id, nome = _parse_id_nome_arg("--comune", value)
    return ComuneInput(id=comune_id, nome=nome)


def _parse_comune_url_arg(value: str) -> tuple[str, str]:
    """Formato atteso: 'id:url' (es.
    'donnas:https://x.it/calendario-donnas.pdf') — a differenza di
    --sub-ato/--comune, qui i due punti separano id e URL, non id e
    nome: un URL può contenere altri ':' (es. 'https://'), quindi il
    partition avviene sul PRIMO ':' incontrato, non l'ultimo."""
    comune_id, sep, url = value.partition(":")
    comune_id = comune_id.strip()
    url = url.strip()
    if not sep or not comune_id or not url:
        raise argparse.ArgumentTypeError(
            f"--comune-url non valido: {value!r} (atteso 'id:url', es. 'donnas:https://...')"
        )
    return comune_id, url


def main() -> None:
    """Entry point CLI:

        uv run bordeus-ingest --sub-ato=sub-ato-e:"Sub-ATO E — Mont-Rose e Walser" \\
            --gestore="TeknoService Italia" \\
            --url=https://www.teknoserviceitalia.com/rifiuti \\
            --comune=donnas:Donnas \\
            --comune=bard:Bard \\
            --comune-url=donnas:https://www.teknoserviceitalia.com/calendario-donnas.pdf

    Un'area Sub-ATO (--sub-ato) può avere più fonti condivise (--url,
    ripetibile — tipico: una pagina specifica dell'area + un contenuto
    condiviso da tutte le aree dello stesso gestore, come un
    vocabolario) e copre sempre uno o più comuni (--comune, ripetibile).
    Se un comune ha contenuto specifico non condiviso con gli altri
    comuni dell'area (tipicamente un calendario di raccolta), aggiungilo
    con --comune-url=id:url (ripetibile, anche più volte per lo stesso
    comune). Area e comuni vengono creati/aggiornati in Postgres
    (upsert) prima dell'ingestion, poi l'area riceve la propria
    collection isolata nel vector store — i chunk da --comune-url
    restano nella stessa collection, solo taggati col comune.
    """
    load_dotenv()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--sub-ato",
        dest="sub_ato",
        required=True,
        metavar="id:Nome",
        help="area Sub-ATO da ingerire (es. 'sub-ato-e:Sub-ATO E — Mont-Rose e Walser')",
    )
    parser.add_argument(
        "--gestore",
        default="",
        help="nome del gestore/consorzio dell'area (informativo, mostrato all'utente dal bot)",
    )
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        required=True,
        help="URL condiviso dall'area da cui ingerire (ripetibile: una pagina specifica + eventuale contenuto condiviso)",
    )
    parser.add_argument(
        "--comune",
        dest="comuni",
        action="append",
        type=_parse_comune_arg,
        required=True,
        metavar="id:Nome",
        help="comune coperto da quest'area (ripetibile per più comuni)",
    )
    parser.add_argument(
        "--comune-url",
        dest="comune_urls",
        action="append",
        type=_parse_comune_url_arg,
        default=[],
        metavar="id:url",
        help="URL specifico di UN comune, non condiviso dall'area (ripetibile — tipico: calendario di raccolta)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="modello sentence-transformers da usare (default: env EMBEDDING_MODEL, o il default del pacchetto)",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL non impostata", file=sys.stderr)
        raise SystemExit(1)

    sub_ato_id, sub_ato_nome = _parse_id_nome_arg("--sub-ato", args.sub_ato)

    comune_urls: dict[str, list[str]] = {}
    for comune_id, url in args.comune_urls:
        comune_urls.setdefault(comune_id, []).append(url)

    embeddings = get_embeddings(args.model)

    run_sub_ato(
        sub_ato_id,
        sub_ato_nome,
        args.gestore,
        args.comuni,
        args.urls,
        database_url,
        embeddings,
        comune_urls=comune_urls,
    )


if __name__ == "__main__":
    main()
