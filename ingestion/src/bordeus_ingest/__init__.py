"""bordeus_ingest: ingestion semi-automatica della knowledge base di
bordeus.

## Com'è cambiata rispetto alla v0.1.0

La prima versione era una pipeline automatica: scaricava le pagine del
gestore, ne seguiva i link a PDF, indovinava una categoria con
un'euristica a parole chiave e caricava tutto nel vector store. Il
problema non era la meccanica ma il risultato: il contenuto che conta —
il vocabolario oggetto/categoria, i calendari — vive in tabelle e in
griglie a colori, e un'estrazione automatica ne perde l'allineamento.
Serviva comunque riscrivere tutto a mano, quindi l'automazione non
risparmiava il lavoro che contava: produceva solo una bozza scadente da
buttare.

Ora il flusso è in tre passaggi, con una persona nel mezzo:

1. **`extract-*`** — un estrattore legge la fonte del gestore (PDF del
   riciclabolario, immagine del calendario) e produce Markdown sotto
   `knowledge/`. Si lancia quando il gestore pubblica una revisione, non
   ad ogni ingestion.
2. **rilettura umana** — il Markdown si corregge a mano. È il passaggio
   che rende "semi" la semi-automazione, ed è il motivo per cui il
   formato di scambio è Markdown e non JSON.
3. **`sync`** — porta i Markdown in Postgres: i documenti nel vector
   store, i calendari in `raccolta_date`.

## Il calendario non è più contenuto RAG

Le date di raccolta vanno in una tabella relazionale, non nel vector
store, e il bot le legge tramite tool calling invece che dal contesto.
Un elenco di 50 date in un chunk di testo è la forma peggiore possibile
per la domanda che gli utenti fanno davvero ("quando passano?"): il
modello deve confrontare date leggendole dal prompt, e quando sbaglia
sbaglia in modo plausibile. Vedi `bordeus_common.calendario`.

## Comandi

    # ingestion completa di un'area (guidata dal manifest area.toml)
    uv run bordeus-ingest sync --area=sub-ato-e

    # solo i calendari (veloce: nessun embedding da ricalcolare)
    uv run bordeus-ingest sync --area=sub-ato-e --only=calendari

    # estrazione di una fonte nuova -> Markdown da rileggere
    uv run bordeus-ingest extract-vocabolario \\
        --pdf=https://www.teknoserviceitalia.com/.../riciclabolario.pdf \\
        --out=knowledge/sub-ato-e/guide
    # una sola estrazione, scritta nella cartella di ogni comune a cui si applica
    uv run bordeus-ingest extract-calendario \\
        --image=fonti/BARD-DONNAS-HONE.jpeg --area=sub-ato-e \\
        --comuni=bard,donnas,hone --periodo=2026_semestre_2

    uv run bordeus-ingest extract-calendario \\
        --image=fonti/FRAZ-BARD-DONNAS.jpeg --area=sub-ato-e \\
        --frazioni=donnas/albard,donnas/bondon,donnas/les_pians,bard/crous \\
        --periodo=2026_semestre_2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from bordeus_common.embed import DEFAULT_MODEL_NAME, EMBEDDING_DIM, get_embeddings
from bordeus_common.log import get_logger, setup_logging
from dotenv import load_dotenv

from . import knowledge, manifest, pipeline, viz
from .pipeline import run_area

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIM",
    "get_embeddings",
    "knowledge",
    "manifest",
    "pipeline",
    "run_area",
    "viz",
]

logger = get_logger("bordeus_ingest")


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL non impostata", file=sys.stderr)
        raise SystemExit(1)
    return url


def _parse_materiali_map(value: str | None) -> dict[str, str] | None:
    """Formato: 'Umido=organico,Carta=carta'. None (flag assente) = mappa
    di default del gestore TeknoService, vedi extract/vocabolario.py."""
    if not value:
        return None
    mappa: dict[str, str] = {}
    for coppia in value.split(","):
        chiave, sep, valore = coppia.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(
                f"--materiali-map non valida vicino a {coppia!r} "
                "(atteso 'Da=A,Da2=A2')"
            )
        mappa[chiave.strip()] = valore.strip()
    return mappa


def _cmd_sync(args: argparse.Namespace) -> None:
    database_url = _database_url()
    area = pipeline.load_manifest(args.area)

    con_rag = args.only in ("all", "rag")
    con_calendari = args.only in ("all", "calendari")

    # get_embeddings carica il modello e ne verifica la dimensione: costa
    # secondi e VRAM, quindi lo costruiamo solo se serve davvero.
    embeddings = get_embeddings(args.model) if con_rag else None

    run_area(
        area,
        database_url,
        embeddings=embeddings,
        con_rag=con_rag,
        con_calendari=con_calendari,
        reset=args.reset,
    )


def _cmd_extract_vocabolario(args: argparse.Namespace) -> None:
    from .extract import vocabolario

    scritti = vocabolario.run(
        args.pdf,
        Path(args.out),
        materiali_map=_parse_materiali_map(args.materiali_map),
    )
    print(f"\n{len(scritti)} file scritti in {args.out}")
    print(
        "Rileggili prima di eseguire 'sync': le voci con nota e quelle non "
        "mappate sono i punti in cui l'estrazione sbaglia più spesso. Se "
        "nell'area qualche comune o frazione divide carta e cartone, "
        "riclassifica a mano come 'cartone' le voci che il riciclabolario "
        "mette sotto 'Carta' (scatole, imballaggi, cassette della frutta)."
    )


def _parse_destinazioni(args: argparse.Namespace) -> list[Path]:
    """Destinazioni di un calendario estratto: `--out` esplicito, oppure
    `--comuni`/`--frazioni` risolti nella struttura di `_calendari/`."""
    if args.out:
        return [Path(args.out)]

    if not args.area:
        raise SystemExit(
            "serve --out oppure --area insieme a --comuni/--frazioni "
            "(con --area le destinazioni sono calcolate dalla struttura di "
            "_calendari/, così non devi copiare i file a mano)"
        )

    area_dir = knowledge.resolve_area_dir(args.area)
    nome = args.periodo if args.periodo.endswith(".md") else f"{args.periodo}.md"

    destinazioni: list[Path] = []
    for comune_id in _lista(args.comuni):
        destinazioni.append(knowledge.calendario_dir(area_dir, comune_id) / nome)
    for coppia in _lista(args.frazioni):
        comune_id, sep, hamlet = coppia.partition("/")
        if not sep:
            raise SystemExit(
                f"--frazioni: {coppia!r} non valida, attesa la forma "
                "'comune/frazione' (es. donnas/albard)"
            )
        destinazioni.append(
            knowledge.calendario_dir(area_dir, comune_id, hamlet) / nome
        )

    if not destinazioni:
        raise SystemExit("nessuna destinazione: usa --comuni e/o --frazioni")
    return destinazioni


def _lista(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _cmd_extract_calendario(args: argparse.Namespace) -> None:
    from .extract import calendario

    destinazioni = _parse_destinazioni(args)
    scritti = calendario.run(
        Path(args.image),
        destinazioni,
        model=args.model,
        base_url=args.base_url,
    )
    print(f"\n{len(scritti)} file scritti:")
    for path in scritti:
        print(f"  {path}")
    print(
        "\nRileggili prima di eseguire 'sync'. Il legame con i comuni è già "
        "dato dal percorso: non c'è niente da dichiarare in area.toml, se non "
        "lo schema di raccolta ('materiali') quando questo calendario ne usa "
        "uno diverso da quello del comune."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bordeus-ingest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p_sync = sub.add_parser(
        "sync",
        help="porta i Markdown curati di un'area in Postgres (vector store + calendari)",
    )
    p_sync.add_argument(
        "--area",
        required=True,
        metavar="ID|PERCORSO",
        help=(
            "id dell'area (es. 'sub-ato-e', risolto sotto knowledge/) oppure "
            "percorso a una cartella d'area contenente area.toml"
        ),
    )
    p_sync.add_argument(
        "--only",
        choices=("all", "rag", "calendari"),
        default="all",
        help=(
            "quale metà eseguire (default: all). 'calendari' salta il "
            "caricamento del modello di embedding: usalo per correggere una "
            "data senza ricalcolare gli embedding dell'intera area"
        ),
    )
    p_sync.add_argument(
        "--reset",
        action="store_true",
        help=(
            "svuota la collection dell'area prima di riscriverla. Serve dopo "
            "un cambio di strategia di chunking o quando si toglie una voce da "
            "un Markdown: l'upsert aggiorna i chunk che ricalcola uguali ma non "
            "cancella quelli che non produce più, e i vecchi restano a "
            "competere nel retrieval"
        ),
    )
    p_sync.add_argument(
        "--model",
        default=None,
        help="modello di embedding (default: env EMBEDDING_MODEL, o quello del pacchetto)",
    )
    p_sync.set_defaults(func=_cmd_sync)

    p_voc = sub.add_parser(
        "extract-vocabolario",
        help="PDF del riciclabolario -> un Markdown per lettera (BOZZA da rileggere)",
    )
    p_voc.add_argument(
        "--pdf", required=True, help="percorso locale o URL http(s) del PDF"
    )
    p_voc.add_argument(
        "--out",
        required=True,
        metavar="CARTELLA",
        help="cartella di destinazione, tipicamente knowledge/<area_id>/guide",
    )
    p_voc.add_argument(
        "--materiali-map",
        dest="materiali_map",
        default=None,
        metavar="Voce=materiale,...",
        help=(
            "traduce i termini del riciclabolario nei materiali canonici "
            "(carta, cartone, vetro, organico, plastica, metalli, "
            "indifferenziato). Default: mappa TeknoService. Il vocabolario "
            "registra materiali, non nomi di bidoni: quale bidone sia dipende "
            "dal comune e dalla frazione, e la traduzione la fa il manifest"
        ),
    )
    p_voc.set_defaults(func=_cmd_extract_vocabolario)

    p_cal = sub.add_parser(
        "extract-calendario",
        help="immagine di un calendario -> Markdown (BOZZA da rileggere)",
    )
    p_cal.add_argument(
        "--image", required=True, help="immagine della griglia del calendario"
    )
    p_cal.add_argument(
        "--area",
        default=None,
        metavar="ID|PERCORSO",
        help="area di destinazione, usata con --comuni/--frazioni",
    )
    p_cal.add_argument(
        "--comuni",
        default=None,
        metavar="bard,donnas,hone",
        help=(
            "comuni a cui si applica questo calendario. Il file viene scritto "
            "in _calendari/<comune>/ per ciascuno, da una sola estrazione: è "
            "così che si evita di copiare le versioni a mano e dimenticarne una"
        ),
    )
    p_cal.add_argument(
        "--frazioni",
        default=None,
        metavar="donnas/albard,bard/crous",
        help=(
            "frazioni a cui si applica questo calendario, come coppie "
            "comune/frazione. Scritto in _calendari/<comune>/_frazioni/<frazione>/"
        ),
    )
    p_cal.add_argument(
        "--periodo",
        default="calendario",
        metavar="2026_semestre_2",
        help="nome del file, senza estensione (default: 'calendario')",
    )
    p_cal.add_argument(
        "--out",
        default=None,
        metavar="FILE.md",
        help="percorso esplicito di un singolo file, alternativo a --area/--comuni",
    )
    p_cal.add_argument(
        "--model",
        default=None,
        help="modello multimodale (default: env INGEST_VISION_MODEL)",
    )
    p_cal.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="endpoint OpenAI-compatibile (default: env INGEST_VISION_BASE_URL)",
    )
    p_cal.set_defaults(func=_cmd_extract_calendario)

    return parser


def main() -> None:
    load_dotenv()
    # LOG_LEVEL accetta anche TRACE (5): vedi bordeus_common.log.
    setup_logging()

    args = build_parser().parse_args()
    try:
        args.func(args)
    except manifest.ManifestError as exc:
        # Errore di configurazione, non un bug: un traceback qui
        # nasconderebbe il messaggio, che è già scritto per essere letto.
        print(f"Manifest non valido: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
