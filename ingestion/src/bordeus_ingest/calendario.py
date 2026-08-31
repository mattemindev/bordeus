"""Parsing dei Markdown di calendario e caricamento in `raccolta_date`.

Formato atteso (quello prodotto da `extract/calendario.py`, e quello dei
file già curati a mano nel progetto):

    # Calendario Porta a Porta - GIUGNO '26 through NOVEMBRE '26

    ## ORGANICO (Bidone Marrone)

    - 01/06/2026
    - 04/06/2026

    ## IMB. PLAST. E METALLI (Sacchi Gialli Semitrasparenti)

    - 03/06/2026

`MarkdownHeaderTextSplitter` sui soli `##` invece di una regex sul testo
intero: la categoria arriva già come metadata pulito, e le date restano
associate alla propria sezione senza dover tenere a mano uno stato
"categoria corrente" mentre si scorre il file — che è il punto in cui un
parser scritto a mano sbaglia quando incontra una riga inattesa.

## La categoria è il punto fragile

Il vocabolario dice "Organico", il calendario intitola la sezione
"ORGANICO (Bidone Marrone)". Sono la stessa categoria e devono
combaciare, altrimenti il tool del bot cerca una categoria che non
esiste e risponde "nessuna data trovata" — senza nessun errore, il che
lo rende difficile da notare. La normalizzazione (maiuscolo, spazi
compattati, colore staccato in un campo a parte) è
`bordeus_common.calendario.normalizza_categoria`, la stessa usata in
lettura: un solo posto dove quella regola è definita.

`carica()` segnala esplicitamente le categorie trovate, così chi cura i
dati può confrontarle a occhio con quelle del vocabolario subito dopo
l'ingestion invece di scoprire il disallineamento da una risposta
sbagliata del bot.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path

import psycopg
from bordeus_common.calendario import (
    RigaCalendario,
    normalizza_categoria,
    replace_calendario,
    replace_materiali,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger("bordeus_ingest")

# "ORGANICO (Bidone Marrone)" -> categoria + colore. Il colore è
# informativo (aiuta l'utente a riconoscere il bidone), non una chiave.
_CATEGORIA_CON_COLORE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")

# Formati accettati: gg/mm/aaaa (quello dei file curati) e aaaa-mm-gg
# (ISO, che alcuni modelli restituiscono di loro iniziativa anche quando
# il prompt chiede l'altro). Accettarli entrambi costa una riga e evita
# di perdere un intero semestre per una scelta di formato del modello.
_FORMATI_DATA = ("%d/%m/%Y", "%Y-%m-%d")


def _parse_data(testo: str) -> date | None:
    for formato in _FORMATI_DATA:
        try:
            return datetime.strptime(testo, formato).date()  # noqa: DTZ007 — solo la data: un calendario di raccolta non ha orario né fuso
        except ValueError:
            continue
    return None


def parse(testo: str) -> list[RigaCalendario]:
    """Estrae (categoria, colore, data) da un Markdown di calendario.

    Le righe non riconosciute come data vengono contate e loggate in
    blocco, non ignorate in silenzio: un file in cui il 90% delle righe
    non è una data è quasi sempre un estrattore che ha prodotto
    spazzatura, e va visto subito.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "categoria_raw")], strip_headers=True
    )

    righe: list[RigaCalendario] = []
    scartate = 0

    for sezione in splitter.split_text(testo):
        categoria_raw = sezione.metadata.get("categoria_raw", "").strip()
        if not categoria_raw:
            # Testo prima della prima intestazione `##` (il titolo `#`
            # del file): nessuna categoria a cui appartenere.
            continue

        match = _CATEGORIA_CON_COLORE.match(categoria_raw)
        if match:
            categoria, colore = match.group(1).strip(), match.group(2).strip()
        else:
            categoria, colore = categoria_raw, ""

        for riga_testo in sezione.page_content.splitlines():
            pulita = riga_testo.strip().lstrip("-*").strip()
            if not pulita:
                continue
            data = _parse_data(pulita)
            if data is None:
                scartate += 1
                continue
            righe.append(
                RigaCalendario(categoria=categoria, data=data, colore=colore)
            )

    if scartate:
        logger.warning(
            "%d righe non riconosciute come data e scartate (%d date valide "
            "estratte) — se la proporzione ti sembra sbagliata, controlla il "
            "file: il formato atteso è un elenco puntato di gg/mm/aaaa sotto "
            "una intestazione '## CATEGORIA (Colore)'",
            scartate,
            len(righe),
        )

    return righe


def valida_mappatura(
    mappatura: dict[str, str | None], categorie_presenti: set[str], contesto: str
) -> None:
    """Ogni categoria citata nella mappatura deve esistere in almeno uno
    dei calendari della destinazione.

    Il controllo è sull'**unione** delle categorie di tutti i file di
    quella destinazione, non su un file alla volta: la mappatura
    descrive lo schema di raccolta di un comune o di una frazione, che
    vale per tutti i suoi semestri, mentre un singolo file copre un
    semestre e può legittimamente non contenere una categoria (una
    raccolta stagionale, un semestre in cui un flusso non è passato).
    Validare per file segnalerebbe quei casi come errori.

    Intercetta invece la classe di errore più insidiosa della pipeline.
    Il gestore usa nomi leggermente diversi nei propri materiali: il
    volantino "modalità di conferimento" di Bard-Donnas-Hône scrive
    "CARTA E CARTONE", la griglia del calendario scrive "CARTA E
    CARTONI". Chi compila il manifest copia dal volantino, la categoria
    non combacia con quella finita in `raccolta_date`, e il risultato non
    è un errore ma un silenzio: lo strumento non trova mai date per la
    carta, e nessuno se ne accorge finché un utente non fa la domanda.
    """
    ignote = {
        categoria
        for categoria in mappatura.values()
        if categoria is not None
        and normalizza_categoria(categoria) not in categorie_presenti
    }
    if ignote:
        raise ValueError(
            f"{contesto}: la mappatura dei materiali cita categorie che non "
            f"esistono in nessun calendario di questa destinazione: "
            f"{', '.join(sorted(ignote))}. Categorie realmente presenti: "
            f"{', '.join(sorted(categorie_presenti))}. "
            "Attenzione ai nomi che il gestore scrive in modo diverso fra "
            "volantino e griglia del calendario: qui vale quello della griglia."
        )


def carica_destinazione(
    conn: psycopg.Connection,
    comune_id: str,
    hamlet: str,
    files: list[tuple[Path, str]],
    materiali: dict[str, str | None] | None = None,
    fonte=None,
) -> int:
    """Carica tutti i calendari di una destinazione (comune o frazione)
    e la sua mappatura dei materiali.

    Una destinazione alla volta, non un file alla volta, perché la
    mappatura è per destinazione mentre le date sono per file: solo
    guardando insieme tutti i semestri di un comune si sa quali categorie
    esistono davvero, che è ciò contro cui la mappatura va validata.

    Le date sono sostituite per `source` (il file di provenienza), così
    correggere un semestre e rieseguire non tocca gli altri e non lascia
    righe orfane di una versione precedente.
    """
    etichetta = f"{comune_id}/{hamlet}" if hamlet else comune_id

    parsati: list[tuple[str, list[RigaCalendario]]] = []
    categorie_presenti: set[str] = set()
    for path, source in files:
        righe = parse(path.read_text(encoding="utf-8"))
        if not righe:
            logger.warning("nessuna data estratta da %s: salto", source)
            continue
        parsati.append((source, righe))
        categorie_presenti.update(normalizza_categoria(r.categoria) for r in righe)

    mappatura = materiali or {}
    if mappatura and categorie_presenti:
        valida_mappatura(mappatura, categorie_presenti, contesto=etichetta)

        senza_materiale = categorie_presenti - {
            normalizza_categoria(c) for c in mappatura.values() if c
        }
        if senza_materiale:
            # Non è un errore: una categoria può esistere senza che
            # nessun materiale canonico ci porti (es. un flusso che il
            # vocabolario non nomina). Ma è anche il sintomo di un
            # materiale dimenticato nella mappatura, e in quel caso lo
            # strumento non troverebbe mai quelle date.
            logger.warning(
                "%s: categorie presenti nel calendario ma non raggiungibili da "
                "nessun materiale: %s — se dovrebbero esserlo, aggiungi il "
                "materiale in area.toml",
                etichetta,
                ", ".join(sorted(senza_materiale)),
            )

    totale = 0
    for source, righe in parsati:
        totale += replace_calendario(
            conn,
            comune_id,
            hamlet,
            source,
            righe,
            fonte_nome=fonte.nome if fonte else "",
            fonte_url=fonte.url if fonte else "",
        )

    if parsati:
        replace_materiali(conn, comune_id, hamlet, etichetta, mappatura)

    logger.info(
        "%s: %d date da %d file, categorie: %s",
        etichetta,
        totale,
        len(parsati),
        ", ".join(sorted(categorie_presenti)) or "nessuna",
    )
    if mappatura:
        non_pap = sorted(m for m, c in mappatura.items() if c is None)
        if non_pap:
            logger.info(
                "  non raccolti porta a porta: %s", ", ".join(non_pap)
            )
    else:
        logger.info(
            "  nessuna mappatura dei materiali: i nomi delle categorie qui "
            "sopra devono combaciare con quelli usati nel vocabolario"
        )

    return totale
