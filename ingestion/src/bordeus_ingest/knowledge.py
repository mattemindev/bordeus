"""Gestione della cartella locale `knowledge/`: dove i documenti scaricati
vengono salvati prima di essere caricati con i loader di LangChain
(step 2 della pipeline), e il manifest che ne registra i metadati
originali (URL sorgente, tipo di file, categoria, comune specifico se
presente).

Struttura:

    knowledge/<area_id>/<categoria>/<file>                          # contenuto condiviso dall'area
    knowledge/<area_id>/_comuni/<comune_id>/<categoria>/<file>     # contenuto specifico di un comune
    knowledge/<area_id>/manifest.json                                # un solo manifest per l'area, entrambi i tipi di contenuto

`area_id` è lo id del Sub-ATO (es. "sub-ato-e"), non del singolo
comune: un'area può coprire più comuni che condividono le stesse guide
(vedi migrations/0002_sub_ato.sql) — una cartella per area tiene i dati
di aree diverse separati fin dal filesystem, coerente con l'isolamento
usato nel resto del progetto (una collection per area nel vector store).

Non tutto il contenuto di un'area è però davvero condiviso: il
calendario di raccolta porta a porta, ad esempio, può variare da un
comune all'altro della stessa area per motivi logistici, anche se
gestiti dalla stessa azienda (caso reale osservato con TeknoService
Italia in Valle d'Aosta: Donnas e Pont-Saint-Martin, comuni confinanti
nello stesso Sub-ATO E, hanno calendari diversi). Il contenuto
specifico di un comune finisce quindi in una sottocartella dedicata
(`_comuni/<comune_id>/`) e viene taggato nel manifest con il proprio
`comune_id` — a differenza vuoto per il contenuto condiviso — così il
bot può filtrare correttamente in fase di retrieval (vedi
`bordeus_common.vectorstore.get_vectorstore`/`bot/rag.py`): un utente
vede sempre il contenuto condiviso dell'area PIÙ quello specifico del
proprio comune, mai quello di un comune vicino.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

# Radice di knowledge/, sorella di src/ e notebooks/ dentro ingestion/.
KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "knowledge"

# Nome della sottocartella che ospita il contenuto specifico di un
# comune, dentro knowledge/<area_id>/. Con underscore iniziale per
# distinguerla a colpo d'occhio dalle cartelle di categoria
# (calendario/guide/moduli/servizi/altro), che non hanno mai underscore.
_COMUNI_SUBDIR = "_comuni"


@dataclass
class ManifestEntry:
    source_url: str
    kind: str  # "html" | "pdf" | "markdown"
    tipo: str  # categoria da classify.guess_doc_type
    comune_id: str = ""  # vuoto = contenuto condiviso dall'area, altrimenti specifico di quel comune


def area_dir(area_id: str) -> Path:
    return KNOWLEDGE_ROOT / area_id


def tipo_dir(area_id: str, tipo: str, comune_id: str = "") -> Path:
    """Cartella di categoria, per il contenuto condiviso dell'area
    (comune_id vuoto, comportamento invariato) oppure per il
    contenuto specifico di un comune (sotto _comuni/<comune_id>/)."""
    if comune_id:
        return area_dir(area_id) / _COMUNI_SUBDIR / comune_id / tipo
    return area_dir(area_id) / tipo


def short(text: str, max_len: int = 50) -> str:
    """Accorcia una stringa (tipicamente un URL) per la barra di
    avanzamento, così non sfonda la larghezza del terminale."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def sanitize_filename(url: str, default_ext: str) -> str:
    """Deriva un nome file sicuro dall'URL, distintivo abbastanza da non
    collidere tra fonti diverse dello stesso Sub-ATO: da quando
    l'ingestion supporta più URL per area (es. una pagina specifica +
    un vocabolario condiviso), due URL che terminano entrambi con "/"
    genererebbero altrimenti lo stesso "index.html" — se finissero
    anche nella stessa categoria, il secondo sovrascriverebbe il primo
    in silenzio (bug reale trovato testando il caso multi-URL: due path
    "/category/subato-d/" e "/vocabolario/" collidevano). Per questo
    usiamo tutti i segmenti del path, non solo l'ultimo."""
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    stem = "-".join(segments) if segments else parsed.netloc.replace(".", "-")
    name = re.sub(r"[^A-Za-z0-9._-]", "-", stem)
    if not name.lower().endswith(default_ext.lower()):
        name = f"{name}{default_ext}"
    return name


def target_path(area_id: str, tipo: str, filename: str, comune_id: str = "") -> Path:
    """Percorso di destinazione per un file, creando la cartella se
    manca. Sovrascrive sempre in caso di ri-esecuzione (coerente con
    l'idempotenza usata altrove nella pipeline: una re-ingestion
    aggiorna, non accumula copie)."""
    directory = tipo_dir(area_id, tipo, comune_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def manifest_path(area_id: str) -> Path:
    return area_dir(area_id) / "manifest.json"


def load_manifest(area_id: str) -> dict[str, dict]:
    path = manifest_path(area_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(area_id: str, manifest: dict[str, dict]) -> None:
    path = manifest_path(area_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def register_file(
    area_id: str,
    manifest: dict[str, dict],
    file_path: Path,
    source_url: str,
    kind: str,
    tipo: str,
    comune_id: str = "",
) -> None:
    """Aggiunge (o aggiorna) una entry del manifest per un file appena
    salvato. Muta `manifest` in-place; il chiamante decide quando
    persisterlo su disco con save_manifest (tipicamente una volta sola,
    a fine crawl, non ad ogni singolo file)."""
    rel = str(file_path.relative_to(area_dir(area_id)))
    manifest[rel] = asdict(
        ManifestEntry(source_url=source_url, kind=kind, tipo=tipo, comune_id=comune_id)
    )
