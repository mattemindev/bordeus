"""Manifest dell'area (`knowledge/<area_id>/area.toml`): anagrafica dei
comuni e **schema di raccolta** di ciascuno.

## Cosa NON sta più qui: il legame calendario -> comune

I calendari sono organizzati per comune sul filesystem, quindi il
percorso È il legame e non serve dichiararlo:

    _calendari/<comune_id>/<periodo>.md
    _calendari/<comune_id>/_frazioni/<hamlet>/<periodo>.md

Una versione precedente teneva i calendari in una cartella piatta e
dichiarava in questo file a quali comuni si applicasse ciascun file.
Evitava di duplicare contenuto identico (Bard, Donnas e Hône condividono
lo stesso calendario), ma per sapere quale calendario valesse per un
comune bisognava leggere il manifest invece di guardare una cartella —
e chi cura i dati guarda le cartelle.

Il prezzo del cambio è la duplicazione: lo stesso semestre esiste ora in
più copie, e una copia dimenticata durante un aggiornamento non dà
errore, dà un comune che risponde con le date del semestre scorso. Due
contromisure, entrambe necessarie perché la disciplina umana non basta:

1. `extract-calendario --comuni=bard,donnas,hone` estrae **una volta**
   dall'immagine e scrive tutte le copie da sé. Il momento in cui la
   duplicazione nasce è anche l'unico in cui è automatica, quindi
   dimenticarne una richiede di aver fatto il lavoro a mano apposta.
2. `sync` raggruppa i calendari per hash del contenuto e lo mostra a
   log: se Bard e Donnas condividono un semestre, si vedono in un solo
   gruppo, e il giorno in cui uno dei due cambia si vedono in due. È lo
   stesso dato letto in modo da rendere visibile una divergenza.

## Cosa sta qui: lo schema di raccolta

Quale flusso locale raccoglie quale materiale. Non è deducibile dal
filesystem e non è nel calendario stesso (che dice solo "ORGANICO,
giovedì"), quindi va dichiarato.

```toml
id = "sub-ato-e"
nome = "Sub-ATO E — Mont-Rose e Walser"
gestore = "TeknoService Italia"

[[comuni]]
id = "donnas"
nome = "Donnas"
materiali = { carta = "CARTA E CARTONI", cartone = "CARTA E CARTONI" }

[[frazioni]]
comune = "donnas"
id = "albard"
nome = "Albard"
materiali = { carta = "CARTA", cartone = "CARTONE" }
```

`[[frazioni]]` serve solo per dare un nome leggibile o per dichiarare
uno schema diverso da quello del comune: una frazione che ha soltanto un
calendario diverso è già dichiarata dalla propria cartella. Senza
`materiali`, una frazione eredita lo schema del proprio comune.

Un valore vuoto (`vetro = ""`) significa "non raccolto porta a porta
qui". `materiali` assente del tutto = nessuna mappatura, il materiale
vale direttamente come nome di categoria.

## Le fonti

`[[fonti]]` collega i Markdown curati alla pubblicazione del gestore da
cui provengono, così le risposte possono citarla.

```toml
[[fonti]]
file = "vocabolario/*.md"
nome = "Riciclabolario TeknoService Italia"
url = "https://.../riciclabolario.pdf"
```

Con il vecchio crawl l'URL si aveva "gratis", perché era il programma a
scaricare il file. Ora il PDF lo apre una persona, quindi l'URL va
dichiarato — una volta per pubblicazione, non per file. È anche più
onesto: un Markdown curato a mano non è *il* PDF, ne è una rilettura, e
la fonte da mostrare all'utente è la pubblicazione del gestore.

`tomllib` è nella libreria standard da Python 3.11: nessuna dipendenza
aggiunta per leggere il manifest.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_FILENAME = "area.toml"


def _parse_materiali(raw: object, contesto: str) -> tuple[tuple[str, str | None], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ManifestError(
            f"{contesto}: 'materiali' deve essere una tabella "
            'materiale = "CATEGORIA" (es. cartone = "CARTONE")'
        )
    voci: list[tuple[str, str | None]] = []
    for materiale, categoria in raw.items():
        chiave = str(materiale).strip().lower()
        if not chiave:
            raise ManifestError(f"{contesto}: materiale senza nome")
        # "" o false = dichiarato non raccolto porta a porta qui.
        # Distinto dall'assenza del materiale, che significa solo "non
        # previsto" e produce un messaggio diverso all'utente.
        if categoria is False or (isinstance(categoria, str) and not categoria.strip()):
            voci.append((chiave, None))
        else:
            voci.append((chiave, str(categoria).strip()))
    return tuple(voci)


@dataclass(frozen=True)
class FonteSpec:
    """Pubblicazione del gestore da cui proviene un gruppo di file.

    `file` è un glob sul percorso relativo all'area
    (`vocabolario/*.md`, `_calendari/bard/*.md`): i Markdown curati sono
    riletture di una pubblicazione, e più file vengono di solito dalla
    stessa. Il primo glob che combacia vince, quindi si dichiarano dal
    più specifico al più generale.
    """

    file: str
    nome: str
    url: str = ""

    def combacia(self, source: str) -> bool:
        return fnmatch.fnmatch(source, self.file)


@dataclass(frozen=True)
class FrazioneSpec:
    comune_id: str
    hamlet: str
    nome: str
    materiali: tuple[tuple[str, str | None], ...] = ()
    # False = dedotta dalla cartella dei calendari, non dichiarata nel
    # manifest. Serve a sapere se `materiali` vuoto significhi "eredita
    # dal comune" (sempre, per ora) o "dichiarato vuoto".
    dichiarata: bool = True

    def mappatura(self) -> dict[str, str | None]:
        return dict(self.materiali)


@dataclass(frozen=True)
class ComuneSpec:
    id: str
    nome: str
    materiali: tuple[tuple[str, str | None], ...] = ()

    def mappatura(self) -> dict[str, str | None]:
        return dict(self.materiali)


@dataclass(frozen=True)
class AreaManifest:
    id: str
    nome: str
    gestore: str
    comuni: tuple[ComuneSpec, ...]
    frazioni: tuple[FrazioneSpec, ...] = ()
    fonti: tuple[FonteSpec, ...] = ()
    percorso: Path = field(default=Path("."))

    @property
    def area_dir(self) -> Path:
        return self.percorso.parent

    def fonte_per(self, source: str) -> FonteSpec | None:
        """Pubblicazione di provenienza di un file, dal suo percorso
        relativo all'area. None se nessun glob combacia: la fonte è
        facoltativa, un'area senza `[[fonti]]` funziona come prima e le
        risposte semplicemente non la citano."""
        return next((f for f in self.fonti if f.combacia(source)), None)

    def comune(self, comune_id: str) -> ComuneSpec | None:
        return next((c for c in self.comuni if c.id == comune_id), None)

    def frazione(self, comune_id: str, hamlet: str) -> FrazioneSpec | None:
        return next(
            (
                f
                for f in self.frazioni
                if f.comune_id == comune_id and f.hamlet == hamlet
            ),
            None,
        )

    def mappatura_per(self, comune_id: str, hamlet: str = "") -> dict[str, str | None]:
        """Schema di raccolta effettivo per una destinazione. Una
        frazione senza `materiali` propri eredita quello del comune: è il
        caso normale (la maggior parte delle frazioni ha un calendario
        diverso ma lo stesso schema), e obbligare a ripeterlo
        moltiplicherebbe le occasioni di scriverlo in modo incoerente."""
        if hamlet:
            frazione = self.frazione(comune_id, hamlet)
            if frazione is not None and frazione.materiali:
                return frazione.mappatura()
        comune = self.comune(comune_id)
        return comune.mappatura() if comune else {}


class ManifestError(ValueError):
    """Manifest sintatticamente valido ma incoerente (es. una frazione
    assegnata a un comune non dichiarato). Sollevata al caricamento, non
    a metà ingestion: un errore di battitura in un id deve fermare tutto
    prima di aver scritto qualcosa su Postgres, non dopo."""


def nome_leggibile(identificatore: str) -> str:
    return identificatore.replace("_", " ").replace("-", " ").title()


def load(path: Path) -> AreaManifest:
    if not path.exists():
        raise ManifestError(
            f"manifest non trovato: {path}. Ogni area ha bisogno di un "
            f"{MANIFEST_FILENAME} che dichiari i comuni e il loro schema di "
            "raccolta — vedi il docstring di bordeus_ingest.manifest."
        )

    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    for chiave in ("id", "nome"):
        if not raw.get(chiave):
            raise ManifestError(f"{path}: campo obbligatorio {chiave!r} mancante")

    comuni: list[ComuneSpec] = []
    for voce in raw.get("comuni", []):
        comune_id = str(voce.get("id", "")).strip()
        if not comune_id:
            raise ManifestError(f"{path}: una voce [[comuni]] non ha 'id'")
        comuni.append(
            ComuneSpec(
                id=comune_id,
                nome=str(voce.get("nome", nome_leggibile(comune_id))),
                materiali=_parse_materiali(
                    voce.get("materiali"), f"{path}: comune {comune_id!r}"
                ),
            )
        )

    if not comuni:
        raise ManifestError(f"{path}: nessun comune dichiarato in [[comuni]]")

    comuni_noti = {c.id for c in comuni}

    frazioni: list[FrazioneSpec] = []
    for voce in raw.get("frazioni", []):
        comune_id = str(voce.get("comune", "")).strip()
        hamlet = str(voce.get("id", "")).strip()
        if not comune_id or not hamlet:
            raise ManifestError(
                f"{path}: una voce [[frazioni]] deve avere 'comune' e 'id' "
                '(es. comune = "donnas", id = "albard")'
            )
        if comune_id not in comuni_noti:
            raise ManifestError(
                f"{path}: la frazione {hamlet!r} è assegnata al comune "
                f"{comune_id!r}, che non è dichiarato in [[comuni]]"
            )
        frazioni.append(
            FrazioneSpec(
                comune_id=comune_id,
                hamlet=hamlet,
                nome=str(voce.get("nome", nome_leggibile(hamlet))),
                materiali=_parse_materiali(
                    voce.get("materiali"), f"{path}: frazione {comune_id}/{hamlet}"
                ),
            )
        )

    fonti: list[FonteSpec] = []
    for voce in raw.get("fonti", []):
        file = str(voce.get("file", "")).strip()
        nome = str(voce.get("nome", "")).strip()
        if not file or not nome:
            raise ManifestError(
                f"{path}: ogni voce [[fonti]] deve avere 'file' (un glob "
                "relativo alla cartella dell'area) e 'nome'"
            )
        fonti.append(
            FonteSpec(file=file, nome=nome, url=str(voce.get("url", "")).strip())
        )

    return AreaManifest(
        id=str(raw["id"]).strip(),
        nome=str(raw["nome"]).strip(),
        gestore=str(raw.get("gestore", "")).strip(),
        comuni=tuple(comuni),
        frazioni=tuple(frazioni),
        fonti=tuple(fonti),
        percorso=path.resolve(),
    )
