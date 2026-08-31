# bordeus-ingest

Ingestion **semi-automatica** della knowledge base di bordeus: dalle
fonti pubblicate dal gestore rifiuti a Postgres, passando per Markdown
curato a mano.

Due destinazioni, non una:

- **vector store** (`langchain-postgres`, una collection per area
  Sub-ATO) — vocabolario, guide, informazioni generali. Contenuto su cui
  la ricerca per similarità è lo strumento giusto: l'utente descrive un
  oggetto con parole sue e va trovata la voce corrispondente anche se il
  vocabolario la chiama diversamente.
- **tabella `raccolta_date`** — i calendari di raccolta. Contenuto usato
  tramite tool calling con una query indicizzata.

## Perché semi-automatica

La prima versione era una pipeline completamente automatica: scaricava
le pagine del gestore, ne seguiva i link a PDF, indovinava una categoria
con un'euristica a parole chiave e caricava tutto.
Tuttavia la complessità e l'eterogeneità dei PDF rendeva difficile
l'estrazione del testo (specialmente per i PDF dei calendari).

Il flusso ora è in tre passaggi, con una persona nel mezzo:

1. **estrazione** (`extract-vocabolario`, `extract-calendario`) — la
   macchina fa il lavoro meccanico e produce Markdown leggibile.
2. **rilettura umana** — si corregge quello che serve. È il passaggio
   che rende "semi" la semi-automazione, ed è il motivo per cui il
   formato intermedio è Markdown e non JSON: deve essere correggibile a
   mano da chi non sta guardando il codice.
3. **`sync`** — porta i Markdown in Postgres.

Gli estrattori si lanciano quando il gestore pubblica una revisione,
non ad ogni ingestion. `sync` si lancia ogni volta che i Markdown
cambiano.

## Comandi

```bash
# ingestion completa di un'area (guidata da knowledge/<area>/area.toml)
uv run bordeus-ingest sync --area=sub-ato-e

# ...svuotando prima la collection: necessario dopo un cambio di chunking
uv run bordeus-ingest sync --area=sub-ato-e --reset

# solo i calendari: salta il caricamento del modello di embedding, quindi
# correggere una data non costa il ricalcolo degli embedding dell'area
uv run bordeus-ingest sync --area=sub-ato-e --only=calendari

# solo il RAG
uv run bordeus-ingest sync --area=sub-ato-e --only=rag

# estrazione di una fonte nuova -> Markdown DA RILEGGERE
uv run bordeus-ingest extract-vocabolario \
    --pdf=https://www.teknoserviceitalia.com/.../riciclabolario.pdf \
    --out=knowledge/sub-ato-e/guide

uv run bordeus-ingest extract-calendario \
    --image=fonti/BARD-DONNAS-HONE.jpeg \
    --area=sub-ato-e --comuni=bard,donnas,hone --periodo=2026_semestre_2
```

## `knowledge/` e il manifest `area.toml`

```
knowledge/
└── sub-ato-e/
    ├── area.toml                       # anagrafica + schema di raccolta
    ├── guide/                          # RAG, condiviso dall'area (tipo = nome cartella)
    │   ├── A.md
    │   └── ...
    ├── info/                           # RAG, altre informazioni (orari ecocentro, ...)
    ├── _calendari/                     # -> raccolta_date, NON nel vector store
    │   ├── donnas/
    │   │   ├── 2026_semestre_2.md
    │   │   ├── 2027_semestre_1.md
    │   │   └── _frazioni/
    │   │       ├── albard/2026_semestre_2.md
    │   │       └── bondon/2026_semestre_2.md
    │   ├── bard/
    │   │   └── _frazioni/crous/...
    │   └── hone/
    └── _comuni/
        └── <comune_id>/<tipo>/*.md     # RAG specifico di UN comune
```

Il **`tipo`** di un documento RAG è il nome della cartella che lo contiene.
Le cartelle che iniziano con `_` sono di servizio e non diventano mai un `tipo`.

**Il legame calendario -> comune è il percorso**, non una dichiarazione:
`_calendari/donnas/` contiene i calendari di Donnas,
`_calendari/donnas/_frazioni/albard/` quelli della frazione Albard, che
hanno la precedenza. Per sapere quale calendario vale per un comune si
apre la sua cartella.

Il prezzo è la duplicazione: ad esempio Bard, Donnas e Hône condividono lo stesso
calendario.

- **`extract-calendario` scrive tutte le copie da sé**, da una sola
  estrazione (una sola chiamata al modello, che è la parte lenta)

- **`sync` raggruppa i calendari per hash del contenuto** e lo mostra

### Cosa dichiara il manifest

Solo l'anagrafica e lo **schema di raccolta**, che non è deducibile né
dal filesystem né dal calendario (che dice solo "ORGANICO, giovedì"):

```toml
[[comuni]]
id = "donnas"
nome = "Donnas"
materiali = { carta = "CARTA E CARTONI", cartone = "CARTA E CARTONI", ... }

[[frazioni]]
comune = "donnas"
id = "albard"
nome = "Albard"
materiali = { carta = "CARTA", cartone = "CARTONE", ... }
```

`[[frazioni]]` serve solo a dare un nome leggibile o a dichiarare uno
schema diverso: una frazione che ha soltanto un calendario diverso è già
dichiarata dalla propria cartella, e senza `materiali` eredita lo schema
del comune.

## Frazioni e fallback

`raccolta_date.hamlet` vale `''` per "calendario dell'intero comune".
In lettura (`bordeus_common.calendario`) la risoluzione è a due livelli:
se esiste anche una sola riga per la frazione dell'utente, quella
frazione ha un calendario proprio e si usa quello; altrimenti si ricade
sul comune. Una frazione senza override eredita quindi il calendario del
comune — che è il comportamento corretto, non un ripiego.

Il controllo è per frazione e non per (frazione, categoria): un override
reale è sempre un calendario completo, non una singola categoria
spostata di giorno.

## Materiali vs categorie di raccolta

Il vocabolario registra il **materiale** dell'oggetto (carta, cartone,
vetro, organico, plastica, metalli, indifferenziato), non il nome del
bidone. Il manifest dichiara, per ogni calendario, quale flusso di
raccolta locale copre quale materiale.

Il caso reale che lo impone: Bard, Donnas e Hône raccolgono carta e
cartone **insieme** (`CARTA E CARTONI`), ma le frazioni di Bard e
Donnas — Crous, Albard, Les Pians, Bondon — li raccolgono **separati**,
in due giorni diversi. Stesso comune, due schemi.

Il vocabolario è condiviso dall'intera area, quindi non può nominare la
categoria: quale sia dipende da chi sta chiedendo. Se lo facesse, un
utente di Albard si sentirebbe indicare un bidone che nel suo caso non
esiste, e la scelta fra i due flussi ricadrebbe sul modello — che ha in
mano solo la descrizione dell'oggetto e, sbagliando, restituirebbe
comunque una data plausibile.

L'asimmetria che conta: **unire è facile, dividere no.** Due comuni che
chiamano lo stesso bidone in modo diverso si risolvono con un alias. Ma
quando un comune divide `CARTA E CARTONI` in `CARTA` e `CARTONE`, il
nome della categoria non contiene l'informazione per scegliere: serve
sapere se l'oggetto è carta o cartone, e lo sa solo il vocabolario.

```toml
[[comuni]]
id = "donnas"
materiali = { carta = "CARTA E CARTONI", cartone = "CARTA E CARTONI", ... }

[[frazioni]]
comune = "donnas"
id = "albard"
materiali = { carta = "CARTA", cartone = "CARTONE", ... }
```

Lo schema sta sul comune e, quando serve, sulla frazione: lo stesso
comune può averne due, uno per il capoluogo e uno per le frazioni. Un
valore vuoto
(`vetro = ""`) significa "non raccolto porta a porta qui" — utile per un
comune che raccoglie il vetro solo all'ecocentro, e prima
indistinguibile da un errore di battitura. `materiali` assente del tutto
= nessuna mappatura, il materiale vale direttamente come nome di
categoria (il comportamento precedente: le aree con un solo schema non
devono dichiarare niente).

### Validazione dei nomi

Ogni categoria citata in `materiali` deve esistere in almeno uno dei
calendari di quella destinazione, o `sync` fallisce elencando i nomi
reali. Il controllo è sull'unione dei semestri, non su un file alla
volta: la mappatura descrive lo schema di un comune, mentre un singolo
file copre un semestre e può legittimamente non contenere una categoria. Serve contro una
trappola concreta: lo stesso gestore scrive `CARTA E CARTONE` nel
volantino "modalità di conferimento" e `CARTA E CARTONI` nella griglia
del calendario. Qui vale quello della griglia, perché è quello che
finisce in `raccolta_date`. Senza il controllo, l'errore non darebbe
un'eccezione ma un silenzio: nessuna data trovata per la carta, e
nessuno se ne accorge finché un utente non fa la domanda.

### Carta e cartone nel riciclabolario

⚠️ Il riciclabolario di TeknoService **non distingue** carta da cartone:
mette anche le scatole sotto "Carta". `extract-vocabolario` riflette
questo, quindi le voci di cartone (scatole, imballaggi, cassette della
frutta) vanno riclassificate **a mano** come `cartone` nel Markdown. Il
volantino del gestore elenca esattamente quali oggetti sono cartone: è
la fonte da cui copiare. Senza questa correzione, le frazioni che
dividono i due flussi ricevono il giorno sbagliato.

## Perché il calendario non è più nel RAG

Un semestre di calendario finisce in uno o due chunk da ~50 date in
testo libero. Recuperare quel chunk e metterlo nel prompt significa
chiedere al modello di confrontare date leggendole da un elenco: un
compito algoritmico esatto, in cui gli LLM sbagliano in modo
_plausibile_ — una data della settimana sbagliata sembra una risposta
corretta, e l'utente non ha modo di accorgersene.

Ora le date stanno in una tabella con un indice su
`(comune_id, hamlet, categoria, data)`, il confronto lo fa Postgres
(`ORDER BY data ASC LIMIT 1`) e il modello riceve una sola data già
corretta chiamando uno strumento. Vedi
`common/src/bordeus_common/calendario.py`,
`bot/src/bordeus_bot/calendario.py` e
`migrations/0003_raccolta_date.sql`.

## Chunking — dove nasce la qualità del retrieval

`MarkdownHeaderTextSplitter` sui confini `#`/`##`, non
`MarkdownTextSplitter`. Quest'ultimo è in realtà un
`RecursiveCharacterTextSplitter` con i separatori del Markdown:
_preferisce_ tagliare sulle intestazioni ma resta vincolato a
`chunk_size`, e se lo spazio residuo finisce a ridosso di
un'intestazione la include e taglia subito dopo — l'intestazione resta
in coda al chunk precedente, separata dal contenuto che descrive.

### RAG Vocabolario: una riga = un chunk

Le tabelle di questo progetto (il vocabolario oggetto → conferimento)
sono elenchi di **record indipendenti**, e.g., "Tazzina in ceramica → RUR".

Metterne più di una nello stesso chunk rompe il retrieval in modo
misurabile.

L'oggetto e la categoria finiscono anche nei metadata (`oggetto`,
`conferimento`), utili per ispezionare cosa è stato recuperato senza
riparsare il testo.

### ⚠️ Dopo un cambio di chunking serve `--reset`

L'id di un chunk include il suo contenuto, quindi l'upsert aggiorna
quelli che ricalcola identici ma **non cancella** quelli che l'ingestion
non produce più. Cambiando strategia di chunking (o togliendo una voce
da un Markdown) i vecchi chunk restano nella collection e continuano a
competere nel retrieval — con il risultato che una correzione ai dati
sembra non avere effetto, o ne ha metà.

```bash
uv run bordeus-ingest sync --area=sub-ato-e --reset
```

## Debug del retrieval

`notebooks/ingest.ipynb` esegue la pipeline stadio per stadio e include
la funzione `diagnosi()`, che per una descrizione d'oggetto mostra cosa
recupera il bot e in che posizione esce la voce attesa. Serve a separare
tre problemi che dall'esterno si assomigliano:

- **la voce non c'è nei chunk** → problema di dati, si corregge il
  Markdown;
- **c'è ma esce oltre i primi `k`** → problema di retrieval: chunk troppo
  grossi, o `k` troppo basso;
- **è nei primi `k` ma la risposta è sbagliata** → problema di prompt, e
  si continua in `bot/notebooks/rag_eval.ipynb`.

Sul bot, `LOG_LEVEL=TRACE` mostra la stessa cosa in esercizio: query,
filtro, chunk recuperati con la loro fonte, e prompt assemblato.

## Estrazione del calendario da immagine

I gestori pubblicano i calendari come griglie a colori: una colonna per
mese, una riga per giorno, categoria indicata dal colore della cella
oltre che dal testo. L'OCR classico legge il testo e perde la struttura,
che è proprio l'informazione che serve — è la posizione della cella
nella colonna a dire di che giorno si tratta. Sono stati provati
`unstructured`, `paddleocr` e `rapidocr`: nessuno risolve il problema,
e tutti costavano diversi GB di dipendenze. Sono stati rimossi.

La strada scelta è un modello multimodale con **output strutturato**
(`response_format` con uno schema Pydantic): il modello non può
rispondere in prosa, deve riempire i campi. Non garantisce che le date
siano giuste, garantisce che siano processabili.

Endpoint OpenAI-compatibile, configurato da ambiente
(`INGEST_VISION_BASE_URL`, `INGEST_VISION_MODEL`,
`INGEST_VISION_API_KEY`): funziona con Ollama in locale o con un
provider remoto senza toccare il codice. Il modello deve supportare sia
le immagini sia `response_format` — uno che non lo fa fallisce con un
errore esplicito.

**Rileggere sempre l'output.**

## Setup

Richiede [uv](https://docs.astral.sh/uv/) e Postgres con `pgvector` in
esecuzione (il `docker-compose.yml` in `../deployment/` basta).

```bash
uv sync                    # installa runtime + dev (notebook, viz) di default
cp .env.example .env       # e compila DATABASE_URL, opzionalmente EMBEDDING_MODEL
```

`uv sync --no-dev` installa solo le dipendenze runtime (niente
Jupyter/matplotlib) — vedi "Dipendenze: runtime vs dev" più sotto.

### Aprire il notebook con il kernel giusto

`uv sync` da solo non basta per il notebook: va anche aperto con
l'**interprete Python del workspace**, non un Python globale né un
kernel scollegato dal progetto.

`uv.lock` e `.venv` vivono alla radice del workspace (`../.venv`), non
dentro `ingestion/` — anche lanciando `uv sync` da qui dentro. È il
comportamento previsto per un workspace `uv` (come questo, membri
`common`/`ingestion`/`bot`): un solo ambiente condiviso per tutti i
membri, non uno per cartella. Garantisce che `ingestion` e `bot`
installino esattamente la stessa versione di `torch` — la ragione
stessa per cui il workspace esiste (vedi `../pyproject.toml`).

Un errore comune se si lancia `uv sync` (senza `--all-packages`) dalla
radice del workspace invece che da dentro `ingestion/`: quel comando
installa solo le dipendenze del progetto radice (vuote), non quelle di
`ingestion/` — per cui pacchetti come `pdfplumber` risultano
"non trovati" con un `ModuleNotFoundError` pur avendo lanciato `uv sync`
con successo altrove.

**Modo più semplice**, lanciare Jupyter con `uv run` da dentro
`ingestion/`: `uv` risolve da solo l'ambiente giusto del workspace,
anche se non è fisicamente in questa cartella.

```bash
cd ingestion
uv sync                          # se non fatto già
uv run jupyter lab notebooks/
```

**Se usi VS Code** (o un altro editor che si connette a un kernel
esistente invece di lanciare Jupyter lui stesso): registra un kernel
dedicato e selezionalo dall'interfaccia invece di un "Python 3"
generico:

```bash
cd ingestion
uv run python -m ipykernel install --user --name=bordeus-ingest \
    --display-name="bordeus (ingestion)"
```

poi, nel notebook, scegli "bordeus (ingestion)" dal selettore di
kernel in alto a destra (o "Select Kernel" in VS Code).

**Verifica rapida** che il kernel giusto sia attivo, da una cella del
notebook:

```python
import sys
print(sys.executable)  # deve puntare dentro <radice del repo>/.venv/bin/python
```

Se stampa un percorso diverso (es. un Python di sistema, un conda env,
o un venv creato "a mano" dentro `ingestion/`), è quello il problema —
cambia kernel, non serve reinstallare nulla.

### GPU

Se hai una GPU NVIDIA, `torch` è pinnato `<2.8` con indice CUDA 12.6
nel `pyproject.toml` alla radice del workspace (necessari insieme per
le GPU Pascal, es. GTX 1080 — aggiorna il pin e l'indice per GPU più
recenti). `torch` è dichiarato come dipendenza **diretta** in
`common/pyproject.toml` (non solo transitiva via
`bordeus-common`/`sentence-transformers`): affidarsi solo alla
risoluzione transitiva può risultare in un'installazione priva di
`torch`. `ingestion` non ha bisogno diretto di `torch` (lo riceve
transitivamente via `bordeus-common`, per gli embedding).

Vedi i commenti in `../pyproject.toml` e `common/pyproject.toml` per i
dettagli, e `uv run python -c "import torch; print(torch.cuda.is_available())"`
per verificare che la GPU sia vista. Senza GPU, `torch` gira comunque su
CPU (più lento, ma funzionante).

## Modello di embedding

Configurabile con la variabile d'ambiente `EMBEDDING_MODEL` in `.env`
(default: `microsoft/harrier-oss-v1-0.6b`, multilingua — 94 lingue,
italiano incluso — **1024 dimensioni**). `bordeus_common.embed.get_embeddings()`
verifica esplicitamente che il modello scelto produca vettori a 1024
dimensioni e solleva un errore chiaro altrimenti — un mismatch
scriverebbe silenziosamente dati incompatibili nella collection
Postgres. Cambiare dimensione richiede aggiornare `EMBEDDING_DIM` in
`common/src/bordeus_common/embed.py`.

`microsoft/harrier-oss-v1-0.6b` è un modello **decoder-only** con
last-token pooling: le **query** vanno prefissate con un'istruzione in
linguaggio naturale per ottenere buoni risultati (confermato dalla FAQ
della model card ufficiale), i **documenti** no. Qui in ingestion
embeddiamo solo documenti/chunk, quindi non serve nessun prompt — il
prefisso istruzione lo applica il bot lato query (vedi
`../bot/README.md`, `QUERY_INSTRUCTION` in `bot/rag.py`).

Il primo avvio scarica i pesi del modello da Hugging Face Hub; le
esecuzioni successive usano la cache locale (`~/.cache/huggingface`).

## Dipendenze: runtime vs dev

- **`[project.dependencies]`** — lettura dei Markdown, chunking,
  scrittura su Postgres, più `pdfplumber`/`openai` per gli estrattori.
  Installato sempre.
- **`[dependency-groups] dev`** — `jupyterlab`/`ipykernel` (notebook) e
  `matplotlib`/`scikit-learn` (per `viz.py`, import lazy al suo interno).
  Installato di default da `uv sync`, escluso con `uv sync --no-dev`.

## Struttura

Codice condiviso con il bot (embedding, vector store, schema/accesso
dati, tabella dei calendari) non è qui: vive in
`../common/src/bordeus_common/` — vedi `../README.md` per la struttura
completa del workspace.

```
pyproject.toml
.env.example
src/bordeus_ingest/
├── __init__.py        # CLI: sync, extract-vocabolario, extract-calendario
├── manifest.py        # area.toml: comuni, frazioni, schema di raccolta
├── knowledge.py       # convenzioni della cartella knowledge/ (tipo = nome cartella)
├── documents.py       # scoperta dei Markdown RAG -> Document di LangChain
├── chunk.py           # MarkdownHeaderTextSplitter + fallback consapevole delle tabelle
├── calendario.py      # parsing dei Markdown di calendario -> raccolta_date
├── pipeline.py        # orchestrazione: anagrafica, calendari, RAG
├── viz.py             # riduzione t-SNE 2D/3D degli embedding (strumento di ispezione)
└── extract/
    ├── vocabolario.py # PDF del riciclabolario -> un Markdown per lettera
    └── calendario.py  # immagine di un calendario -> Markdown (modello multimodale)
knowledge/
└── sub-ato-e/         # dati curati, versionati: sono il contenuto, non uno scarico temporaneo
notebooks/
├── ingest.ipynb               # pipeline passo per passo + debug del retrieval
├── ingest_vocabolario.ipynb   # prototipi da cui nascono i moduli in extract/
├── ingest_calendario.ipynb
├── ingest_info_generali.ipynb
└── test_ingestion_v2.ipynb
```

`knowledge/` **non** è più in `.gitignore`: nella versione precedente
conteneva file scaricati, ricostruibili rilanciando il crawl. Ora
contiene Markdown curato a mano, che è lavoro umano non riproducibile
automaticamente — va versionato come il codice.
