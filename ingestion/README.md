# bordeus-ingest

Pipeline di ingestion RAG per bordeus, costruita attorno a LangChain per
ogni passaggio: dato l'URL (o gli URL) di un'area di gestione rifiuti
della Valle d'Aosta, scarica pagina HTML e file PDF/Markdown linkati in
una cartella locale `knowledge/`, li carica con i loader di LangChain,
li spezza in chunk e li scrive su Postgres/pgvector tramite
l'integrazione `langchain-postgres`.

**Il bot Telegram (`../bot/`, anch'esso Python) resta il runtime di
inferenza**, e legge dallo stesso vector store scritto qui — un solo
schema, un solo linguaggio, vedi la struttura del progetto in
`../README.md`.

Codice condiviso con il bot (embedding, vector store, schema/accesso
dati) vive in `../common/`, non qui — vedi `../common/pyproject.toml` e
i commenti nei rispettivi moduli.

## Gli embedding sono raggruppati per area Sub-ATO, non per comune

Dal 2024 la Valle d'Aosta è divisa in 5 aree Sub-ATO (Sotto-Ambito
Territoriale Ottimale), ciascuna con un gestore e proprie guide
operative. Un gestore può servire **più aree con contenuti diversi** —
es. Quendoz gestisce sia il Sub-ATO C (Aosta) sia il D (Mont-Cervin ed
Évançon), con pagine guida separate, probabilmente per calendari/
ecocentri specifici di zona. L'**area**, non il nome del gestore, è
quindi la chiave giusta per raggruppare gli embedding:
`collection_name = area_id` (non `comune_id`), con più comuni della
stessa area che condividono la stessa collection invece di duplicare
gli stessi chunk per ciascuno.

Un'area può avere più di una fonte da ingerire in un colpo solo
(`--url`, ripetibile): tipico quando un gestore pubblica sia una pagina
specifica dell'area sia un contenuto condiviso da tutte le sue aree (es.
il vocabolario di Quendoz, https://www.quendoz.it/vocabolario/ — un
glossario oggetto→categoria di smaltimento uguale per ogni area che
gestisce). Includerlo nell'ingestion di ogni area invece che tenerlo in
una collection a parte evita la complessità di un retriever
multi-collection, al costo di duplicare quel contenuto specifico (non
tutta la guida) tra le poche aree dello stesso gestore — un compromesso
ragionevole per un progetto di questa scala.

## Contenuto specifico di un comune: `--comune-url`

Non tutto il contenuto di un'area è davvero condiviso da tutti i comuni
che la compongono. Caso reale: TeknoService Italia (Sub-ATO E) pubblica
un calendario di raccolta porta a porta **diverso per ciascun comune**
— Donnas e Pont-Saint-Martin, comuni confinanti nella stessa area, hanno
giorni di raccolta diversi per motivi logistici, pur sotto lo stesso
gestore. Mettere quel calendario nella collection condivisa dell'area
sarebbe peggio di uno spreco di spazio: sarebbe disinformazione (un
utente di Donnas che riceve il calendario di Pont-Saint-Martin).

`--comune-url=id:url` (ripetibile, anche più volte per lo stesso
comune) ingerisce una fonte specifica di **un solo comune**. I chunk
risultanti restano nella stessa collection dell'area (non una collection
per comune da interrogare e unire — `langchain-postgres` supporta un
filtro `$or` sui metadata, vedi sotto), solo taggati con `comune_id`
nel metadata. Nessun vincolo su quali categorie possano essere
specifiche di un comune (calendario, moduli, o altro): lo decide chi
lancia l'ingestion, caso per caso, in base a come il gestore reale
organizza davvero i contenuti.

```bash
uv run bordeus-ingest --sub-ato=sub-ato-e:"Sub-ATO E — Mont-Rose e Walser" \
    --gestore="TeknoService Italia" \
    --url=https://www.teknoserviceitalia.com/vocabolario \
    --comune=donnas:Donnas --comune=pont-saint-martin:"Pont-Saint-Martin" \
    --comune-url=donnas:https://www.teknoserviceitalia.com/calendario-donnas.pdf \
    --comune-url=pont-saint-martin:https://www.teknoserviceitalia.com/calendario-pont-saint-martin.pdf
```

In fase di retrieval, il bot filtra automaticamente: un utente vede
sempre il contenuto condiviso dell'area **più** quello specifico del
proprio comune, **mai** quello di un comune vicino — vedi
`bot/README.md`, `rag.comune_filter`.

**Fonti dirette PDF/Markdown**: `--comune-url` (e anche `--url`)
funzionano anche se l'URL punta direttamente a un PDF o Markdown, non
solo a una pagina HTML che li linka — rilevato dall'estensione
dell'URL. Necessario proprio per il caso TeknoService: il calendario di
un comune è spesso un PDF pubblicato da solo, non linkato da nessuna
pagina.

## Come funziona: un'area, uno o più comuni, uno o più URL

0. **INSERT/UPSERT di area e comuni**
   (`bordeus_common.db.upsert_sub_ato`/`upsert_comune`) — l'area e
   ciascun comune che vi appartiene vengono creati o aggiornati in
   Postgres (`ON CONFLICT (id) DO UPDATE`, tabelle condivise con il
   bot — vedi `../migrations/`), non serve che esistano già.
1. **Fetch → `knowledge/`** (`pipeline.fetch_to_knowledge`) — per
   ciascun URL sorgente, scarica la pagina HTML e tutti i PDF/Markdown
   linkati (un giro di rete per URL), li classifica per categoria
   (`classify.guess_doc_type`: calendario, guide, moduli, servizi, altro
   — euristica a parole chiave sul contenuto reale, non solo sul nome
   del file, vedi sotto), e li salva su
   `knowledge/<area_id>/<categoria>/`, con un unico `manifest.json`
   per l'area che accumula le voci di tutti gli URL sorgente.
2. **Load** (`loaders.load_knowledge`) — carica i file dalla cartella
   `knowledge/` dell'area con i loader di LangChain: `BSHTMLLoader` per
   l'HTML, `PDFPlumberLoader` per i PDF (un `Document` per pagina),
   `TextLoader` per i Markdown. Ogni `Document` viene arricchito con i
   metadati del manifest (area_id, source_url, kind, tipo).
3. **Split** (`chunk.split_documents`) — `MarkdownTextSplitter` per i
   documenti Markdown nativi (rispetta header e blocchi di codice),
   `RecursiveCharacterTextSplitter` per tutto il resto (testo estratto
   da HTML/PDF, senza struttura Markdown da preservare).
4. **Embed + scrittura** (`bordeus_common.vectorstore.add_chunks`) —
   `PGVector` di `langchain-postgres` calcola gli embedding (tramite
   l'`Embeddings` di `bordeus_common.embed.get_embeddings()`) e scrive
   su Postgres in un'**unica collection per l'area**
   (`collection_name = area_id`), con `id` deterministici per un
   upsert idempotente (ri-lanciare sullo stesso URL aggiorna, non
   duplica).
5. **Visualizzazione** (`viz.py`) — riduzione t-SNE a 2D o 3D degli
   embedding scritti, per ispezionarli visivamente.
6. **Retriever** — `vectorstore.similarity_search(domanda, filter=...)`,
   la stessa chiamata che usa il bot per rispondere (con un filtro sul
   comune specifico dell'utente — vedi `bot/README.md`,
   `rag.comune_filter`). Nel notebook, senza filtro, si vede tutto il
   contenuto dell'area indistintamente.

Per l'uso non interattivo:

```bash
uv run bordeus-ingest \
    --sub-ato=sub-ato-e:"Sub-ATO E — Mont-Rose e Walser" \
    --gestore="TeknoService Italia" \
    --url=https://www.teknoserviceitalia.com/rifiuti \
    --comune=donnas:Donnas \
    --comune=bard:Bard \
    --comune=champorcher:Champorcher

# Un'area con più fonti (gestore in ATI su più aree, contenuto condiviso):
uv run bordeus-ingest \
    --sub-ato=sub-ato-d:"Sub-ATO D — Mont-Cervin ed Évançon" \
    --gestore="Quendoz" \
    --url=https://www.quendoz.it/category/subato-d/ \
    --url=https://www.quendoz.it/vocabolario/ \
    --comune=montjovet:Montjovet \
    --comune=verres:Verrès
```

`--comune` e `--url` sono entrambi ripetibili; il formato di
`--sub-ato`/`--comune` è `id:Nome` — se ometti `:Nome` viene usato lo
id anche come nome.

## Classificazione dei PDF: anche dal contenuto, non solo dal nome file

I nomi dei PDF pubblicati dai gestori sono spesso codici criptici (es.
`FRAZ-BARD-DONNAS.pdf`, `UND-SPECIALI-BARD-DONNAS-HONE.pdf`) da cui è
impossibile indovinare la categoria. Per l'HTML e il Markdown la
classificazione (`classify.guess_doc_type`) usa già un campione del
contenuto reale (rispettivamente il testo pulito della pagina e l'intero
file); per i PDF si applica lo stesso principio.

`fetch.pdf_text_preview` estrae rapidamente le prime pagine di un PDF
(`pdfplumber` — già una dipendenza per `PDFPlumberLoader`, niente da
aggiungere; con il parametro `pages` limita l'apertura alle sole pagine
richieste, non tutto il documento) e usa quel campione per
`classify.guess_doc_type`, non per il contenuto RAG vero e proprio
(quello resta compito di `PDFPlumberLoader` in `loaders.py`, che estrae
l'intero documento).

Silenziosa sugli errori (PDF corrotto, protetto, scansione senza testo
estraibile): restituisce stringa vuota invece di far fallire la
classificazione — al peggio si ricade sul solo URL.

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
uv run jupyter lab notebooks/ingest.ipynb
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

## Perché non Docling

`ingestion/` usa `BSHTMLLoader`/`PDFPlumberLoader` invece di
[Docling](https://github.com/docling-project/docling), che converte
HTML/PDF in Markdown strutturato (tabelle, header preservati) invece del
testo piatto. Qualunque installazione utile di Docling richiede
incondizionatamente `docling-ibm-models` (il modello di
reading-order/layout usato dalla pipeline standard) e quindi
`torch`+`torchvision` — non evitabile scegliendo un backend PDF più
leggero, dato che `docling.document_converter` importa quella pipeline
a livello di modulo indipendentemente dal backend scelto. Per le guide
dei gestori, PDF generati digitalmente e non scansioni, l'estrazione più
semplice di `BSHTMLLoader`/`PDFPlumberLoader` è sufficiente (le tabelle
diventano testo su più righe invece di Markdown strutturato, ma il
contenuto resta leggibile e ricercabile), senza il costo di installare
diversi GB di dipendenze ML aggiuntive.

## Verificare come sono stati salvati i chunk

Query dirette su Postgres, utili dopo un'ingestion per controllare che i
chunk siano finiti nella categoria e nel comune giusti — specialmente
per il calendario, dove un chunk finito "condiviso" invece che specifico
di un comune (o viceversa) è un errore silenzioso: nessuna eccezione,
solo una risposta sbagliata data all'utente sbagliato più avanti.

```sql
-- Dettaglio dei chunk classificati come "calendario": area, comune
-- specifico (vuoto = condiviso dall'area), fonte, e un'anteprima del
-- contenuto.
SELECT
    c.name                          AS area,
    e.cmetadata->>'comune_id'       AS comune,
    e.cmetadata->>'source_url'      AS fonte,
    e.cmetadata->>'kind'            AS tipo_file,
    left(e.document, 200)           AS anteprima
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
WHERE e.cmetadata->>'tipo' = 'calendario'
ORDER BY c.name, e.cmetadata->>'comune_id';

-- Riepilogo: quanti chunk di calendario per area/comune — utile per
-- individuare a colpo d'occhio anomalie (es. un comune con 0 chunk
-- quando te ne aspettavi, o chunk finiti "condivisi" quando dovevano
-- essere specifici di un comune).
SELECT
    c.name AS area,
    COALESCE(NULLIF(e.cmetadata->>'comune_id', ''), '(condiviso, nessun comune)') AS comune,
    count(*) AS numero_chunk
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
WHERE e.cmetadata->>'tipo' = 'calendario'
GROUP BY c.name, comune
ORDER BY c.name, comune;
```

Lanciabili con `psql "$DATABASE_URL" -f query.sql`, o sostituendo
`WHERE e.cmetadata->>'tipo' = 'calendario'` con un'altra categoria
(`'guide'`, `'moduli'`, `'servizi'`, `'altro'`) per ispezionare il resto
della knowledge base allo stesso modo.

## Dove finiscono i documenti scaricati: `knowledge/`

```
knowledge/
└── <area_id>/
    ├── manifest.json          # url originale, tipo file, categoria, comune_id — per ogni file
    ├── calendario/             # contenuto CONDIVISO dall'area (da --url)
    │   └── calendario-raccolta-generale.pdf
    ├── guide/
    │   ├── regolamento.pdf
    │   └── vocabolario.html
    ├── moduli/
    │   └── richiesta-cassonetto.pdf
    ├── servizi/
    │   └── faq-ecocentro.md
    └── _comuni/                # contenuto SPECIFICO di un comune (da --comune-url)
        ├── donnas/
        │   └── calendario/
        │       └── calendario-donnas.pdf
        └── pont-saint-martin/
            └── calendario/
                └── calendario-pont-saint-martin.pdf
```

`knowledge/` è nel `.gitignore` (dati scaricati, non codice) — si
ricostruisce rilanciando `fetch_to_knowledge`/`run_sub_ato`. La
categoria è stimata da `classify.guess_doc_type` (vedi sopra per i PDF):
volutamente semplice, facile da estendere aggiungendo voci in
`classify.py` se in futuro serve distinguere meglio.

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

- **`[project.dependencies]`** — fetch, load, split, scrittura su
  Postgres: tutto ciò che serve a `bordeus-ingest` da riga di comando.
  Installato sempre.
- **`[dependency-groups] dev`** — `jupyterlab`/`ipykernel` (notebook) e
  `matplotlib`/`scikit-learn` (per `viz.py`, import lazy al suo interno).
  Installato di default da `uv sync`, escluso con `uv sync --no-dev`.

## Struttura

Codice condiviso con il bot (embedding, vector store, schema/accesso
dati) non è qui: vive in `../common/src/bordeus_common/` — vedi
`../README.md` per la struttura completa del workspace.

```
pyproject.toml
.env.example
src/bordeus_ingest/
├── __init__.py       # entry point CLI (bordeus-ingest): --sub-ato, --gestore, --url (ripetibile), --comune (ripetibile)
├── classify.py       # euristica di categorizzazione (calendario/guide/moduli/servizi/altro)
├── knowledge.py       # gestione cartella knowledge/: nomi file (senza collisioni tra URL diversi), manifest
├── fetch.py            # fetch via HTTP (pagina HTML + scoperta link, PDF, Markdown) + anteprima PDF per classificazione
├── loaders.py            # step 2: caricamento con BSHTMLLoader/PDFPlumberLoader/TextLoader
├── chunk.py                # step 3: chunking (MarkdownTextSplitter o RecursiveCharacterTextSplitter, a seconda del kind)
├── viz.py                    # step 5: riduzione t-SNE 2D/3D, lettura da langchain-postgres
└── pipeline.py                # orchestrazione: fetch_source/save_to_knowledge, run_sub_ato multi-URL/multi-comune
notebooks/
└── ingest.ipynb                  # walkthrough interattivo, incluso multi-comune e retriever
```
