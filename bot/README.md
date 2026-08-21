# bordeus-bot

Bot Telegram di bordeus per la Valle d'Aosta: onboarding del comune,
identificazione dell'oggetto da una foto (vision), risposta sulle
modalità di smaltimento (RAG) — tutto in Python, riusando
`Embeddings`/`PGVector` condivisi con la pipeline di ingestion
(`../ingestion`, `../common`).

## Come funziona

1. **Onboarding** (`/start` o `/comune`): bottone nativo Telegram
   "condividi posizione" (reverse geocoding via Nominatim) o nome del
   comune scritto a mano, con normalizzazione tollerante ad
   accenti/maiuscole. Il comune deve essere già stato creato dalla
   pipeline di ingestion (`../ingestion`), non c'è un meccanismo di seed
   lato bot.
2. **Conferma esplicita**: una volta risolto il comune, il bot NON
   completa subito l'onboarding — mostra il Sub-ATO/gestore risolto
   (es. "Donnas fa parte del Sub-ATO E — Mont-Rose e Walser, gestito da
   TeknoService Italia. È corretto?") con due bottoni inline (Sì/No).
   Solo un tap su "Sì" completa l'onboarding; "No" torna alla richiesta
   iniziale. Lo stato intermedio (comune risolto ma non confermato) è
   tracciato in `users.pending_comune_id` (vedi
   `common/src/bordeus_common/db.py`), non solo in memoria — sopravvive
   a un riavvio del bot.
3. **Estrazione della descrizione dell'oggetto**: prima di interrogare
   il vector store, il bot estrae SOLO la descrizione dell'oggetto —
   mai l'intera frase dell'utente, che introdurrebbe rumore nella
   ricerca per similarità (saluti, formule di cortesia, la
   formulazione della domanda). Due percorsi, stesso principio, ed
   entrambi possono non riconoscere nessun oggetto — in quel caso il
   bot lo dice esplicitamente e chiede un nuovo input, senza fare
   nessuna ricerca RAG su una descrizione vuota o inventata:
   - **Foto**: scaricata, convertita in base64 e mandata a un modello
     multimodale via Ollama (`langchain_ollama.ChatOllama`) per
     identificare l'oggetto principale
     (`identify.identify_object_from_photo`). Se la foto è sfocata,
     mostra più oggetti ugualmente in primo piano senza un soggetto
     chiaro, o non contiene nessun oggetto riconoscibile, il bot chiede
     di scattarne un'altra o di descrivere l'oggetto a parole.
   - **Testo**: una chiamata Ollama separata estrae solo la descrizione
     dell'oggetto dal messaggio (`identify.identify_object_from_text`) —
     es. da "Ciao, dove butto una bottiglia di plastica?" estrae
     "bottiglia di plastica", scartando il resto. Se il messaggio non
     menziona nessun oggetto specifico (un saluto, una domanda fuori
     tema), il bot chiede di riformulare.
4. **RAG a due fasi**: la descrizione dell'oggetto (mai la frase intera
   dell'utente, foto o testo che sia) diventa la query sul vector store
   dell'**area Sub-ATO** del comune dell'utente (`rag.py`,
   `answer_question`) — stesso vector store scritto dall'ingestion
   (`langchain-postgres`, collection isolata per area, non per comune:
   più comuni della stessa area condividono lo stesso vector store,
   cachato una volta per area — vedi `Service.vectorstore_for_comune`
   in `telegram_bot.py`). Il
   retrieval avviene in due passaggi, non uno solo:
   1. **Vocabolario** (`rag.vocabolario_filter`): la descrizione
      dell'oggetto cerca qui — recupera come/dove smaltire l'oggetto
      (es. "bottiglia di plastica → bidone giallo").
   2. **Calendario** (`rag.calendario_filter`): usa come query il testo
      recuperato al passaggio 1 (non la descrizione dell'oggetto) per
      trovare il giorno di raccolta pertinente — il vocabolario di
      solito nomina il materiale, un match migliore contro il
      calendario. Così la risposta include sempre anche il giorno di
      passaggio quando disponibile, non solo se l'utente lo chiede
      esplicitamente.

   Entrambi i passaggi applicano un filtro `$or`
   (`rag.comune_filter`) sui metadata dei chunk: contenuto condiviso
   dell'area PIÙ quello specifico del comune dell'utente, mai quello di
   un comune vicino (caso reale: TeknoService Italia pubblica calendari
   di raccolta diversi per Donnas e Pont-Saint-Martin, comuni confinanti
   — vedi `ingestion/README.md`, sezione "Contenuto specifico di un
   comune"). Entrambi i contesti (vocabolario + calendario) finiscono
   nel system prompt, che istruisce il modello a menzionare sempre il
   giorno quando pertinente e a non inventarlo quando non lo è (es.
   oggetti destinati all'ecocentro, non al porta a porta) — e a
   rispondere ESCLUSIVAMENTE in base al contesto recuperato, non con
   conoscenza generica sullo smaltimento rifiuti che il modello
   potrebbe già avere: le regole cambiano da comune a comune e da
   gestore a gestore, una risposta plausibile ma generica può essere
   sbagliata per l'utente specifico.

5. **Multilingua**: pensato soprattutto per i turisti — la Valle
   d'Aosta ne accoglie molti, spesso per periodi più lunghi dei
   residenti con l'italiano come lingua madre. La lingua è sempre presa
   da `update.effective_user.language_code` (l'impostazione del client
   Telegram dell'utente, non rilevata dal testo del messaggio: più
   affidabile su frasi brevi, e funziona anche per le foto, dove non
   c'è testo da analizzare), con fallback all'italiano per lingue non
   tradotte o non dichiarate. Due parti distinte, entrambe coperte:
   - **Messaggi statici del bot** (onboarding, conferma, errori,
     bottoni, messaggio di attesa — 21 messaggi in tutto): tradotti
     esplicitamente in italiano, francese, inglese, spagnolo e tedesco
     (`i18n.py`, funzione `t()`).
   - **Risposte generate dall'LLM** (il RAG vero e proprio): il
     `SYSTEM_PROMPT_TEMPLATE` (vedi `rag.py`) include un'istruzione
     esplicita a rispondere sempre nella lingua dell'utente, **anche se
     il contesto recuperato è in italiano** (le guide dei gestori lo
     sono quasi sempre) — non una traduzione a parte con una seconda
     chiamata, la generazione avviene già nella lingua giusta nella
     stessa chiamata a Ollama.

## Setup

Richiede [uv](https://docs.astral.sh/uv/), Postgres con `pgvector` in
esecuzione (`../deployment/docker-compose.yml`), **Ollama** in esecuzione
con un modello multimodale scaricato (`ollama pull gemma3:...` o
equivalente — qualunque modello con supporto immagini funziona, basta
configurarne il nome in `.env`), e almeno un comune già popolato dalla
pipeline di ingestion.

```bash
cd bot
cp .env.example .env   # compila TELEGRAM_BOT_TOKEN, DATABASE_URL, OLLAMA_*
uv sync
uv run bordeus-bot
```

`uv sync` risolve anche `bordeus-common` (dipendenza di workspace, vedi
`../pyproject.toml`) — stesso `Embeddings`, stesso `PGVector`, stessa
build di `torch` (dipendenza diretta di `common/pyproject.toml`, con
l'override di sorgente CUDA dichiarato una volta sola a livello di
workspace — non va ripetuto qui).

## Valutare RAG e confrontare system prompt

`notebooks/rag_eval.ipynb` — notebook interattivo per testare qualità
del retrieval e confrontare varianti di system prompt sullo stesso
vector store del bot, senza passare da Telegram per ogni iterazione.
Riusa il codice reale di `rag.py` (`QUERY_INSTRUCTION`, `comune_filter`,
`SYSTEM_PROMPT_TEMPLATE`), non lo duplica.

```bash
cd bot
uv sync                          # installa anche Jupyter/pandas (gruppo dev)
uv run jupyter lab notebooks/rag_eval.ipynb
```

Stessa nota su `uv.lock`/`.venv` alla radice del workspace e sulla
selezione del kernel giusto in `ingestion/README.md`, sezione "Aprire il
notebook con il kernel giusto" — vale identica anche qui.

## Query asimmetriche (harrier-oss-v1)

`microsoft/harrier-oss-v1-0.6b` è un modello decoder-only: le **query**
vanno prefissate con un'istruzione in linguaggio naturale per ottenere
buoni risultati, i **documenti** no (confermato dalla FAQ della model
card). L'ingestion (che embedda solo documenti) non ne ha bisogno; il
bot sì — vedi `QUERY_INSTRUCTION` in `rag.py`, passata a
`bordeus_common.embed.get_embeddings(query_instruction=...)`, che la
applica soltanto a `embed_query` (mai a `embed_documents`) tramite
`query_encode_kwargs` di `HuggingFaceEmbeddings`.

## Struttura

```
pyproject.toml         # dipende da bordeus-common (workspace source)
.env.example
src/bordeus_bot/
├── __init__.py         # entry point CLI (bordeus-bot), assembla tutto
├── config.py            # configurazione da env
├── i18n.py                # messaggi multilingua: 21 messaggi statici in it/fr/en/es/de, t()/language_name()
├── geocode.py               # reverse geocoding (Nominatim/OpenStreetMap)
├── identify.py                # identificazione oggetto da foto o testo (ChatOllama)
├── rag.py                       # retrieval a due fasi (vocabolario+calendario) + risposta multilingua (ChatOllama)
└── telegram_bot.py                 # handler python-telegram-bot: /start, /comune, testo, foto, posizione
notebooks/
└── rag_eval.ipynb                    # valutazione retrieval + confronto system prompt
```

Risoluzione comune per nome, aree Sub-ATO e profilo utente non sono
qui: vivono in `../common/src/bordeus_common/db.py`, condiviso con
`../ingestion/` (che li usa per l'upsert di aree e comuni) — vedi
`../README.md` per la struttura completa del workspace.

## Note implementative

- **Bloccante ma in thread separato**: nessuna delle librerie usate
  (psycopg sync, requests, i client Ollama) ha un percorso asyncio
  nativo. Le chiamate bloccanti girano con `asyncio.to_thread(...)`
  invece che direttamente nell'event loop di python-telegram-bot — non è
  async "fino in fondo", ma evita che una richiesta lenta blocchi tutte
  le altre conversazioni in corso. Per un uso a più utenti concorrenti
  converrebbe un pool di connessioni Postgres invece di aprirne una
  nuova per richiesta (`bordeus_common.db.connect_light`, usata oggi
  per ogni messaggio).
- **Vector store cachati per area, non per comune**
  (`Service.vectorstore_for_comune`): risolve comune -> area Sub-ATO
  (query a Postgres) e costruisce il `PGVector` alla prima richiesta di
  quell'area, poi lo riusa — comuni diversi della stessa area
  condividono lo stesso vector store cachato, evitando di ricrearlo ad
  ogni messaggio. Il filtro per comune specifico si applica per singola
  domanda (`rag.comune_filter`), non alla costruzione: comuni diversi
  della stessa area condividono l'oggetto cachato ma hanno filtri
  diversi — per questo non usiamo `.as_retriever()` (che fissa i
  `search_kwargs`, incluso un eventuale filtro, alla costruzione), ma
  chiamiamo `similarity_search(..., filter=...)` direttamente.
- **Nessun seed di comuni/aree lato bot**: è la pipeline di ingestion a
  creare/aggiornare aree e comuni (`bordeus_common.db.upsert_sub_ato`/
  `upsert_comune`, chiamate da `bordeus_ingest.pipeline.run_sub_ato`,
  in un pacchetto diverso — vedi `../ingestion/README.md`) — il bot li
  legge soltanto.
- **Conferma con stato persistito, non in memoria**: `pending_comune_id`
  vive in Postgres (`users`), non in una variabile del processo bot —
  un riavvio del bot durante una conferma in sospeso non la perde (anche
  se, in pratica, l'utente dovrà comunque ritoccare il bottone se il
  messaggio Telegram originale non è più nella sessione).
