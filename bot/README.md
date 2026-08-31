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
4. **RAG (un passaggio) + tool calling per il calendario**: la
   descrizione dell'oggetto (mai la frase intera dell'utente, foto o
   testo che sia) diventa la query sul vector store dell'**area
   Sub-ATO** del comune dell'utente (`rag.py`, `answer_question`) —
   stesso vector store scritto dall'ingestion (`langchain-postgres`,
   collection isolata per area, non per comune: più comuni della stessa
   area condividono lo stesso vector store, cachato una volta per area
   — vedi `Service.vectorstore_for_comune` in `telegram_bot.py`).

   Il retrieval avviene in **un solo passaggio**, sul vocabolario/le
   guide (`rag.vocabolario_filter`): recupera come e dove smaltire
   l'oggetto (es. "bottiglia di plastica → imballaggi in plastica e
   metalli"). Il filtro è un `$or` (`rag.comune_filter`) sui metadata
   dei chunk: contenuto condiviso dell'area PIÙ quello specifico del
   comune dell'utente, mai quello di un comune vicino.

   Il **giorno di raccolta non arriva più dal retrieval**. Nella
   versione precedente c'era un secondo passaggio RAG che recuperava i
   chunk di calendario e li metteva nel prompt; quel passaggio è stato
   rimosso perché chiedeva al modello la cosa in cui è meno affidabile:
   leggere ~50 date da un elenco in testo libero e trovare la prima
   successiva a oggi. Un compito algoritmico esatto, in cui gli errori
   sono _plausibili_ — una data sbagliata di una settimana sembra una
   risposta corretta, e l'utente non ha modo di accorgersene.

   Ora le date vivono in una tabella Postgres (`raccolta_date`, vedi
   `migrations/0003_raccolta_date.sql`) e il modello le ottiene
   chiamando lo strumento `trova_prossima_raccolta`
   (`bot/src/bordeus_bot/calendario.py`): il confronto fra date lo fa
   Postgres con un `ORDER BY data ASC LIMIT 1` su un indice, e il
   modello riceve una sola data già corretta, che deve solo riportare.
   Il system prompt gli impone di usare lo strumento per qualunque
   giorno di raccolta e di non calcolarne né inventarne mai uno.

   Due proprietà dello strumento vale la pena notare:
   - **`comune_id` non è un parametro del modello.** Lo strumento espone
     solo `categoria_rifiuto`; comune e frazione arrivano dal profilo
     già confermato in onboarding e sono legati alla funzione tramite
     chiusura Python. Se fossero parametri, un'allucinazione o
     un'istruzione infilata nel messaggio dell'utente potrebbero far
     rispondere con il calendario di un comune diverso — sbagliato in un
     modo che l'utente non può riconoscere. Stesso principio per cui un
     handler HTTP prende l'identità dalla sessione e non dal body.
   - **Il parametro è un materiale, non una categoria di raccolta.** Il
     modello passa `carta`, `cartone`, `vetro`...: termini intrinseci
     all'oggetto, uguali ovunque. La traduzione nel flusso di raccolta
     locale la fa lo strumento, sui dati. Serve perché lo schema varia
     anche dentro lo stesso comune: Bard e Donnas raccolgono carta e
     cartone insieme, le loro frazioni (Crous, Albard, Les Pians,
     Bondon) separatamente. Con la categoria come parametro, quella
     scelta ricadrebbe sul modello — che ha solo la descrizione
     dell'oggetto e, sbagliando, restituirebbe comunque una data
     plausibile. L'elenco dei materiali validi viene dal database, non
     da una costante, quindi un primo tentativo sbagliato si
     autocorregge al secondo giro (massimo `rag.MAX_GIRI_TOOL`).

   Il contesto di vocabolario finisce comunque nel system prompt, che
   istruisce il modello a rispondere ESCLUSIVAMENTE in base ad esso e ai
   risultati degli strumenti, non con conoscenza generica sullo
   smaltimento rifiuti che potrebbe già avere: le regole cambiano da
   comune a comune e da gestore a gestore, una risposta plausibile ma
   generica può essere sbagliata per l'utente specifico.

   **Il modello deve supportare il tool calling.** Verificato: `gemma3`
   non lo supporta in Ollama, nemmeno nella variante 27b — `gemma4` sì.
   Con un modello che non lo supporta il bot continua a funzionare
   (`tools=[]` è un caso previsto), ma le risposte non indicano mai il
   giorno di passaggio.

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
con un modello multimodale scaricato (`ollama pull gemma4:...` o
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
Riusa il codice reale di `rag.py` (`QUERY_INSTRUCTION`,
`vocabolario_filter`, `SYSTEM_PROMPT_TEMPLATE`, `answer_question`) e lo
strumento reale di `calendario.py`, non li duplica.

**Un solo percorso di generazione, quello vero.** Anche il confronto fra
varianti di system prompt passa da `answer_question` con gli strumenti
attivi, scambiando solo il testo del prompt con il parametro
`system_prompt_template`. Una versione precedente del notebook
reimplementava la generazione a mano per poter cambiare prompt, ma
quella copia non faceva tool calling: confrontava prompt su un percorso
che il bot non usa più, e una variante poteva sembrare migliore solo
perché il giorno di raccolta non veniva mai chiesto.

Il notebook accende `TRACE` su `bordeus_bot`, così per ogni variante si
vede il prompt assemblato e con quale materiale è stato chiamato lo
strumento — che è la parte che i controlli automatici non possono
verificare.

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

```plaintext
pyproject.toml         # dipende da bordeus-common (workspace source)
.env.example
src/bordeus_bot/
├── __init__.py         # entry point CLI (bordeus-bot), assembla tutto
├── config.py            # configurazione da env
├── i18n.py                # messaggi multilingua: 21 messaggi statici in it/fr/en/es/de, t()/language_name()
├── geocode.py               # reverse geocoding (Nominatim/OpenStreetMap)
├── identify.py                # identificazione oggetto da foto o testo (ChatOllama)
├── calendario.py                # tool trova_prossima_raccolta: comune legato in chiusura, categorie dal DB
├── inline.py                      # modalità inline: @nomebot da qualunque chat
├── ui.py                            # messaggio di attesa a fasi, "sta scrivendo", messaggi lunghi
├── rag.py                         # retrieval sul vocabolario + tool calling + risposta multilingua (ChatOllama)
└── telegram_bot.py                  # handler python-telegram-bot: /start, /comune, testo, foto, posizione
notebooks/
└── rag_eval.ipynb                    # valutazione retrieval + confronto system prompt
```

Risoluzione comune per nome, aree Sub-ATO e profilo utente non sono
qui: vivono in `../common/src/bordeus_common/db.py`, condiviso con
`../ingestion/` (che li usa per l'upsert di aree e comuni) — vedi
`../README.md` per la struttura completa del workspace.

## Interfaccia

### Il messaggio di attesa racconta le fasi

Una richiesta impiega diversi secondi. Invece di un unico "un
attimo...", un solo messaggio viene aggiornato man mano:

    📖 Sto leggendo il tuo messaggio...
    🔎 Ho capito: tazzina in ceramica
       Cerco come si smaltisce a Donnas...
    → Va nel RUR. La prossima raccolta è il 04/09/2026.

La fase centrale mostra **l'oggetto riconosciuto**, che è
l'informazione più utile durante l'attesa: è anche il punto in cui il
bot sbaglia più spesso, e vedendolo l'utente può riformulare subito
invece di aspettare una risposta che sarà comunque inutile. Le fasi sono
descritte in termini di cosa succede per l'utente — nessun riferimento a
retrieval, embedding o nomi di modelli.

Lo stesso messaggio diventa poi la risposta, così la chat non accumula
messaggi di servizio, e l'indicatore "sta scrivendo" resta acceso per
tutta la durata (Telegram lo fa scadere dopo ~5 secondi, va rinnovato).
Vedi `ui.Progress`.

### Modalità inline: usare il bot in una chat con un'altra persona

`@nomebot una tazzina rotta` funziona in **qualsiasi** chat, anche dove
il bot non è stato aggiunto: la risposta viene inviata come un normale
messaggio dell'utente, e l'altra persona la legge senza dover interagire
col bot.

Una query inline non ha una chat propria, ma ha `from_user.id` — che
**coincide con il `chat_id`** della conversazione privata fra utente e
bot. Chi ha fatto l'onboarding lì si ritrova già configurato ovunque.
Chi non l'ha mai fatto riceve un unico risultato che lo indirizza alla
chat privata: la modalità inline non ha modo di condurre un onboarding
(niente bottoni, niente posizione).

Va abilitata una volta con `/setinline` su @BotFather, altrimenti
Telegram non invia mai questi update.

**Il vincolo è il tempo**: Telegram smette di attendere ben prima che
una generazione completa finisca. Le query sotto i 4 caratteri
ricevono un suggerimento senza generare nulla, e i risultati sono
cachati per utente (`is_personal=True`) — altrimenti si genererebbe una
risposta per ogni tasto premuto, su una GPU che ne regge poche in
parallelo.

### Formattazione: HTML, non Markdown

Telegram non interpreta il Markdown se non glielo si chiede, quindi
senza `parse_mode` l'utente legge `**Vetro**` con gli asterischi in
chiaro. Fra i tre formati si usa **HTML** e non MarkdownV2: MarkdownV2
obbliga a fare l'escape di una quindicina di caratteri (`.`, `-`, `!`,
`(`, `)`…) che compaiono di continuo in un testo normale, e un solo
carattere dimenticato fa fallire l'invio dell'intera risposta. In HTML
i caratteri da proteggere sono tre.

`ui.to_html` fa l'escape **prima** della conversione, quindi un `<`
scritto dal modello diventa testo e non un tag: l'output del modello non
può iniettare markup. Converte `**grassetto**` e `` `codice` ``, e
lascia stare `_corsivo_` di proposito — gli underscore compaiono negli
URL e negli id (`les_pians`, `sub-ato-e`), e trasformarli in corsivo
spezzerebbe proprio il testo che deve restare copiabile.

L'invio ricade sul testo semplice se Telegram rifiuta il markup
(`ui._invia`). Non è pignoleria: il testo viene da un modello, e fra
"risposta senza grassetto" e "nessuna risposta" la scelta è ovvia — ma
va scritta, altrimenti il caso raro diventa un messaggio perso in
silenzio.

Le anteprime dei link sono disattivate: la fonte è spesso un PDF di
qualche megabyte, e Telegram allegherebbe una scheda di download sotto
ogni risposta, più ingombrante della risposta stessa.

### La fonte in fondo alla risposta

Le risposte citano la pubblicazione del gestore da cui viene il
contenuto usato:

    Va nel RUR (Sacchi Grigi). La prossima raccolta è il 04/09/2026.

    Fonte:
    • Riciclabolario TeknoService Italia     ← link cliccabile

Serve perché la risposta riguarda una regola comunale che cambia da
gestore a gestore: senza provenienza è un'asserzione da credere sulla
fiducia. Conta soprattutto **quando il bot sbaglia**, perché lascia
all'utente un modo di accorgersene.

Il piè di pagina è composto **in Python** (`rag.Risposta.fonti`,
`telegram_bot._con_fonti`), dai metadata dei chunk recuperati e dagli
strumenti effettivamente invocati — mai chiesto al modello. Un URL
generato token per token è un URL che prima o poi viene inventato, e una
fonte sbagliata è peggio di nessuna fonte: sposta la fiducia su
qualcosa che non si può verificare.

Una fonte per riga, con l'URL agganciato al nome invece che stampato per
esteso: un link a un PDF del gestore occupa due righe e rende
illeggibile la coda del messaggio, mentre il nome della pubblicazione è
l'informazione che serve.

La provenienza si dichiara una volta per pubblicazione in `[[fonti]]`
nel manifest dell'area (vedi `ingestion/README.md`), e viaggia nei
metadata dei chunk e nelle colonne `fonte_nome`/`fonte_url` di
`raccolta_date`. Un'area che non ne dichiara funziona come prima: le
risposte semplicemente non citano nulla.

### Il bottone "Non è questo"

Sotto ogni risposta. L'identificazione dell'oggetto è il punto in cui il
bot sbaglia più spesso, ed è anche quello in cui l'utente se ne accorge
subito, perché il messaggio di attesa gli ha appena mostrato cosa aveva
capito.

Al tap, il messaggio successivo dell'utente viene usato **come
descrizione dell'oggetto così com'è**, saltando del tutto
l'identificazione: è il punto del bottone, altrimenti il secondo
tentativo potrebbe sbagliare esattamente come il primo. Il bottone viene
tolto dal messaggio dopo l'uso.

Lo stato sta in `user_data` (per utente, non per chat): in un gruppo più
persone possono avere una correzione in sospeso contemporaneamente, e
legarla alla chat farebbe scambiare le correzioni fra membri diversi. È
in memoria, non su Postgres — se il bot riparte, al massimo un utente
riscrive la frase.

### Richieste in parallelo

`MAX_RICHIESTE_PARALLELE` (default 2) limita quante richieste vengono
servite insieme. Il modello gira in locale su una sola GPU: oltre un
certo numero le richieste non vanno più veloci, si contendono la stessa
VRAM e rallentano tutte. L'attesa in coda avviene mentre l'utente vede
già "ho capito: X", quindi si legge come elaborazione in corso, non come
bot bloccato.

## Log: il livello TRACE

`logging` si ferma a DEBUG. I prompt di sistema completi sono però una
categoria a sé: indispensabili per capire perché il modello ha risposto
in un certo modo, ma migliaia di caratteri ripetuti a ogni messaggio.
A DEBUG renderebbero DEBUG inutilizzabile per tutto il resto — chi lo
attiva per seguire una query o l'onboarding si troverebbe il terminale
pieno di prompt.

`TRACE` (5, definito in `bordeus_common.log`) è quindi il livello del
"cosa ha visto esattamente il modello":

- il **system prompt assemblato**, con lingua, contesto e strumenti già
  interpolati — non ricostruibile dagli altri log, perché dipende da
  tutti e tre insieme;
- i **chunk recuperati**, con fonte, tipo e comune di provenienza, e il
  filtro usato;
- la **descrizione degli strumenti** offerti al modello;
- gli **argomenti** con cui il modello ha chiamato lo strumento, la
  risoluzione materiale → categoria, e il **risultato** restituito;
- i prompt di `identify.py` e la descrizione dell'oggetto estratta;
- la **risposta finale** grezza.

```bash
LOG_LEVEL=TRACE uv run bordeus-bot
```

⚠️ **A TRACE finisce nei log anche il contenuto dei messaggi degli
utenti.** È per il debug locale e per i notebook, non per un'istanza che
serve persone vere; il bot stampa un warning all'avvio se lo trova
attivo.

Nei notebook conviene accenderlo su un solo logger, per non tirarsi
dietro il rumore delle librerie:

```python
from bordeus_common.log import setup_logging, set_level
setup_logging("INFO")
set_level("bordeus_bot", "TRACE")
```

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
- **Il tool del calendario NON è cachato, il vector store sì**
  (`Service.tools_for_comune` vs `Service.vectorstore_for_comune`). La
  differenza è deliberata: il vector store è per area e il filtro per
  comune si applica alla singola query, quindi si può condividere fra
  utenti; il tool ha invece il comune dell'utente legato nella chiusura,
  e riusarlo fra conversazioni significherebbe rispondere a un utente
  con il calendario di un altro. Viene costruito per ogni richiesta
  (una query a Postgres per le categorie disponibili, dentro
  `asyncio.to_thread` come tutto il resto del percorso bloccante).
- **Frazioni: schema pronto, onboarding no.** `raccolta_date` distingue
  già i calendari per frazione e `bordeus_common.calendario` ricade da
  solo sul calendario del comune quando la frazione non ha un override.
  `users` però non ha ancora una colonna per la frazione, quindi
  `tools_for_comune` passa `hamlet=""` e tutti gli utenti di un comune
  vedono il calendario comunale. Il giorno in cui l'onboarding
  risolverà la frazione (via reverse geocoding OSM, che restituisce
  l'hamlet nella stessa risposta Nominatim già usata per il comune) non
  c'è altro da cambiare lato retrieval.
- **Nessun seed di comuni/aree lato bot**: è la pipeline di ingestion a
  creare/aggiornare aree e comuni (`bordeus_common.db.upsert_sub_ato`/
  `upsert_comune`, chiamate da `bordeus_ingest.pipeline.registra_anagrafica`,
  in un pacchetto diverso — vedi `../ingestion/README.md`) — il bot li
  legge soltanto.
