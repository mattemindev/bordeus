-- Date di raccolta porta a porta: tabella relazionale dedicata, NON
-- vector store.
--
-- Motivo: "qual è la prossima raccolta dell'organico dopo oggi?" è un
-- lookup esatto su un insieme ordinato di date, non una ricerca per
-- similarità semantica. Nel vector store un semestre di calendario
-- finisce in uno o due chunk da ~50 date in testo libero, e il modello
-- deve fare aritmetica sulle date leggendole dal prompt — inaffidabile
-- (errori di uno, date già passate scelte per sbaglio). Qui la stessa
-- domanda diventa un ORDER BY data ASC LIMIT 1 su un indice, e il
-- modello riceve una sola data già corretta tramite tool calling (vedi
-- bordeus_common.calendario, bot/calendario.py).
--
-- `hamlet` = frazione, con la convenzione della stringa vuota per
-- "vale per l'intero comune" — la stessa già usata per `comune_id` nei
-- metadata dei chunk (vuoto = condiviso dall'area). Preferita a NULL
-- perché entra in un indice UNIQUE senza il trattamento speciale che
-- NULL richiede in SQL (NULL != NULL, quindi ON CONFLICT non
-- riconoscerebbe due righe "senza frazione" come duplicate) e perché
-- evita `IS NOT DISTINCT FROM` sparso nelle query.
--
-- Un calendario vale tipicamente per PIÙ comuni (caso reale: Bard,
-- Donnas e Hône condividono lo stesso file), e una frazione può avere
-- un calendario diverso da quello del proprio comune per motivi
-- logistici (caso reale: Albard, Bondon e Les Pians per Donnas, Crous
-- per Bard). Il legame file -> (comuni, frazioni) è dichiarato nel
-- manifest `area.toml` dell'area, non dedotto dalla posizione del file
-- su disco: dedurlo obbligherebbe a tenere 14 copie dello stesso
-- contenuto per coprire 4 calendari distinti.
--
-- `source` è il nome del file Markdown da cui la riga proviene: serve
-- alla ri-ingestione idempotente (si cancellano le righe di QUEL file
-- per QUELLA coppia comune/frazione e si reinseriscono), così correggere
-- un calendario ed eseguire di nuovo l'ingestion non lascia righe
-- orfane di una versione precedente.
CREATE TABLE IF NOT EXISTS raccolta_date (
    id BIGSERIAL PRIMARY KEY,
    comune_id TEXT NOT NULL REFERENCES comuni (id) ON DELETE CASCADE,
    hamlet TEXT NOT NULL DEFAULT '', -- '' = vale per l'intero comune
    categoria TEXT NOT NULL, -- normalizzata in MAIUSCOLO all'ingestion
    colore TEXT NOT NULL DEFAULT '', -- es. "Bidone Marrone" — informativo
    data DATE NOT NULL,
    source TEXT NOT NULL DEFAULT '', -- file Markdown di provenienza
    fonte_nome TEXT NOT NULL DEFAULT '',
    fonte_url TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now ()
);

-- Copre la query del tool (comune + frazione + categoria, prossima data
-- dopo oggi): tutte le colonne del WHERE più `data` come ultima chiave,
-- così Postgres può risolvere ORDER BY data ASC LIMIT 1 scorrendo
-- l'indice invece di ordinare un set intermedio.
CREATE INDEX IF NOT EXISTS idx_raccolta_lookup ON raccolta_date (comune_id, hamlet, categoria, data);

-- Stessa data, stessa categoria, stesso comune/frazione = una sola riga,
-- anche se due file diversi la dichiarano (es. semestri che si
-- sovrappongono di qualche giorno ai bordi).
CREATE UNIQUE INDEX IF NOT EXISTS idx_raccolta_unico ON raccolta_date (comune_id, hamlet, categoria, data);

-- Elenco delle frazioni note per ogni comune. Non strettamente
-- necessaria per il tool (che sa già ricadere sul calendario del comune
-- quando una frazione non ha override), ma serve a chi cura i dati per
-- sapere quali frazioni esistono senza doverle dedurre da
-- raccolta_date, e sarà il punto di aggancio naturale se in futuro
-- l'onboarding risolverà la frazione dell'utente via reverse geocoding
-- OSM (oggi non lo fa: `users` non ha una colonna hamlet).
CREATE TABLE IF NOT EXISTS frazioni (
    comune_id TEXT NOT NULL REFERENCES comuni (id) ON DELETE CASCADE,
    hamlet TEXT NOT NULL,
    nome TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now (),
    PRIMARY KEY (comune_id, hamlet)
);