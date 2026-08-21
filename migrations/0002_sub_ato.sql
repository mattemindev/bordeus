-- Sub-ATO (Sotto-Ambito Territoriale Ottimale): area di gestione rifiuti
-- che raggruppa più comuni sotto lo stesso gestore E le stesse guide
-- operative. Introdotta osservando il caso reale della Valle d'Aosta
-- (5 Sub-ATO dal 2024): un gestore può servire più aree con contenuti
-- DIVERSI (es. Quendoz gestisce sia il Sub-ATO C sia il D, con pagine
-- guida separate — probabilmente per ecocentri/calendari specifici di
-- zona) — l'AREA è quindi la chiave giusta per raggruppare gli
-- embedding, non il nome del gestore.
--
-- Sostituisce il modello precedente, dove gli embedding erano
-- raggruppati per singolo comune (`collection_name = comune_id` in
-- langchain-postgres): con più comuni nella stessa area che condividono
-- le stesse guide, quello schema avrebbe richiesto di duplicare gli
-- embedding per ciascun comune. Ora `collection_name = sub_ato_id`.
--
-- `comuni.gestore`/`comuni.source_base_url` (dalla migration 0001) sono
-- superate da questa: quell'informazione appartiene all'area (un
-- comune non ha un gestore "suo", eredita quello della sua area), non
-- al singolo comune — vengono quindi rimosse qui.

CREATE TABLE IF NOT EXISTS sub_ato (
    id         TEXT PRIMARY KEY,
    nome       TEXT NOT NULL,          -- es. "Sub-ATO E — Mont-Rose e Walser"
    gestore    TEXT NOT NULL DEFAULT '', -- es. "TeknoService Italia" — informativo, non FK: un'area può avere più gestori in ATI (es. "ATI Quendoz/Aprica")
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE comuni
    ADD COLUMN IF NOT EXISTS sub_ato_id TEXT REFERENCES sub_ato (id) ON DELETE SET NULL;

ALTER TABLE comuni DROP COLUMN IF EXISTS gestore;
ALTER TABLE comuni DROP COLUMN IF EXISTS source_base_url;

-- Stato di onboarding "in sospeso": tra quando l'utente indica un comune
-- (posizione o testo) e quando conferma esplicitamente il Sub-ATO/gestore
-- risolto, prima di essere marcato onboarded=true. Nullo quando non c'è
-- nessuna conferma in sospeso.
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS pending_comune_id TEXT REFERENCES comuni (id) ON DELETE SET NULL;
