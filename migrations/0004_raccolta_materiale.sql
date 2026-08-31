-- Mappatura materiale -> categoria di raccolta locale.
--
-- Il problema che risolve, con il caso reale che l'ha motivata:
-- Bard, Donnas e Hône raccolgono carta e cartone INSIEME (un'unica
-- categoria "CARTA E CARTONI"), ma le frazioni di Bard e Donnas
-- (Crous, Albard, Les Pians, Bondon) li raccolgono SEPARATI, in due
-- giorni diversi. Stesso comune, due schemi.
--
-- Prima di questa tabella il vocabolario nominava direttamente la
-- categoria di raccolta ("scatola di cartone -> Carta e Cartoni"). Ma
-- il vocabolario è condiviso dall'intera area, mentre lo schema di
-- raccolta varia per comune e per frazione: un utente di Albard si
-- sentiva indicare un bidone che nel suo caso non esiste, e lo
-- strumento del calendario doveva rifiutare la categoria e sperare che
-- il modello indovinasse al secondo tentativo quale dei due flussi
-- fosse quello giusto. Indovinare: con la sola descrizione
-- dell'oggetto, e restituendo comunque una data plausibile se sbagliava.
--
-- L'asimmetria che conta: **unire è facile, dividere no**. Se due
-- comuni chiamano lo stesso bidone in modo diverso basta un alias. Ma
-- quando un comune divide "CARTA E CARTONI" in "CARTA" e "CARTONE", il
-- nome della categoria non contiene l'informazione necessaria a
-- scegliere — serve sapere se l'oggetto è carta o cartone, e questo lo
-- sa solo il vocabolario. Da qui la separazione fra due concetti che
-- prima erano una stringa sola:
--
--   materiale  intrinseco all'oggetto, uguale ovunque: carta, cartone,
--              vetro, organico, plastica, metalli, indifferenziato.
--              È ciò che il vocabolario registra e ciò che il modello
--              passa allo strumento.
--   categoria  il flusso di raccolta locale, come lo chiama il gestore
--              in QUEL comune/frazione. È ciò che sta in raccolta_date.
--
-- Molti-a-uno per costruzione: dentro un comune un materiale finisce in
-- esattamente un bidone, quindi la mappatura è una funzione e la
-- risoluzione non è mai ambigua. Da cui la chiave primaria.
--
-- `categoria` NULL ha un significato preciso e utile: "questo materiale
-- NON è raccolto porta a porta qui" (es. il vetro solo all'ecocentro in
-- un comune). Prima era indistinguibile da un errore di battitura.
--
-- Se per una coppia (comune, frazione) non esiste NESSUNA riga qui, la
-- risoluzione ricade sull'identità (il materiale è usato come nome di
-- categoria) — cioè il comportamento precedente a questa tabella. Le
-- aree che non hanno bisogno della distinzione non devono dichiarare
-- niente.

CREATE TABLE IF NOT EXISTS raccolta_materiale (
    comune_id  TEXT NOT NULL REFERENCES comuni (id) ON DELETE CASCADE,
    hamlet     TEXT NOT NULL DEFAULT '',  -- '' = vale per l'intero comune
    materiale  TEXT NOT NULL,             -- canonico, minuscolo
    categoria  TEXT,                      -- NULL = non raccolto porta a porta qui
    source     TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (comune_id, hamlet, materiale)
);
