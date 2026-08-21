"""Euristica per classificare un documento in una categoria (calendario,
guide, moduli, servizi, altro) a partire da URL, titolo e un campione di
contenuto testuale.

Categorie in italiano, coerenti con il resto del progetto (i documenti
stessi — regolamenti comunali, calendari di raccolta — sono in italiano).
È un'euristica a parole chiave volutamente semplice per un PoC: non
richiede training né dipendenze aggiuntive, ed è facile da estendere
aggiungendo voci a _KEYWORDS.
"""

from __future__ import annotations

# Ordine = priorità di match: la prima categoria le cui parole chiave
# compaiono nel testo vince. "calendario" prima di "guide" perché un
# calendario di raccolta a volte si autodefinisce anche "guida alla
# raccolta", ma è più utile categorizzarlo per la sua natura pratica
# (date/orari) che come guida generica.
_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "calendario",
        ["calendario", "porta a porta", "giorni di raccolta", "orari di conferimento"],
    ),
    ("guide", ["regolamento", "guida", "linee guida", "vademecum", "faq"]),
    (
        "moduli",
        ["modulistica", "modulo", "richiesta", "domanda di", "autocertificazione"],
    ),
    (
        "servizi",
        [
            "servizi",
            "servizio",
            "ecocentro",
            "isola ecologica",
            "ingombranti",
            "compostaggio",
        ],
    ),
]

DEFAULT_CATEGORY = "altro"

# Categorie note, utile altrove (es. per iterare tutte le sottocartelle
# attese di knowledge/<area_id>/).
CATEGORIES = [c for c, _ in _KEYWORDS] + [DEFAULT_CATEGORY]


def guess_doc_type(url: str = "", title: str = "", text_sample: str = "") -> str:
    """Restituisce la categoria stimata. Il campione di testo è opzionale
    (parametro con default vuoto) ma va sempre passato quando disponibile:
    classificare solo dal nome del file (specialmente i PDF, spesso
    nominati con codici criptici come "FRAZ-BARD-DONNAS.pdf") è
    inaffidabile — vedi fetch.pdf_text_preview, usata proprio per questo.
    Funziona comunque solo con url/title se text_sample è vuoto/assente,
    semplicemente con meno segnale a disposizione."""
    haystack = " ".join([url, title, text_sample[:2000]]).lower()
    for category, keywords in _KEYWORDS:
        if any(kw in haystack for kw in keywords):
            return category
    return DEFAULT_CATEGORY
