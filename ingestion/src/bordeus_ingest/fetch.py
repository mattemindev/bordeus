"""Fetch via HTTP di pagine HTML, PDF e Markdown.

Responsabilità volutamente ristretta al solo network + parsing minimo per
la scoperta dei link: il salvataggio su disco (knowledge.py) e la vera
estrazione del contenuto (loaders.py, con i loader di LangChain) sono
altrove. Qui l'HTML viene sì fatto un parsing con BeautifulSoup, ma solo
per trovare i link a PDF/Markdown e un campione di testo per la
classificazione — l'HTML *grezzo* (non ripulito) è quello che viene
restituito e poi salvato, così il loader di LangChain lo rielabora lui
in modo consistente con come rielabora ogni altro file HTML.

`pdf_text_preview` fa lo stesso per i PDF: un'estrazione leggera (poche
pagine, con `pdfplumber` — già una dipendenza per `PDFPlumberLoader` in
loaders.py, niente da aggiungere) usata SOLO per dare un campione di
contenuto a `classify.guess_doc_type`. Prima di questa funzione i PDF
venivano classificati solo dal nome del file, che spesso è un codice
criptico (es. "FRAZ-BARD-DONNAS.pdf") da cui è impossibile indovinare la
categoria.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("bordeus_ingest")

USER_AGENT = (
    "bordeus-ingest/0.2 (+https://github.com/; proof of concept, uso non commerciale)"
)

_HEADERS = {"User-Agent": USER_AGENT}

_PDF_EXTENSIONS = (".pdf",)
_MARKDOWN_EXTENSIONS = (".md", ".markdown")


@dataclass
class Page:
    title: str
    raw_html: str
    text_sample: str  # solo per classificazione, non salvato su disco
    pdf_links: list[str] = field(default_factory=list)
    markdown_links: list[str] = field(default_factory=list)


def _categorize_links(
    soup: BeautifulSoup, base_url: str
) -> tuple[list[str], list[str]]:
    """Trova i link a PDF e Markdown nella pagina, risolti ad URL
    assoluti. Guarda solo l'estensione del path (non l'URL intera), così
    un link con querystring viene comunque riconosciuto."""
    pdf_links: list[str] = []
    markdown_links: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        path = href.split("?", 1)[0].split("#", 1)[0].lower()

        if path.endswith(_PDF_EXTENSIONS):
            bucket = pdf_links
        elif path.endswith(_MARKDOWN_EXTENSIONS):
            bucket = markdown_links
        else:
            continue

        absolute = urljoin(base_url, href)
        if absolute not in seen:
            seen.add(absolute)
            bucket.append(absolute)

    return pdf_links, markdown_links


def fetch_page(url: str, timeout: float = 20.0) -> Page:
    """Scarica una pagina HTML. Ritorna l'HTML grezzo (da salvare così
    com'è) più titolo, un campione di testo pulito (solo per la
    classificazione del tipo, non persistito) e i link a PDF/Markdown
    trovati."""
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    pdf_links, markdown_links = _categorize_links(soup, url)

    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup.select("script, style, nav, footer, noscript"):
        tag.decompose()
    text_sample = " ".join(soup.get_text(separator=" ").split())[:2000]

    return Page(
        title=title,
        raw_html=resp.text,
        text_sample=text_sample,
        pdf_links=pdf_links,
        markdown_links=markdown_links,
    )


def fetch_pdf_bytes(url: str, timeout: float = 30.0) -> bytes:
    """Scarica un PDF e ne restituisce i byte grezzi, da salvare su
    disco: l'estrazione del testo la fa il loader di LangChain (step 2),
    non più qui."""
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def pdf_text_preview(content: bytes, max_pages: int = 2, max_chars: int = 3000) -> str:
    """Estrae rapidamente il testo delle prime pagine di un PDF (con
    `pdfplumber`, limitando l'apertura alle sole pagine richieste tramite
    il parametro `pages`, invece di parsare tutto il documento), solo
    per dare un campione di contenuto reale a `classify.guess_doc_type`.
    Non usata per il contenuto RAG: quella resta responsabilità del
    caricamento con `PDFPlumberLoader` (loaders.py), che estrae l'intero
    documento.

    Silenziosa sugli errori (PDF corrotto, protetto, scansione senza
    testo, ecc.): restituisce stringa vuota invece di far fallire tutta
    la classificazione — un'anteprima persa non deve bloccare
    l'ingestion, al peggio si ricade sulla classificazione da URL."""
    try:
        with pdfplumber.open(
            BytesIO(content), pages=list(range(1, max_pages + 1))
        ) as pdf:
            parts: list[str] = []
            total_len = 0
            for page in pdf.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    logger.warning(
                        "errore durante l'estrazione del testo dalla pagina di un pdf"
                    )
                    continue
                parts.append(text)
                total_len += len(text)
                if total_len >= max_chars:
                    break
    except Exception:
        return ""

    return " ".join(" ".join(parts).split())[:max_chars]


def fetch_markdown_text(url: str, timeout: float = 20.0) -> str:
    """Scarica un file Markdown come testo grezzo. Nessuna
    normalizzazione degli spazi bianchi: la sintassi Markdown (righe
    vuote, indentazione, blocchi di codice) è significativa e va
    preservata fino allo splitter."""
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text.strip()
