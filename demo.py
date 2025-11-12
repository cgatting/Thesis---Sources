"""
RefScore — Source↔Document Relevancy & Coverage
------------------------------------------------
A drop‑in Python module to score uploaded references (from .bib / Zotero JSON /
CSV / DOI list) against a LaTeX or PDF document, producing:
  • A ranked RefScore per source (Alignment, Numbers/Units, Entities, Methods, Recency, Authority)
  • Per‑section coverage stats + weakest sentences (gaps)
  • Evidence reasons per match, suitable for GUI display

Design goals
- Works even if heavy NLP libs aren't installed (graceful fallbacks)
- Minimal side effects; pure functions where possible
- Easy to integrate with existing DEEPSEARCH.py (sentence splitter, embeddings)

CLI usage
---------
python refscore_project.py \
  --doc path/to/document.tex \
  --sources refs.bib refs.json refs.csv \
  --out results_dir

Programmatic usage
------------------
from refscore_project import (ingest_sources, parse_document, compute_refscores,
                              section_coverage_report, weakest_sentences)

sources = ingest_sources(["my.bib", "zotero.json"])               # list[Source]
doc = parse_document("paper.tex")                                   # DocPack
scores = compute_refscores(doc, sources)                             # list[SourceScore]
coverage = section_coverage_report(doc, scores)                      # dict
weak = weakest_sentences(doc, scores, top_k=10)                      # list

Outputs (CLI):
  out/sources_ranked.json
  out/section_coverage.json
  out/gaps.json

Dependencies (optional, with fallbacks):
  sentence_transformers, spacy, quantulum3, bibtexparser, fitz (PyMuPDF), scikit-learn
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import json
import logging
import math
import os
import sys
import re
import time
from collections import Counter, defaultdict
from datetime import datetime

# ----------------------------- Logging -------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("RefScore")

# -------------------------- Optional Imports --------------------------------

# Embeddings
_EMBEDDER = None
_TFIDF = None
try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Loaded SentenceTransformer all-MiniLM-L6-v2")
except Exception:
    try:
        # optional TF-IDF instead
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        _TFIDF = (TfidfVectorizer(stop_words="english"), cosine_similarity)
        log.info("Falling back to TF-IDF cosine for alignment")
    except Exception:
        log.warning("No embedding or TF-IDF available; using Jaccard fallback")

# NER
_SPACY_NLP = None
try:
    import spacy  # type: ignore
    # Try small English model; if unavailable, nlp will be None and we fallback
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
        log.info("spaCy en_core_web_sm loaded for NER")
    except Exception:
        _SPACY_NLP = spacy.blank("en")
        log.info("spaCy blank model for tokenization (NER disabled)")
except Exception:
    log.info("spaCy not available; entity extraction will fallback")

# Numbers & units
_QUANT = None
try:
    from quantulum3 import parser as quant_parser  # type: ignore
    _QUANT = quant_parser
    log.info("quantulum3 loaded for number/unit parsing")
except Exception:
    log.info("quantulum3 not available; using regex number/unit detection")

# BibTeX
_BIBTEX = None
try:
    import bibtexparser  # type: ignore
    _BIBTEX = bibtexparser
    log.info("bibtexparser available")
except Exception:
    log.info("bibtexparser not available; .bib support limited to naive parse")

# PDF
_FITZ = None
try:
    import fitz  # PyMuPDF
    _FITZ = fitz
    log.info("PyMuPDF available for PDF parsing")
except Exception:
    log.info("PyMuPDF not available; PDF parsing disabled")

# HTTP (optional for Crossref metadata)
try:
    import requests
except Exception:
    requests = None  # type: ignore

# ------------------------------ GUI Imports ---------------------------------
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import customtkinter as ctk
except Exception:
    ctk = None  # type: ignore
    tk = None   # type: ignore

# ------------------------------- Data Models --------------------------------

@dataclass
class Source:
    source_id: str                  # bibkey, DOI, or filename-based id
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    authors: List[str] = dataclasses.field(default_factory=list)
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclass
class Sentence:
    text: str
    section: str
    idx: int                        # global sentence index

@dataclass
class DocPack:
    sentences: List[Sentence]
    sections: List[str]             # ordered unique sections
    meta: Dict[str, Any]            # {"path":..., "type":"tex|pdf"}

@dataclass
class RefEvidence:
    alignment: float
    entities: float
    number_unit: float
    method_metric: float
    recency: float
    authority: float
    reasons: List[str]

@dataclass
class SourceScore:
    source: Source
    refscore: float
    per_sentence: Dict[int, RefEvidence]  # sentence idx -> evidence

# --------------------------- Utility & NLP ----------------------------------

_STOPWORDS = set("""
a an the and or of to in on for with from by as is are was were be being been this that these those it its into through over under above below up down about between during before after while again further then once here there all any both each few more most other some such no nor not only own same so than too very can will just don don should now""".split())
_METHOD_TOKENS = set(
    ["svm","hmm","hmm-based","bert","roberta","xgboost","random forest","shap","lime",
     "ablation","bayesian","rl","reinforcement","regression","classification","segmentation",
     "cosine","euclidean","kmeans","dbscan","lstm","gru","transformer","attention",
     "precision","recall","f1","accuracy","auc","roc","mae","rmse","mape","cross-validation",
     "benchmark","dataset","corpus","experiment","controlled study","user study"]
)
_UNITS = [
    "%","percent","ms","s","sec","seconds","minutes","hours","hz","khz","mhz","ghz",
    "v","mv","ma","a","w","kw","db","°c","c","kelvin","k","gb","mb","kb"
]

_DEF_WORD_RE = re.compile(r"\\(textbf|emph)\{([^}]+)\}")

_DEF_PUNCT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    text = re.sub(r"[^a-z0-9%\. ]+", " ", text.lower())
    toks = [t for t in text.split() if t and t not in _STOPWORDS]
    return toks


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    return len(A & B) / float(len(A | B))


# --------------------------- Alignment (semantic) ---------------------------

def alignment_score(a: str, b: str) -> float:
    """Similarity of sentence a vs source text b in [0,1]. Tries embedder, else TFIDF, else Jaccard."""
    a = _normalize(a)
    b = _normalize(b)
    try:
        if _EMBEDDER is not None:
            va, vb = _EMBEDDER.encode([a, b], normalize_embeddings=True)
            # cosine since normalized
            return float(max(0.0, min(1.0, (va @ vb))))
    except Exception:
        pass
    try:
        if _TFIDF is not None:
            vec, cos = _TFIDF
            X = vec.fit_transform([a, b])
            s = cos(X[0], X[1])[0, 0]
            return float(max(0.0, min(1.0, s)))
    except Exception:
        pass
    # Jaccard fallback
    return _jaccard(_tokens(a), _tokens(b))


# --------------------------- Entities (NER-ish) -----------------------------

def extract_entities(text: str) -> List[str]:
    text = text.strip()
    ents: List[str] = []
    if _SPACY_NLP is not None and hasattr(_SPACY_NLP, "pipe"):
        try:
            doc = _SPACY_NLP(text)
            if hasattr(doc, "ents") and doc.ents:
                for e in doc.ents:
                    if e.label_ in {"ORG","PRODUCT","WORK_OF_ART","NORP","GPE","PERSON","EVENT"}:
                        ents.append(e.text.lower())
        except Exception:
            pass
    # fallback: crude proper-noun-ish and acronym detection
    caps = re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,})\b", text)
    acr = re.findall(r"\b([A-Z]{2,})\b", text)
    for c in caps + acr:
        ents.append(c.lower())
    # dedupe
    out: List[str] = []
    seen = set()
    for e in ents:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def entity_overlap(a: str, b: str) -> float:
    ea, eb = extract_entities(a), extract_entities(b)
    if not ea or not eb:
        return 0.0
    return _jaccard(ea, eb)


# ------------------------ Numbers & units matching --------------------------

_NUM_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)")
_UNIT_RE = re.compile(r"\b(" + "|".join([re.escape(u) for u in _UNITS]) + r")\b")


def extract_numbers_units(text: str) -> List[Tuple[float, Optional[str]]]:
    pairs: List[Tuple[float, Optional[str]]] = []
    if _QUANT is not None:
        try:
            quants = _QUANT.parse(text)
            for q in quants:
                try:
                    val = float(q.value)
                except Exception:
                    continue
                unit = None
                if getattr(q, "unit", None):
                    unit = getattr(q.unit, "name", None) or getattr(q.unit, "entity", None)
                pairs.append((val, unit.lower() if unit else None))
            if pairs:
                return pairs
        except Exception:
            pass
    # fallback regex
    nums = [float(x) for x in _NUM_RE.findall(text)]
    units = _UNIT_RE.findall(text.lower())
    # naive pairing: attach first unit if any
    unit = units[0].lower() if units else None
    for n in nums:
        pairs.append((n, unit))
    return pairs


def number_unit_match(sent: str, source: str, rel_tol: float = 0.07) -> Tuple[float, List[str]]:
    """Return (score, reasons). Score 1 if any number in sentence matches a number in source within rel_tol
    and (if available) unit matches; otherwise a soft score based on closest distance."""
    s_nums = extract_numbers_units(sent)
    r_nums = extract_numbers_units(source)
    if not s_nums or not r_nums:
        return 0.0, []
    best = 0.0
    reasons: List[str] = []
    for sv, su in s_nums:
        for rv, ru in r_nums:
            if su and ru and su != ru:
                continue
            if rv == 0:
                continue
            rel = abs(sv - rv) / max(1e-9, abs(rv))
            cand = 1.0 if rel <= rel_tol else max(0.0, 1.0 - rel)
            if cand > best:
                best = cand
                reasons = [f"numeric match {sv}≈{rv}{' ' + su if su else ''} (rel={rel:.2f})"]
    return best, reasons


# ------------------------ Methods / metrics overlap -------------------------

def method_metric_overlap(sent: str, source: str) -> float:
    a = set(_tokens(sent))
    b = set(_tokens(source))
    hits = 0
    for tok in _METHOD_TOKENS:
        parts = tok.split()
        if len(parts) == 1:
            if tok in a and tok in b:
                hits += 1
        else:
            # phrase match
            if tok in _normalize(sent) and tok in _normalize(source):
                hits += 1
    # normalize by a simple cap
    return min(1.0, hits / 6.0)


# ------------------------ Recency & authority -------------------------------

def recency_score(year: Optional[int], half_life: int = 6) -> float:
    if not year:
        return 0.0
    age = max(0, datetime.now().year - year)
    # exponential half-life decay: score=exp(-ln(2)*age/half_life)
    return math.exp(-math.log(2) * (age / max(1e-6, half_life)))


def authority_score(venue: str) -> float:
    v = venue.lower().strip()
    if not v:
        return 0.0
    # extremely lightweight heuristic; adjust as needed
    if any(k in v for k in ["nature", "science", "neurips", "icml", "acl", "emnlp", "cvpr", "iclr"]):
        return 1.0
    if any(k in v for k in ["arxiv", "workshop", "preprint"]):
        return 0.35
    # default medium
    return 0.6


# --------------------------- Scoring pipeline -------------------------------

DEFAULT_WEIGHTS = {
    "alignment": 0.45,
    "number_unit": 0.20,
    "entities": 0.15,
    "method_metric": 0.10,
    "recency": 0.07,
    "authority": 0.03,
}


def score_pair(sentence: str, source: Source, weights: Dict[str, float] = DEFAULT_WEIGHTS) -> RefEvidence:
    src_text = source.title + ". " + (source.abstract or "")
    a = alignment_score(sentence, src_text)
    e = entity_overlap(sentence, src_text)
    n, nreasons = number_unit_match(sentence, src_text)
    m = method_metric_overlap(sentence, src_text)
    r = recency_score(source.year)
    au = authority_score(source.venue)
    reasons = []
    if a >= 0.6: reasons.append(f"semantic match {a:.2f}")
    if e >= 0.3: reasons.append(f"entity overlap {e:.2f}")
    reasons += nreasons
    if m >= 0.2: reasons.append("method/metric overlap")
    if r >= 0.5: reasons.append("recent source")
    if au >= 0.8: reasons.append("high-authority venue")
    return RefEvidence(a, e, n, m, r, au, reasons)


def _weighted(e: RefEvidence, w: Dict[str, float]) -> float:
    return (w["alignment"] * e.alignment +
            w["number_unit"] * e.number_unit +
            w["entities"] * e.entities +
            w["method_metric"] * e.method_metric +
            w["recency"] * e.recency +
            w["authority"] * e.authority)


def compute_refscores(doc: DocPack, sources: List[Source], weights: Dict[str, float] = DEFAULT_WEIGHTS) -> List[SourceScore]:
    scores: List[SourceScore] = []
    for src in sources:
        per: Dict[int, RefEvidence] = {}
        for sent in doc.sentences:
            ev = score_pair(sent.text, src, weights)
            per[sent.idx] = ev
        # aggregate
        vals = [_weighted(ev, weights) for ev in per.values()]
        refscore = round(sum(vals) / max(1, len(vals)), 4)
        scores.append(SourceScore(src, refscore, per))
    # sort descending
    scores.sort(key=lambda s: s.refscore, reverse=True)
    return scores


# ----------------------------- Coverage reports -----------------------------

def section_coverage_report(doc: DocPack, source_scores: List[SourceScore]) -> Dict[str, Any]:
    # compute per-section best evidence per sentence
    # For each sentence, take the max weighted evidence across sources
    best_per_sentence: Dict[int, float] = {}
    for sc in source_scores:
        for idx, ev in sc.per_sentence.items():
            val = _weighted(ev, DEFAULT_WEIGHTS)
            if idx not in best_per_sentence or val > best_per_sentence[idx]:
                best_per_sentence[idx] = val
    section_map: Dict[str, List[float]] = defaultdict(list)
    for s in doc.sentences:
        section_map[s.section].append(best_per_sentence.get(s.idx, 0.0))
    section_stats = {
        sec: {
            "avg_support": round(sum(vals) / max(1, len(vals)), 3),
            "count": len(vals)
        }
        for sec, vals in section_map.items()
    }
    return {
        "sections": doc.sections,
        "section_stats": section_stats,
    }


def weakest_sentences(doc: DocPack, source_scores: List[SourceScore], top_k: int = 10) -> List[Dict[str, Any]]:
    best_per_sentence: Dict[int, Tuple[float, Optional[Source]]] = {}
    best_ev_store: Dict[int, Optional[RefEvidence]] = {}
    for sc in source_scores:
        for idx, ev in sc.per_sentence.items():
            val = _weighted(ev, DEFAULT_WEIGHTS)
            if (idx not in best_per_sentence) or (val > best_per_sentence[idx][0]):
                best_per_sentence[idx] = (val, sc.source)
                best_ev_store[idx] = ev
    # collect items with low support
    items: List[Tuple[float, Sentence]] = []
    for s in doc.sentences:
        val = best_per_sentence.get(s.idx, (0.0, None))[0]
        items.append((val, s))
    items.sort(key=lambda x: x[0])
    out: List[Dict[str, Any]] = []
    for val, s in items[:top_k]:
        ev = best_ev_store.get(s.idx)
        out.append({
            "idx": s.idx,
            "section": s.section,
            "support": round(val, 3),
            "sentence": s.text,
            "reasons": (ev.reasons if ev else [])
        })
    return out


# ------------------------------- Ingestion ----------------------------------

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")


def _crossref_fetch_by_doi(doi: str) -> Dict[str, Any]:
    if not requests:
        return {}
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("message", {})
    except Exception:
        pass
    return {}


def _crossref_fetch_by_title(title: str) -> Dict[str, Any]:
    if not requests:
        return {}
    url = "https://api.crossref.org/works"
    try:
        r = requests.get(url, params={"query.bibliographic": title, "rows": 1}, timeout=10)
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            return items[0] if items else {}
    except Exception:
        pass
    return {}


def _build_source_from_crossref(msg: Dict[str, Any]) -> Optional[Source]:
    if not msg:
        return None
    title = " ".join(msg.get("title", [])).strip()
    doi = msg.get("DOI", "")
    year = None
    for k in ("published-print", "published-online", "issued"):
        try:
            year = int(msg[k]["date-parts"][0][0])
            break
        except Exception:
            continue
    venue = msg.get("container-title", [""])
    venue = venue[0] if venue else ""
    authors = []
    for a in msg.get("author", []) or []:
        nm = " ".join([a.get("given", ""), a.get("family", "")]).strip()
        if nm:
            authors.append(nm)
    abstract = msg.get("abstract", "")
    # Crossref abstracts may be JATS XML; strip tags
    abstract = re.sub(r"<[^>]+>", " ", abstract or "").strip()
    sid = doi or re.sub(r"\W+", "_", title.lower())[:50]
    return Source(source_id=sid, title=title, abstract=abstract, year=year, venue=venue, doi=doi, authors=authors)


def ingest_sources(paths: List[str]) -> List[Source]:
    sources: List[Source] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".bib":
            sources.extend(_ingest_bib(p))
        elif ext in (".json",):
            sources.extend(_ingest_json(p))
        elif ext in (".csv",):
            sources.extend(_ingest_csv(p))
        else:
            # Try to read as plain text list of DOIs or titles
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = _DOI_RE.search(line)
                    if m:
                        msg = _crossref_fetch_by_doi(m.group(0))
                        s = _build_source_from_crossref(msg)
                        if s: sources.append(s)
                    else:
                        msg = _crossref_fetch_by_title(line)
                        s = _build_source_from_crossref(msg)
                        if s: sources.append(s)
    # dedupe by doi or title
    unique: Dict[str, Source] = {}
    for s in sources:
        key = s.doi or s.title.lower()
        if key and key not in unique:
            unique[key] = s
    return list(unique.values())


def _ingest_bib(path: str) -> List[Source]:
    out: List[Source] = []
    if _BIBTEX:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                db = _BIBTEX.load(f)
            for e in db.entries:
                title = e.get("title", "").strip("{} ")
                doi = e.get("doi", "")
                venue = e.get("journal", e.get("booktitle", ""))
                year = None
                try:
                    year = int(e.get("year"))
                except Exception:
                    pass
                authors = []
                if "author" in e:
                    authors = [a.strip() for a in re.split(r"\s+and\s+", e["author"]) if a.strip()]
                s = Source(source_id=e.get("ID", doi or title[:40]), title=title, abstract=e.get("abstract", ""),
                           year=year, venue=venue, doi=doi, authors=authors, extra=e)
                out.append(s)
            return out
        except Exception as ex:
            log.warning(f"bibtexparser failed ({ex}); switching to naive .bib parse")
    # naive: read titles and dois
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    entries = re.split(r"@\w+\s*\{", content)[1:]
    for ent in entries:
        title = ""
        doi = ""
        m = re.search(r"title\s*=\s*\{([^}]+)\}", ent, re.I)
        if m:
            title = m.group(1)
        m = re.search(r"doi\s*=\s*\{([^}]+)\}", ent, re.I)
        if m:
            doi = m.group(1)
        key = ent.split(",", 1)[0].strip()
        if title or doi:
            out.append(Source(source_id=key or doi or title[:40], title=title, doi=doi))
    return out


def _ingest_json(path: str) -> List[Source]:
    out: List[Source] = []
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        # Zotero export (JSON) pattern
        if isinstance(data, list):
            for item in data:
                title = item.get("title", "")
                abstract = item.get("abstractNote", "")
                year = None
                try:
                    year = int(item.get("date", "")[:4])
                except Exception:
                    pass
                venue = item.get("publicationTitle", item.get("proceedingsTitle", ""))
                doi = item.get("DOI", "")
                authors = []
                for c in item.get("creators", []) or []:
                    nm = " ".join([c.get("firstName", ""), c.get("lastName", "")]).strip()
                    if nm:
                        authors.append(nm)
                sid = item.get("key", doi or re.sub(r"\W+", "_", title.lower())[:40])
                out.append(Source(sid, title, abstract, year, venue, doi, authors, extra=item))
        else:
            # generic dict list under a key
            for item in data.get("items", []):
                title = item.get("title", "")
                if not title:
                    continue
                doi = item.get("doi", item.get("DOI", ""))
                year = None
                try:
                    year = int(item.get("year", "")[:4])
                except Exception:
                    pass
                out.append(Source(item.get("id", doi or title[:40]), title=title, abstract=item.get("abstract", ""), year=year))
    except Exception as ex:
        log.warning(f"JSON ingest failed for {path}: {ex}")
    return out


def _ingest_csv(path: str) -> List[Source]:
    out: List[Source] = []
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or row.get("Title") or "").strip()
            doi = (row.get("doi") or row.get("DOI") or "").strip()
            year = None
            try:
                y = (row.get("year") or row.get("Year") or "").strip()
                if y:
                    year = int(y[:4])
            except Exception:
                pass
            venue = (row.get("venue") or row.get("journal") or row.get("Journal") or "").strip()
            abstract = (row.get("abstract") or row.get("Abstract") or "").strip()
            sid = (row.get("id") or doi or re.sub(r"\W+", "_", title.lower())[:40])
            out.append(Source(sid, title, abstract, year, venue, doi))
    return out


# ------------------------------- Document -----------------------------------

_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\{([^}]+)\}")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\\])")


def parse_latex(path: str) -> DocPack:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    # Remove LaTeX comments
    text = re.sub(r"%.*", "", text)
    # Extract sections
    sections: List[Tuple[int, str]] = []  # (pos, name)
    for m in _SECTION_RE.finditer(text):
        sections.append((m.start(), m.group(2)))
    sections.append((len(text), "__END__"))
    sections.sort()
    # Walk sections and split into sentences
    out_sentences: List[Sentence] = []
    ordered_sections: List[str] = []
    idx = 0
    for i in range(len(sections) - 1):
        start, name = sections[i]
        end = sections[i + 1][0]
        body = text[start:end]
        sec_name = name
        if sec_name not in ordered_sections:
            ordered_sections.append(sec_name)
        # strip LaTeX commands
        body = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", body)
        body = re.sub(r"\\[a-zA-Z]+", " ", body)
        body = re.sub(r"\{[^}]*\}", " ", body)
        # split sentences (simple heuristic)
        parts = _SENT_SPLIT.split(body)
        for p in parts:
            p = _normalize(p)
            if len(p.split()) < 6:
                continue
            out_sentences.append(Sentence(p.strip(), sec_name, idx))
            idx += 1
    return DocPack(out_sentences, ordered_sections, {"path": path, "type": "tex"})


def parse_pdf(path: str) -> DocPack:
    if not _FITZ:
        raise RuntimeError("PDF parsing requires PyMuPDF (fitz)")
    doc = _FITZ.open(path)
    texts: List[str] = []
    for page in doc:
        texts.append(page.get_text("text"))
    full = "\n".join(texts)
    # naive section detection by headings (all caps lines or numbered)
    lines = [l.strip() for l in full.splitlines() if l.strip()]
    sections: List[Tuple[int, str]] = []
    buf = []
    for i, l in enumerate(lines):
        if re.match(r"^(\d+\.|[A-Z][A-Z0-9\- ]{4,})$", l):
            sections.append((len("\n".join(lines[:i])), l[:80]))
    sections.append((len(full), "__END__"))
    # Build sentences
    out_sentences: List[Sentence] = []
    ordered_sections: List[str] = []
    idx = 0
    for i in range(len(sections) - 1):
        start, name = sections[i]
        end = sections[i + 1][0]
        body = full[start:end]
        sec_name = name
        if sec_name not in ordered_sections:
            ordered_sections.append(sec_name)
        parts = _SENT_SPLIT.split(body)
        for p in parts:
            p = _normalize(p)
            if len(p.split()) < 6:
                continue
            out_sentences.append(Sentence(p.strip(), sec_name, idx))
            idx += 1
    return DocPack(out_sentences, ordered_sections, {"path": path, "type": "pdf"})


def parse_document(path: str) -> DocPack:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tex":
        return parse_latex(path)
    elif ext == ".pdf":
        return parse_pdf(path)
    else:
        raise ValueError("Unsupported document type; use .tex or .pdf")


# ------------------------------- Serialization ------------------------------

def _evidence_to_dict(ev: RefEvidence) -> Dict[str, Any]:
    return {
        "alignment": round(ev.alignment, 3),
        "entities": round(ev.entities, 3),
        "number_unit": round(ev.number_unit, 3),
        "method_metric": round(ev.method_metric, 3),
        "recency": round(ev.recency, 3),
        "authority": round(ev.authority, 3),
        "reasons": ev.reasons,
    }


def to_json_sources(scores: List[SourceScore]) -> List[Dict[str, Any]]:
    out = []
    for sc in scores:
        out.append({
            "source": {
                "id": sc.source.source_id,
                "title": sc.source.title,
                "doi": sc.source.doi,
                "year": sc.source.year,
                "venue": sc.source.venue,
                "authors": sc.source.authors,
            },
            "refscore": round(sc.refscore, 3),
            "top_reasons": _collect_top_reasons(sc, top_n=3),
        })
    return out


def _collect_top_reasons(sc: SourceScore, top_n: int = 3) -> List[str]:
    # pick the most frequent reasons across sentences
    c = Counter()
    for ev in sc.per_sentence.values():
        for r in ev.reasons:
            c[r] += 1
    return [r for r, _ in c.most_common(top_n)]


# ------------------------------- CLI Runner ---------------------------------

def run_cli(doc_path: str, source_paths: List[str], out_dir: str) -> None:
    t0 = time.time()
    log.info("Parsing document…")
    doc = parse_document(doc_path)
    log.info(f"Document has {len(doc.sentences)} sentences across {len(doc.sections)} sections")

    log.info("Ingesting sources…")
    sources = ingest_sources(source_paths)
    log.info(f"Loaded {len(sources)} sources")

    log.info("Scoring… (this may take a minute if embeddings are used)")
    scores = compute_refscores(doc, sources)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sources_ranked.json"), "w", encoding="utf-8") as f:
        json.dump(to_json_sources(scores), f, indent=2, ensure_ascii=False)

    coverage = section_coverage_report(doc, scores)
    with open(os.path.join(out_dir, "section_coverage.json"), "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    gaps = weakest_sentences(doc, scores, top_k=12)
    with open(os.path.join(out_dir, "gaps.json"), "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2, ensure_ascii=False)

    log.info(f"Done in {time.time()-t0:.1f}s. Outputs in {out_dir}")


# --------------------------- Integration helpers ----------------------------

def build_sentence_index(doc: DocPack) -> Dict[int, Sentence]:
    return {s.idx: s for s in doc.sentences}


def best_sources_for_sentence(scores: List[SourceScore], sentence_idx: int, top_k: int = 5) -> List[Tuple[Source, float, RefEvidence]]:
    rows: List[Tuple[Source, float, RefEvidence]] = []
    for sc in scores:
        ev = sc.per_sentence.get(sentence_idx)
        if ev:
            rows.append((sc.source, _weighted(ev, DEFAULT_WEIGHTS), ev))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


# --------------------------------- GUI --------------------------------------

class RefScoreApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("RefScore — Source↔Document GUI")
        self.geometry("900x700")

        # State
        self.doc_path_var = tk.StringVar()
        self.out_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "out_refscore"))
        self.sources: List[str] = []
        self.running = False

        self._build_ui()

    def _build_ui(self) -> None:
        # Top input frame
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=12, pady=12)

        # Document chooser
        ctk.CTkLabel(top, text="Document (.tex or .pdf)").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        doc_entry = ctk.CTkEntry(top, textvariable=self.doc_path_var, width=500)
        doc_entry.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ctk.CTkButton(top, text="Browse", command=self._browse_doc).grid(row=0, column=2, padx=6, pady=6)

        # Sources chooser
        ctk.CTkLabel(top, text="Sources (.bib/.json/.csv/DOI list)").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.sources_box = ctk.CTkTextbox(top, width=500, height=80)
        self.sources_box.grid(row=1, column=1, padx=6, pady=6, sticky="w")
        btns = ctk.CTkFrame(top)
        btns.grid(row=1, column=2, padx=6, pady=6, sticky="n")
        ctk.CTkButton(btns, text="Add", command=self._browse_sources).pack(fill="x", pady=4)
        ctk.CTkButton(btns, text="Clear", command=self._clear_sources).pack(fill="x")

        # Output directory
        ctk.CTkLabel(top, text="Output directory").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        out_entry = ctk.CTkEntry(top, textvariable=self.out_dir_var, width=500)
        out_entry.grid(row=2, column=1, padx=6, pady=6, sticky="w")
        ctk.CTkButton(top, text="Browse", command=self._browse_out).grid(row=2, column=2, padx=6, pady=6)

        # Run button and status
        actions = ctk.CTkFrame(self)
        actions.pack(fill="x", padx=12)
        self.run_btn = ctk.CTkButton(actions, text="Run RefScore", command=self._run_refscore)
        self.run_btn.pack(side="left", padx=6, pady=6)
        self.status_lbl = ctk.CTkLabel(actions, text="Idle")
        self.status_lbl.pack(side="left", padx=12)

        # Results tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)

        self.tab_sources = self.tabs.add("Sources")
        self.tab_coverage = self.tabs.add("Coverage")
        self.tab_gaps = self.tabs.add("Gaps")

        self.sources_out = ctk.CTkTextbox(self.tab_sources)
        self.sources_out.pack(fill="both", expand=True, padx=6, pady=6)

        self.coverage_out = ctk.CTkTextbox(self.tab_coverage)
        self.coverage_out.pack(fill="both", expand=True, padx=6, pady=6)

        self.gaps_out = ctk.CTkTextbox(self.tab_gaps)
        self.gaps_out.pack(fill="both", expand=True, padx=6, pady=6)

    def _browse_doc(self) -> None:
        path = filedialog.askopenfilename(title="Select document", filetypes=[("LaTeX/PDF", "*.tex *.pdf"), ("All", "*.*")])
        if path:
            self.doc_path_var.set(path)

    def _browse_sources(self) -> None:
        paths = filedialog.askopenfilenames(title="Select sources", filetypes=[
            ("Refs", "*.bib *.json *.csv"),
            ("All", "*.*")
        ])
        if paths:
            for p in paths:
                if p not in self.sources:
                    self.sources.append(p)
            self._refresh_sources_box()

    def _clear_sources(self) -> None:
        self.sources = []
        self._refresh_sources_box()

    def _refresh_sources_box(self) -> None:
        self.sources_box.delete("1.0", "end")
        for p in self.sources:
            self.sources_box.insert("end", p + "\n")

    def _browse_out(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.out_dir_var.set(path)

    def _run_refscore(self) -> None:
        if self.running:
            return
        doc_path = self.doc_path_var.get().strip()
        out_dir = self.out_dir_var.get().strip()
        if not doc_path:
            messagebox.showerror("Missing document", "Please select a .tex or .pdf document")
            return
        if not self.sources:
            messagebox.showerror("Missing sources", "Please add at least one source file")
            return
        self.running = True
        self.run_btn.configure(state="disabled")
        self.status_lbl.configure(text="Running…")

        import threading

        def worker():
            try:
                doc = parse_document(doc_path)
            except Exception as ex:
                self.after(0, lambda: self._finish_with_error(f"Failed to parse document: {ex}"))
                return
            try:
                sources = ingest_sources(self.sources)
            except Exception as ex:
                self.after(0, lambda: self._finish_with_error(f"Failed to ingest sources: {ex}"))
                return
            try:
                scores = compute_refscores(doc, sources)
                coverage = section_coverage_report(doc, scores)
                gaps = weakest_sentences(doc, scores, top_k=12)
            except Exception as ex:
                self.after(0, lambda: self._finish_with_error(f"Scoring failed: {ex}"))
                return
            # Write outputs
            try:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "sources_ranked.json"), "w", encoding="utf-8") as f:
                    json.dump(to_json_sources(scores), f, indent=2, ensure_ascii=False)
                with open(os.path.join(out_dir, "section_coverage.json"), "w", encoding="utf-8") as f:
                    json.dump(coverage, f, indent=2, ensure_ascii=False)
                with open(os.path.join(out_dir, "gaps.json"), "w", encoding="utf-8") as f:
                    json.dump(gaps, f, indent=2, ensure_ascii=False)
            except Exception as ex:
                log.warning(f"Failed to write outputs: {ex}")
            # Show results in UI
            self.after(0, lambda: self._display_results(scores, coverage, gaps, out_dir))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_with_error(self, msg: str) -> None:
        self.running = False
        self.run_btn.configure(state="normal")
        self.status_lbl.configure(text="Idle")
        messagebox.showerror("Error", msg)

    def _display_results(self, scores: List[SourceScore], coverage: Dict[str, Any], gaps: List[Dict[str, Any]], out_dir: str) -> None:
        self.running = False
        self.run_btn.configure(state="normal")
        self.status_lbl.configure(text=f"Done. Outputs in {out_dir}")

        # Sources
        self.sources_out.delete("1.0", "end")
        src_json = to_json_sources(scores)
        self.sources_out.insert("end", json.dumps(src_json, indent=2, ensure_ascii=False))

        # Coverage
        self.coverage_out.delete("1.0", "end")
        self.coverage_out.insert("end", json.dumps(coverage, indent=2, ensure_ascii=False))

        # Gaps
        self.gaps_out.delete("1.0", "end")
        self.gaps_out.insert("end", json.dumps(gaps, indent=2, ensure_ascii=False))


def launch_gui() -> None:
    if ctk is None:
        raise RuntimeError("customtkinter is not installed. Install it via 'pip install customtkinter'.")
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = RefScoreApp()
    app.mainloop()


# --------------------------------- Main -------------------------------------
if __name__ == "__main__":
    # If no arguments, or --gui provided, launch GUI.
    if len(sys.argv) == 1 or ("--gui" in sys.argv):
        # Strip args if --gui was used so tkinter sees clean argv
        if "--gui" in sys.argv:
            sys.argv = [sys.argv[0]]
        launch_gui()
    else:
        ap = argparse.ArgumentParser(description="RefScore — rate uploaded references against a document")
        ap.add_argument("--doc", required=True, help="Path to .tex or .pdf document")
        ap.add_argument("--sources", nargs='+', required=True, help="One or more .bib/.json/.csv/DOI-list files")
        ap.add_argument("--out", default="out_refscore", help="Output directory")
        args = ap.parse_args()
        run_cli(args.doc, args.sources, args.out)
