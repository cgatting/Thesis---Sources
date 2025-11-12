from __future__ import annotations
import re
import json
import time
import os
from typing import Any, Dict, List, Optional
import requests

class SourceRankingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        try:
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None
        self.cache: Dict[str, Any] = {}
        self.headers = {'User-Agent': 'RefScore/1.0 (mailto:example@example.com)'}
        self.timeout = 30
        self.max_retries = 4
        self.min_interval = 1.0
        self.last_time = 0.0

    def _wait(self):
        now = time.time()
        dt = now - self.last_time
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self.last_time = time.time()

    def extract_terms(self, text: str, top_k: int = 15) -> List[str]:
        words = re.findall(r"[A-Za-z]{4,}", text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_k]]

    def build_query(self, text: str, use_refine: bool) -> str:
        terms = self.extract_terms(text)
        return " ".join(terms)

    def search(self, query: str, rows: int) -> List[Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                self._wait()
                resp = requests.get(
                    'https://api.crossref.org/works',
                    params={'query': query, 'rows': rows},
                    headers=self.headers,
                    timeout=self.timeout
                )
                if resp.status_code == 429:
                    ra = int(resp.headers.get('Retry-After', 60))
                    time.sleep(ra)
                    continue
                resp.raise_for_status()
                return resp.json().get('message', {}).get('items', [])
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    return []
                time.sleep(min(30, 2 ** attempt))
        return []

    def _embed(self, text: str):
        if self.model is not None:
            return self.model.encode(text)
        tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
        return set(tokens)

    def _similarity(self, a: Any, b: Any) -> float:
        if self.model is not None:
            import numpy as np
            v1 = a.reshape(1, -1)
            v2 = b.reshape(1, -1)
            from sklearn.metrics.pairwise import cosine_similarity as _cos
            return float(_cos(v1, v2)[0][0])
        inter = len(a & b)
        union = len(a | b) if (a or b) else 1
        return inter / union

    def rank(self, doc_text: str, rows: int = 100, threshold: float = 0.3, use_refine: bool = True) -> List[Dict[str, Any]]:
        key = json.dumps({"h": hash(doc_text), "r": rows, "t": threshold, "u": use_refine}, sort_keys=True)
        cached = self.cache.get(key)
        if cached:
            return cached
        if not doc_text.strip():
            return []
        terms = self.extract_terms(doc_text)
        query = self.build_query(doc_text, use_refine)
        items = self.search(query, rows)
        if not items:
            return []
        doc_vec = self._embed(doc_text)
        scored: List[Dict[str, Any]] = []
        for it in items:
            if not it.get('container-title') or not (it.get('title') or [''])[0]:
                continue
            title = (it.get('title') or [''])[0]
            abstract = it.get('abstract', '')
            combined = f"{title} {abstract}".strip()
            vec = self._embed(combined)
            sim = self._similarity(doc_vec, vec)
            lc = combined.lower()
            overlap = sum(1 for t in terms if t in lc)
            score = 0.85 * sim + 0.15 * (overlap / max(1, len(terms)))
            if score < threshold:
                continue
            authors = []
            for a in it.get('author', []) or []:
                fam = a.get('family', '')
                giv = a.get('given', '')
                s = fam
                if giv:
                    s += ", " + giv
                if s:
                    authors.append(s)
            year = None
            try:
                year = it.get('published-print', {}).get('date-parts', [[None]])[0][0] or it.get('published-online', {}).get('date-parts', [[None]])[0][0]
            except Exception:
                year = None
            link = it.get('URL') or (f"https://doi.org/{it.get('DOI')}" if it.get('DOI') else "")
            summary = abstract[:300] if abstract else title[:120]
            scored.append({
                "title": title,
                "authors": "; ".join(authors),
                "year": str(year or ''),
                "journal": (it.get('container-title') or [''])[0],
                "doi": it.get('DOI', ''),
                "score": float(round(score, 4)),
                "summary": summary,
                "link": link
            })
        top = sorted(scored, key=lambda x: -x['score'])[:10]
        if not top:
            fallback: List[Dict[str, Any]] = []
            for it in items:
                title = (it.get('title') or [''])[0]
                if not title:
                    continue
                abstract = it.get('abstract', '')
                combined = f"{title} {abstract}".strip()
                lc = combined.lower()
                overlap = sum(1 for t in terms if t in lc)
                base = overlap / max(1, len(terms))
                doi_bonus = 0.05 if it.get('DOI') else 0.0
                try:
                    year = it.get('published-print', {}).get('date-parts', [[None]])[0][0] or it.get('published-online', {}).get('date-parts', [[None]])[0][0]
                except Exception:
                    year = None
                recency_bonus = 0.0
                try:
                    if year:
                        recency_bonus = max(0.0, min(0.1, (int(year) - 2000) * 0.002))
                except Exception:
                    recency_bonus = 0.0
                score = float(round(base + doi_bonus + recency_bonus, 4))
                authors = []
                for a in it.get('author', []) or []:
                    fam = a.get('family', '')
                    giv = a.get('given', '')
                    s = fam
                    if giv:
                        s += ", " + giv
                    if s:
                        authors.append(s)
                link = it.get('URL') or (f"https://doi.org/{it.get('DOI')}" if it.get('DOI') else "")
                summary = abstract[:300] if abstract else title[:120]
                fallback.append({
                    "title": title,
                    "authors": "; ".join(authors),
                    "year": str(year or ''),
                    "journal": (it.get('container-title') or [''])[0],
                    "doi": it.get('DOI', ''),
                    "score": score,
                    "summary": summary,
                    "link": link
                })
            top = sorted(fallback, key=lambda x: -x['score'])[:10]
        self.cache[key] = top
        return top

