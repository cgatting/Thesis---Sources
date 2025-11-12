from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple

from ..models.document import Document
from ..models.scoring import SourceScore
from ..utils.config import Config


class QualityEvaluator:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        qc = self.config.get_quality_config()
        self.language = qc.get("language", "auto")
        self.document_type = qc.get("document_type", "research_paper")
        self.criteria = qc.get("criteria", {})
        self.cov_hi, self.cov_mid = self._get_thresholds(qc)
        self.strength_thr = float(self.criteria.get("strength_support_threshold", 0.65))
        self.weak_thr = float(self.criteria.get("weak_support_threshold", 0.35))
        self.min_examples = int(self.criteria.get("min_examples_per_section", 2))

    def _get_thresholds(self, qc: Dict[str, Any]) -> Tuple[float, float]:
        thr = qc.get("criteria", {}).get("coverage_thresholds", [0.6, 0.3])
        if isinstance(thr, list) and len(thr) >= 2:
            return float(thr[0]), float(thr[1])
        return 0.6, 0.3

    def assess(self, document: Document, scores: List[SourceScore]) -> Dict[str, Any]:
        from .scoring import ScoringEngine
        engine = ScoringEngine(self.config)
        coverage = engine.section_coverage_report(document, scores)
        weak_items = engine.weakest_sentences(document, scores, top_k=50)
        strengths = self._strong_sentences(document, scores, top_k=50)
        overall = float(coverage.get("overall_coverage", 0.0))
        rating = self._overall_rating(overall)
        sections = []
        section_stats = coverage.get("section_stats", {})
        for section in document.sections:
            stats = section_stats.get(section, {"avg_support": 0.0, "count": 0, "min_support": 0.0, "max_support": 0.0})
            sec_weak = [w for w in weak_items if w["section"] == section][:self.min_examples]
            sec_strong = [s for s in strengths if s["section"] == section][:self.min_examples]
            sections.append({
                "section": section,
                "avg_support": float(stats["avg_support"]),
                "rating": self._section_rating(float(stats["avg_support"])),
                "strengths": [self._format_example(e) for e in sec_strong],
                "weaknesses": [self._format_example(e) for e in sec_weak]
            })
        suggestions = self._suggestions(document, coverage, weak_items)
        narrative = self._narrative(document, overall, rating, sections, suggestions)
        return {
            "overall": {
                "coverage": round(overall, 3),
                "rating": rating
            },
            "sections": sections,
            "suggestions": suggestions,
            "text": narrative
        }

    def _strong_sentences(self, document: Document, scores: List[SourceScore], top_k: int = 50) -> List[Dict[str, Any]]:
        from .scoring import ScoringEngine
        engine = ScoringEngine(self.config)
        best_per_sentence: Dict[int, Tuple[float, Any, Any]] = {}
        for score in scores:
            for idx, evidence in score.per_sentence.items():
                w = engine._weighted_score(evidence, engine.weights)
                if idx not in best_per_sentence or w > best_per_sentence[idx][0]:
                    best_per_sentence[idx] = (w, score.source, evidence)
        items: List[Tuple[float, Any]] = []
        for sent in document.sentences:
            w = best_per_sentence.get(sent.idx, (0.0, None, None))[0]
            items.append((w, sent))
        items.sort(key=lambda x: x[0], reverse=True)
        out = []
        for w, sent in items[:top_k]:
            _, src, ev = best_per_sentence.get(sent.idx, (0.0, None, None))
            out.append({"idx": sent.idx, "section": sent.section, "support": round(w, 3), "sentence": sent.text, "reasons": ev.reasons if ev else [], "best_source": getattr(src, "source_id", None)})
        return out

    def _overall_rating(self, coverage: float) -> str:
        if coverage >= self.cov_hi:
            return "Strong"
        if coverage >= self.cov_mid:
            return "Moderate"
        return "Limited"

    def _section_rating(self, support: float) -> str:
        if support >= self.cov_hi:
            return "Strong"
        if support >= self.cov_mid:
            return "Adequate"
        return "Needs Attention"

    def _format_example(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section": item.get("section", ""),
            "support": float(item.get("support", 0.0)),
            "text": item.get("sentence", ""),
            "reason": ", ".join(item.get("reasons", []))
        }

    def _suggestions(self, document: Document, coverage: Dict[str, Any], weak_items: List[Dict[str, Any]]) -> List[str]:
        suggestions: List[str] = []
        stats = coverage.get("section_stats", {})
        for section, s in stats.items():
            if float(s.get("avg_support", 0.0)) < self.weak_thr:
                suggestions.append(f"Strengthen evidence in '{section}' with recent, high-authority sources and concrete numbers.")
        if weak_items:
            suggestions.append("Revise low-support sentences to clarify claims or add citations.")
        return suggestions

    def _narrative(self, document: Document, overall: float, rating: str, sections: List[Dict[str, Any]], suggestions: List[str]) -> str:
        lang = self.language
        if lang == "es":
            intro = f"Cobertura global {overall:.2f} — evaluación {rating}."
            lines = [intro]
            for s in sections:
                lines.append(f"{s['section']}: {s['rating']} (promedio {s['avg_support']:.2f}).")
                if s["strengths"]:
                    ex = s["strengths"][0]
                    lines.append(f"Ejemplo sólido: '{ex['text']}' (apoyo {ex['support']:.2f}).")
                if s["weaknesses"]:
                    exw = s["weaknesses"][0]
                    lines.append(f"Área a mejorar: '{exw['text']}' (apoyo {exw['support']:.2f}).")
            if suggestions:
                lines.append("Sugerencias:")
                lines.extend([f"- {s}" for s in suggestions])
            return "\n".join(lines)
        intro = f"Overall coverage {overall:.2f} — assessment {rating}."
        lines = [intro]
        for s in sections:
            lines.append(f"{s['section']}: {s['rating']} (average support {s['avg_support']:.2f}).")
            if s["strengths"]:
                ex = s["strengths"][0]
                lines.append(f"Strong example: '{ex['text']}' (support {ex['support']:.2f}).")
            if s["weaknesses"]:
                exw = s["weaknesses"][0]
                lines.append(f"Needs attention: '{exw['text']}' (support {exw['support']:.2f}).")
        if suggestions:
            lines.append("Suggestions:")
            lines.extend([f"- {s}" for s in suggestions])
        return "\n".join(lines)

