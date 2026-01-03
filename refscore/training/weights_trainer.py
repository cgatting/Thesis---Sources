from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

from ..core.analyzer import RefScoreAnalyzer
from ..utils.config import Config


class WeightsTrainer:
    def __init__(self, config: Optional[Config] = None, presets_dir: Optional[Path] = None) -> None:
        self.config = config or Config()
        self.presets_dir = presets_dir or (Path.home() / ".config" / "refscore" / "presets")
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def _evaluate(self, jobs: List[Tuple[str, List[str]]], weights: Dict[str, float]) -> Dict[str, Any]:
        analyzer = RefScoreAnalyzer(self.config)
        metrics = {"avg_top_score": 0.0, "overall_coverage": 0.0}
        if not jobs:
            return metrics
        top_scores: List[float] = []
        coverages: List[float] = []
        for doc_path, source_paths in jobs:
            self.config.set_scoring_weights(weights)
            analyzer = RefScoreAnalyzer(self.config)
            doc = analyzer.load_document(doc_path)
            sources = analyzer.load_sources(source_paths)
            scores = analyzer.compute_scores(doc, sources)
            top_scores.append(scores[0].refscore if scores else 0.0)
            coverage = analyzer.get_coverage_report(doc, scores)
            coverages.append(float(coverage.get("overall_coverage", 0.0)))
        metrics["avg_top_score"] = sum(top_scores) / max(1, len(top_scores))
        metrics["overall_coverage"] = sum(coverages) / max(1, len(coverages))
        return metrics

    def _grid(self, search_space: Dict[str, Tuple[float, float, float]]) -> List[Dict[str, float]]:
        keys = list(search_space.keys())
        grids: List[Dict[str, float]] = []
        def backtrack(i: int, cur: Dict[str, float]) -> None:
            if i == len(keys):
                grids.append(cur.copy())
                return
            k = keys[i]
            lo, hi, step = search_space[k]
            v = lo
            while v <= hi + 1e-9:
                cur[k] = round(v, 4)
                backtrack(i + 1, cur)
                v += step
        backtrack(0, {})
        return grids

    def train(self, jobs: List[Tuple[str, List[str]]], search_space: Dict[str, Tuple[float, float, float]], maximize: str = "avg_top_score") -> Tuple[Dict[str, float], Dict[str, Any]]:
        candidates = self._grid(search_space)
        best_weights: Dict[str, float] = self.config.get_scoring_weights()
        best_metrics: Dict[str, Any] = {"avg_top_score": 0.0, "overall_coverage": 0.0}
        best_val = float("-inf")
        for w in tqdm(candidates, desc="Training weights", unit="set"):
            merged = best_weights.copy()
            merged.update(w)
            m = self._evaluate(jobs, merged)
            val = float(m.get(maximize, 0.0))
            if val > best_val:
                best_val = val
                best_weights = merged
                best_metrics = m
        return best_weights, best_metrics

    def save_preset(self, name: str, weights: Dict[str, float], metrics: Dict[str, Any], hyperparams: Dict[str, Any]) -> str:
        ts = int(time.time())
        preset_id = f"{name}_{ts}"
        data = {
            "id": preset_id,
            "name": name,
            "timestamp": ts,
            "weights": weights,
            "metrics": metrics,
            "hyperparams": hyperparams,
        }
        out = self.presets_dir / f"{preset_id}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        index = self.presets_dir / "index.json"
        idx = []
        if index.exists():
            try:
                with open(index, "r", encoding="utf-8") as f:
                    idx = json.load(f)
            except Exception:
                idx = []
        idx = [e for e in idx if e.get("id") != preset_id]
        idx.append({"id": preset_id, "name": name, "timestamp": ts, "metrics": metrics})
        with open(index, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2, ensure_ascii=False)
        return preset_id

    def load_preset(self, preset_id_or_name: str) -> Dict[str, Any]:
        index = self.presets_dir / "index.json"
        if not index.exists():
            return {}
        with open(index, "r", encoding="utf-8") as f:
            idx = json.load(f)
        match = None
        for e in idx:
            if e.get("id") == preset_id_or_name or e.get("name") == preset_id_or_name:
                match = e
                break
        if not match:
            return {}
        preset_file = self.presets_dir / f"{match['id']}.json"
        if not preset_file.exists():
            return {}
        with open(preset_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_presets(self) -> List[Dict[str, Any]]:
        index = self.presets_dir / "index.json"
        if not index.exists():
            return []
        with open(index, "r", encoding="utf-8") as f:
            return json.load(f)

    def apply_preset(self, preset_id_or_name: str) -> bool:
        data = self.load_preset(preset_id_or_name)
        if not data:
            return False
        weights = data.get("weights", {})
        expected = set(self.config.get_scoring_weights().keys())
        if set(weights.keys()) != expected:
            return False
        for k, v in weights.items():
            if not isinstance(v, (int, float)) or v < 0.0 or v > 1.0:
                return False
        prev = self.presets_dir / "active.json"
        history: List[str] = []
        if prev.exists():
            try:
                with open(prev, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        current = data.get("id")
        if current:
            history.append(current)
        with open(prev, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        self.config.set_scoring_weights(weights)
        return True

    def rollback(self) -> Optional[str]:
        prev = self.presets_dir / "active.json"
        if not prev.exists():
            return None
        try:
            with open(prev, "r", encoding="utf-8") as f:
                history: List[str] = json.load(f)
        except Exception:
            history = []
        if not history:
            return None
        history.pop()
        with open(prev, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        last = history[-1] if history else None
        if last:
            data = self.load_preset(last)
            weights = data.get("weights", {})
            self.config.set_scoring_weights(weights)
        return last