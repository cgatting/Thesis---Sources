"""
Scoring models for RefScore academic application.

This module defines the data structures used to represent scoring results
and evidence in the RefScore system.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .source import Source


@dataclass
class RefEvidence:
    """
    Represents evidence for a reference match.
    
    Attributes:
        alignment: Semantic alignment score [0, 1]
        entities: Entity overlap score [0, 1]
        number_unit: Number and unit matching score [0, 1]
        method_metric: Method and metric overlap score [0, 1]
        recency: Recency score based on publication year [0, 1]
        authority: Authority score based on venue [0, 1]
        reasons: Human-readable reasons for the scores
    """
    alignment: float
    entities: float
    number_unit: float
    method_metric: float
    recency: float
    authority: float
    reasons: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate evidence scores."""
        for field_name in ["alignment", "entities", "number_unit", "method_metric", "recency", "authority"]:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    
    def weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted score using provided weights.
        
        Args:
            weights: Dictionary of weights for each dimension
            
        Returns:
            Weighted score between 0.0 and 1.0
        """
        if weights is None:
            weights = {
                "alignment": 0.45,
                "number_unit": 0.20,
                "entities": 0.15,
                "method_metric": 0.10,
                "recency": 0.07,
                "authority": 0.03,
            }
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_sum = (
            weights["alignment"] * self.alignment +
            weights["entities"] * self.entities +
            weights["number_unit"] * self.number_unit +
            weights["method_metric"] * self.method_metric +
            weights["recency"] * self.recency +
            weights["authority"] * self.authority
        )
        
        return weighted_sum / total_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alignment": round(self.alignment, 3),
            "entities": round(self.entities, 3),
            "number_unit": round(self.number_unit, 3),
            "method_metric": round(self.method_metric, 3),
            "recency": round(self.recency, 3),
            "authority": round(self.authority, 3),
            "reasons": self.reasons.copy(),
            "weighted_score": round(self.weighted_score(), 3)
        }


@dataclass
class SourceScore:
    """
    Represents scoring results for a source.
    
    Attributes:
        source: The academic source being scored
        refscore: Overall reference score
        per_sentence: Evidence for each sentence in the document
    """
    source: Source
    refscore: float
    per_sentence: Dict[int, RefEvidence]
    
    def __post_init__(self) -> None:
        """Validate source score data."""
        if not isinstance(self.refscore, (int, float)):
            raise ValueError("refscore must be a number")
        if not (0.0 <= self.refscore <= 1.0):
            raise ValueError("refscore must be between 0.0 and 1.0")
        if not isinstance(self.per_sentence, dict):
            raise ValueError("per_sentence must be a dictionary")
    
    @property
    def evidence_count(self) -> int:
        """Get number of sentences with evidence."""
        return len(self.per_sentence)
    
    @property
    def average_evidence_score(self) -> float:
        """Get average evidence score across all sentences."""
        if not self.per_sentence:
            return 0.0
        # Average alignment (primary semantic indicator) for consistency with tests
        scores = [ev.alignment for ev in self.per_sentence.values()]
        return sum(scores) / len(scores)
    
    def get_top_reasons(self, top_n: int = 3) -> List[str]:
        """
        Get the most frequent reasons across all sentences.
        
        Args:
            top_n: Number of top reasons to return
            
        Returns:
            List of most frequent reasons
        """
        from collections import Counter
        
        reason_counter = Counter()
        for evidence in self.per_sentence.values():
            for reason in evidence.reasons:
                reason_counter[reason] += 1
        
        return [reason for reason, _ in reason_counter.most_common(top_n)]
    
    def get_sentence_scores(self) -> Dict[int, float]:
        """Get mapping of sentence indices to their weighted scores."""
        return {
            idx: evidence.weighted_score()
            for idx, evidence in self.per_sentence.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source": self.source.to_dict(),
            "refscore": round(self.refscore, 3),
            "evidence_count": self.evidence_count,
            "average_evidence_score": round(self.average_evidence_score, 3),
            "top_reasons": self.get_top_reasons(),
            "per_sentence": {
                str(idx): evidence.to_dict()
                for idx, evidence in self.per_sentence.items()
            }
        }
