"""
Core package initialization for RefScore academic application.

This module provides the core functionality for document analysis,
source processing, and scoring algorithms.
"""

from .analyzer import RefScoreAnalyzer
from .scoring import ScoringEngine
from .document_parser import DocumentParser
from .source_loader import SourceLoader

__all__ = [
    "RefScoreAnalyzer",
    "ScoringEngine", 
    "DocumentParser",
    "SourceLoader",
]