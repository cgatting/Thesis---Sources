"""
RefScore Academic Application - A comprehensive BSc-level tool for academic reference analysis.

This package provides a complete solution for analyzing academic references against documents,
featuring a modern GUI, comprehensive scoring algorithms, and extensive academic features.

Author: Academic Project
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Academic Project"
__email__ = "academic@university.edu"

from .core.analyzer import RefScoreAnalyzer
from .models.document import Document, Sentence
from .models.source import Source
from .models.scoring import SourceScore, RefEvidence

__all__ = [
    "RefScoreAnalyzer",
    "Document", 
    "Sentence",
    "Source",
    "SourceScore",
    "RefEvidence",
]