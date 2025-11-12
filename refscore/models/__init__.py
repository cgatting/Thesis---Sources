"""
Data models for RefScore academic application.

This package contains all data models used throughout the application,
providing type safety and data validation.
"""

from .document import Document, Sentence, DocPack
from .source import Source
from .scoring import SourceScore, RefEvidence

__all__ = [
    "Document",
    "Sentence", 
    "DocPack",
    "Source",
    "SourceScore",
    "RefEvidence",
]