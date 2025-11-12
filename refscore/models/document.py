"""
Document models for RefScore academic application.

This module defines the data structures used to represent documents
and their components in the RefScore system.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Sentence:
    """
    Represents a sentence within a document.
    
    Attributes:
        text: The sentence text content
        section: The section name where the sentence appears
        idx: Global sentence index within the document
    """
    text: str
    section: str
    idx: int
    
    def __post_init__(self) -> None:
        """Validate sentence data after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Sentence text cannot be empty")
        if self.idx < 0:
            raise ValueError("Sentence index must be non-negative")
        if not self.section or not self.section.strip():
            raise ValueError("Section name cannot be empty")


@dataclass 
class Document:
    """
    Represents a parsed academic document.
    
    Attributes:
        sentences: List of sentences in the document
        sections: Ordered list of unique section names
        meta: Metadata about the document (path, type, etc.)
    """
    sentences: List[Sentence]
    sections: List[str]
    meta: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate document data after initialization."""
        if not self.sentences:
            raise ValueError("Document must contain at least one sentence")
        if not self.sections:
            raise ValueError("Document must have at least one section")
        if not isinstance(self.meta, dict):
            raise ValueError("Document metadata must be a dictionary")
    
    def get_sentences_by_section(self, section: str) -> List[Sentence]:
        """Get all sentences from a specific section."""
        return [s for s in self.sentences if s.section == section]
    
    def get_section_stats(self) -> Dict[str, int]:
        """Get sentence count statistics per section."""
        from collections import Counter
        section_counts = Counter(s.section for s in self.sentences)
        return dict(section_counts)


@dataclass
class DocPack:
    """
    Legacy compatibility class for document representation.
    
    Maintains compatibility with existing code while providing
    the same functionality as the Document class.
    """
    sentences: List[Sentence]
    sections: List[str]
    meta: Dict[str, Any]
    
    def to_document(self) -> Document:
        """Convert to the new Document format."""
        return Document(
            sentences=self.sentences,
            sections=self.sections,
            meta=self.meta
        )
    
    @classmethod
    def from_document(cls, document: Document) -> DocPack:
        """Create DocPack from Document."""
        return cls(
            sentences=document.sentences,
            sections=document.sections,
            meta=document.meta
        )