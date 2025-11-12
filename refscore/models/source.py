"""
Source models for RefScore academic application.

This module defines the data structures used to represent academic sources
and references in the RefScore system.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re


@dataclass
class Source:
    """
    Represents an academic source or reference.
    
    Attributes:
        source_id: Unique identifier (bibkey, DOI, or filename-based ID)
        title: Source title
        abstract: Source abstract or summary
        year: Publication year
        venue: Publication venue (journal, conference, etc.)
        doi: Digital Object Identifier
        authors: List of author names
        extra: Additional metadata
    """
    source_id: str
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    authors: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate source data after initialization."""
        if not self.source_id or not self.source_id.strip():
            raise ValueError("Source ID cannot be empty")
        
        # Clean and validate DOI
        if self.doi:
            self.doi = self._clean_doi(self.doi)
    
    def _clean_doi(self, doi: str) -> str:
        """Clean and validate DOI format."""
        doi = doi.strip()
        # Remove common prefixes
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        return doi
    
    @property
    def is_valid(self) -> bool:
        """Check if the source has sufficient information for analysis."""
        return bool(self.title.strip() or self.abstract.strip())
    
    @property
    def author_string(self) -> str:
        """Get formatted author string."""
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) == 1:
            return self.authors[0]
        if len(self.authors) == 2:
            return f"{self.authors[0]} and {self.authors[1]}"
        return f"{self.authors[0]} et al."
    
    @property
    def citation_string(self) -> str:
        """Get formatted citation string."""
        parts = []
        
        # Authors (include placeholder when missing)
        parts.append(self.author_string)
        
        # Year
        if self.year:
            parts.append(f"({self.year})")
        
        # Title
        if self.title:
            parts.append(f'"{self.title}"')
        
        # Venue
        if self.venue:
            parts.append(f"{self.venue}")
        
        # DOI
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        
        return ". ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "authors": self.authors.copy(),
            "extra": self.extra.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Source:
        """Create Source from dictionary."""
        return cls(
            source_id=data["source_id"],
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            year=data.get("year"),
            venue=data.get("venue", ""),
            doi=data.get("doi", ""),
            authors=data.get("authors", []),
            extra=data.get("extra", {})
        )
