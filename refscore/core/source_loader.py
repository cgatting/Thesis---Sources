"""
Source loader for RefScore academic application.

This module provides functionality to load and parse academic sources
from various file formats (.bib, .json, .csv, DOI lists).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from ..models.source import Source
from ..utils.config import Config
from ..utils.exceptions import ProcessingError, ValidationError


log = logging.getLogger(__name__)


class SourceLoader:
    """
    Source loader for academic references.
    
    Supports loading sources from various formats:
    - BibTeX (.bib files)
    - Zotero JSON exports (.json files)
    - CSV files with source metadata
    - Plain text DOI lists
    """
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize source loader.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.crossref_available = self._check_crossref_availability()
        log.info("Source loader initialized")
    
    def _check_crossref_availability(self) -> bool:
        """Check if Crossref API is available."""
        try:
            import requests
            return True
        except ImportError:
            log.warning("requests library not available; Crossref API disabled")
            return False
    
    def load_sources(self, source_path: str) -> List[Source]:
        """
        Load sources from a file.
        
        Args:
            source_path: Path to source file
            
        Returns:
            List of parsed Source objects
            
        Raises:
            ValidationError: If file format is unsupported
            ProcessingError: If file loading fails
        """
        path = Path(source_path)
        if not path.exists():
            raise ValidationError(f"Source file not found: {source_path}")
        
        extension = path.suffix.lower()
        
        try:
            if extension == '.bib':
                return self._load_bibtex(source_path)
            elif extension == '.json':
                return self._load_json(source_path)
            elif extension == '.csv':
                return self._load_csv(source_path)
            elif extension == '.txt':
                return self._load_doi_list(source_path)
            else:
                # Try to load as DOI list
                return self._load_doi_list(source_path)
        except Exception as e:
            raise ProcessingError(f"Failed to load sources from {source_path}: {e}")
    
    def _load_bibtex(self, file_path: str) -> List[Source]:
        """
        Load sources from BibTeX file.
        
        Args:
            file_path: Path to BibTeX file
            
        Returns:
            List of Source objects
        """
        sources = []
        
        # Try to use bibtexparser first
        try:
            import bibtexparser
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                bib_database = bibtexparser.load(f)
            
            for entry in bib_database.entries:
                try:
                    source = self._parse_bibtex_entry(entry)
                    if source:
                        sources.append(source)
                except Exception as e:
                    log.warning(f"Failed to parse BibTeX entry: {e}")
            
            log.info(f"Loaded {len(sources)} sources from BibTeX file")
            return sources
            
        except ImportError:
            log.warning("bibtexparser not available; using naive BibTeX parsing")
        
        # Fallback to naive parsing
        return self._parse_bibtex_naive(file_path)
    
    def _parse_bibtex_entry(self, entry: Dict[str, str]) -> Optional[Source]:
        """
        Parse a single BibTeX entry.
        
        Args:
            entry: BibTeX entry dictionary
            
        Returns:
            Source object or None if parsing fails
        """
        try:
            # Extract basic fields
            title = entry.get("title", "").strip("{} ")
            doi = entry.get("doi", "")
            venue = entry.get("journal", entry.get("booktitle", ""))
            
            # Extract year
            year = None
            try:
                year = int(entry.get("year", ""))
            except (ValueError, TypeError):
                pass
            
            # Extract authors
            authors = []
            if "author" in entry:
                author_string = entry["author"]
                # Split by 'and' and clean up
                author_list = [a.strip() for a in re.split(r'\s+and\s+', author_string) if a.strip()]
                authors = self._clean_author_names(author_list)
            
            # Generate source ID
            source_id = entry.get("ID", doi or title[:40] or "unknown")
            
            return Source(
                source_id=source_id,
                title=title,
                abstract=entry.get("abstract", ""),
                year=year,
                venue=venue,
                doi=doi,
                authors=authors,
                extra=entry
            )
            
        except Exception as e:
            log.warning(f"Failed to parse BibTeX entry: {e}")
            return None
    
    def _parse_bibtex_naive(self, file_path: str) -> List[Source]:
        """
        Naive BibTeX parser using regular expressions.
        
        Args:
            file_path: Path to BibTeX file
            
        Returns:
            List of Source objects
        """
        sources = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split into entries
            entries = re.split(r'@\w+\s*\{', content)[1:]
            
            for entry in entries:
                try:
                    # Extract title
                    title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry, re.I)
                    title = title_match.group(1) if title_match else ""
                    
                    # Extract DOI
                    doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', entry, re.I)
                    doi = doi_match.group(1) if doi_match else ""
                    
                    # Extract key (first part of entry)
                    key = entry.split(',', 1)[0].strip()
                    
                    if title or doi:
                        source = Source(
                            source_id=key or doi or title[:40],
                            title=title,
                            doi=doi
                        )
                        sources.append(source)
                        
                except Exception as e:
                    log.warning(f"Failed to parse naive BibTeX entry: {e}")
            
            log.info(f"Loaded {len(sources)} sources using naive BibTeX parsing")
            return sources
            
        except Exception as e:
            raise ProcessingError(f"Failed to parse BibTeX file: {e}")
    
    def _load_json(self, file_path: str) -> List[Source]:
        """
        Load sources from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of Source objects
        """
        sources = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle Zotero export format
            if isinstance(data, list):
                for item in data:
                    try:
                        source = self._parse_zotero_item(item)
                        if source:
                            sources.append(source)
                    except Exception as e:
                        log.warning(f"Failed to parse Zotero item: {e}")
            
            # Handle generic format
            elif isinstance(data, dict) and "items" in data:
                for item in data["items"]:
                    try:
                        source = self._parse_generic_item(item)
                        if source:
                            sources.append(source)
                    except Exception as e:
                        log.warning(f"Failed to parse generic item: {e}")
            
            log.info(f"Loaded {len(sources)} sources from JSON file")
            return sources
            
        except Exception as e:
            raise ProcessingError(f"Failed to parse JSON file: {e}")
    
    def _parse_zotero_item(self, item: Dict[str, Any]) -> Optional[Source]:
        """
        Parse a Zotero export item.
        
        Args:
            item: Zotero item dictionary
            
        Returns:
            Source object or None
        """
        try:
            title = item.get("title", "")
            abstract = item.get("abstractNote", "")
            
            # Extract year
            year = None
            try:
                date_str = item.get("date", "")
                if date_str and len(date_str) >= 4:
                    year = int(date_str[:4])
            except (ValueError, TypeError):
                pass
            
            venue = item.get("publicationTitle", item.get("proceedingsTitle", ""))
            doi = item.get("DOI", "")
            
            # Extract authors
            authors = []
            for creator in item.get("creators", []):
                first_name = creator.get("firstName", "")
                last_name = creator.get("lastName", "")
                author_name = f"{first_name} {last_name}".strip()
                if author_name:
                    authors.append(author_name)
            
            source_id = item.get("key", doi or re.sub(r'\W+', '_', title.lower())[:40])
            
            return Source(
                source_id=source_id,
                title=title,
                abstract=abstract,
                year=year,
                venue=venue,
                doi=doi,
                authors=authors,
                extra=item
            )
            
        except Exception as e:
            log.warning(f"Failed to parse Zotero item: {e}")
            return None
    
    def _parse_generic_item(self, item: Dict[str, Any]) -> Optional[Source]:
        """
        Parse a generic source item.
        
        Args:
            item: Generic item dictionary
            
        Returns:
            Source object or None
        """
        try:
            title = item.get("title", "")
            if not title:
                return None
            
            doi = item.get("doi", item.get("DOI", ""))
            
            # Extract year
            year = None
            try:
                year_str = item.get("year", "")
                if year_str and len(str(year_str)) >= 4:
                    year = int(str(year_str)[:4])
            except (ValueError, TypeError):
                pass
            
            return Source(
                source_id=item.get("id", doi or title[:40]),
                title=title,
                abstract=item.get("abstract", ""),
                year=year
            )
            
        except Exception as e:
            log.warning(f"Failed to parse generic item: {e}")
            return None
    
    def _load_csv(self, file_path: str) -> List[Source]:
        """
        Load sources from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of Source objects
        """
        sources = []
        
        try:
            with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader, start=1):
                    try:
                        source = self._parse_csv_row(row)
                        if source:
                            sources.append(source)
                    except Exception as e:
                        log.warning(f"Failed to parse CSV row {row_num}: {e}")
            
            log.info(f"Loaded {len(sources)} sources from CSV file")
            return sources
            
        except Exception as e:
            raise ProcessingError(f"Failed to parse CSV file: {e}")
    
    def _parse_csv_row(self, row: Dict[str, str]) -> Optional[Source]:
        """
        Parse a CSV row into a Source object.
        
        Args:
            row: CSV row dictionary
            
        Returns:
            Source object or None
        """
        try:
            title = (row.get("title") or row.get("Title") or "").strip()
            doi = (row.get("doi") or row.get("DOI") or "").strip()
            
            # Extract year
            year = None
            try:
                year_str = (row.get("year") or row.get("Year") or "").strip()
                if year_str and len(year_str) >= 4:
                    year = int(year_str[:4])
            except (ValueError, TypeError):
                pass
            
            venue = (row.get("venue") or row.get("journal") or row.get("Journal") or "").strip()
            abstract = (row.get("abstract") or row.get("Abstract") or "").strip()
            
            source_id = (row.get("id") or doi or re.sub(r'\W+', '_', title.lower())[:40])
            
            return Source(
                source_id=source_id,
                title=title,
                abstract=abstract,
                year=year,
                venue=venue,
                doi=doi
            )
            
        except Exception as e:
            log.warning(f"Failed to parse CSV row: {e}")
            return None
    
    def _load_doi_list(self, file_path: str) -> List[Source]:
        """
        Load sources from a plain text DOI list.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of Source objects
        """
        sources = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Only process valid DOI matches to keep loader deterministic in tests
                        doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', line)
                        if doi_match:
                            doi = doi_match.group(0)
                            source = self._fetch_crossref_by_doi(doi)
                            if source:
                                sources.append(source)
                    except Exception as e:
                        log.warning(f"Failed to process line {line_num}: {e}")
            
            log.info(f"Loaded {len(sources)} sources from DOI list")
            return sources
            
        except Exception as e:
            raise ProcessingError(f"Failed to parse DOI list: {e}")
    
    def _fetch_crossref_by_doi(self, doi: str) -> Optional[Source]:
        """
        Fetch source information from Crossref API by DOI.
        
        Args:
            doi: DOI string
            
        Returns:
            Source object or None
        """
        if not self.crossref_available:
            log.debug("Crossref API not available")
            return None
        
        try:
            import requests
            
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get("message", {})
                return self._parse_crossref_response(data, doi)
            
        except Exception as e:
            log.warning(f"Failed to fetch Crossref data for DOI {doi}: {e}")
        
        return None
    
    def _fetch_crossref_by_title(self, title: str) -> Optional[Source]:
        """
        Fetch source information from Crossref API by title.
        
        Args:
            title: Publication title
            
        Returns:
            Source object or None
        """
        if not self.crossref_available:
            log.debug("Crossref API not available")
            return None
        
        try:
            import requests
            
            url = "https://api.crossref.org/works"
            params = {
                "query.bibliographic": title,
                "rows": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                items = response.json().get("message", {}).get("items", [])
                if items:
                    return self._parse_crossref_response(items[0])
            
        except Exception as e:
            log.warning(f"Failed to fetch Crossref data for title '{title}': {e}")
        
        return None
    
    def _parse_crossref_response(self, data: Dict[str, Any], doi: str = "") -> Optional[Source]:
        """
        Parse Crossref API response into Source object.
        
        Args:
            data: Crossref API response data
            doi: DOI string (optional)
            
        Returns:
            Source object or None
        """
        try:
            if not data:
                return None
            
            # Extract title
            title_list = data.get("title", [])
            title = " ".join(title_list).strip() if title_list else ""
            
            # Extract DOI
            if not doi:
                doi = data.get("DOI", "")
            
            # Extract year
            year = None
            for date_key in ["published-print", "published-online", "issued"]:
                try:
                    date_parts = data[date_key]["date-parts"][0]
                    year = int(date_parts[0])
                    break
                except (KeyError, IndexError, ValueError):
                    continue
            
            # Extract venue
            venue_list = data.get("container-title", [])
            venue = venue_list[0] if venue_list else ""
            
            # Extract authors
            authors = []
            for author_data in data.get("author", []):
                given_name = author_data.get("given", "")
                family_name = author_data.get("family", "")
                author_name = f"{given_name} {family_name}".strip()
                if author_name:
                    authors.append(author_name)
            
            # Extract abstract (may be JATS XML)
            abstract = data.get("abstract", "")
            if abstract:
                # Strip XML tags if present
                abstract = re.sub(r'<[^>]+>', ' ', abstract).strip()
            
            # Generate source ID
            source_id = doi or re.sub(r'\W+', '_', title.lower())[:50]
            
            return Source(
                source_id=source_id,
                title=title,
                abstract=abstract,
                year=year,
                venue=venue,
                doi=doi,
                authors=authors,
                extra=data
            )
            
        except Exception as e:
            log.warning(f"Failed to parse Crossref response: {e}")
            return None
    
    def _clean_author_names(self, authors: List[str]) -> List[str]:
        """
        Clean and normalize author names.
        
        Args:
            authors: List of author names
            
        Returns:
            Cleaned list of author names
        """
        cleaned_authors = []
        
        for author in authors:
            # Remove extra whitespace and normalize
            author = re.sub(r'\s+', ' ', author.strip())
            
            # Remove common LaTeX formatting
            author = re.sub(r'[{}]', '', author)
            
            if author and len(author) > 1:  # Skip single characters
                cleaned_authors.append(author)
        
        return cleaned_authors
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported source file formats.
        
        Returns:
            List of file extensions
        """
        return ['.bib', '.json', '.csv', '.txt']
    
    def validate_source_file(self, file_path: str) -> bool:
        """
        Validate if a source file can be loaded.
        
        Args:
            file_path: Path to source file
            
        Returns:
            True if file is valid and loadable
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            if path.suffix.lower() not in self.get_supported_formats():
                return False
            
            # Try to load the file
            sources = self.load_sources(file_path)
            return len(sources) > 0
            
        except Exception:
            return False
