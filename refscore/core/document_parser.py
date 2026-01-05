"""
Document parser for RefScore academic application.

This module provides functionality to parse academic documents
in various formats (LaTeX, PDF) into structured data.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from ..models.document import Sentence, DocPack
from ..utils.config import Config
from ..utils.exceptions import ProcessingError, ValidationError


log = logging.getLogger(__name__)


class DocumentParser:
    """
    Document parser for academic documents.
    
    Supports parsing of LaTeX (.tex) and PDF documents, extracting
    sections, sentences, and metadata for analysis.
    """
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize document parser.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        log.info("Document parser initialized")
    
    def parse_document(self, document_path: str) -> DocPack:
        """
        Parse an academic document.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            DocPack containing parsed document structure
            
        Raises:
            ValidationError: If document format is unsupported
            ProcessingError: If parsing fails
        """
        path = Path(document_path)
        if not path.exists():
            raise ValidationError(f"Document file not found: {document_path}")
        
        extension = path.suffix.lower()
        
        if extension == '.tex':
            return self._parse_latex(document_path)
        elif extension == '.pdf':
            return self._parse_pdf(document_path)
        elif extension == '.docx':
            return self._parse_docx(document_path)
        else:
            raise ValidationError(f"Unsupported document format: {extension}")
    
    def _parse_latex(self, document_path: str) -> DocPack:
        """
        Parse LaTeX document.
        
        Args:
            document_path: Path to LaTeX file
            
        Returns:
            Parsed DocPack
        """
        try:
            with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Remove LaTeX comments
            content = re.sub(r'%.*', '', content)
            
            # Extract sections
            sections = self._extract_latex_sections(content)
            
            # Parse sentences from sections
            sentences, ordered_sections = self._parse_sections(content, sections)
            
            metadata = {
                "path": document_path,
                "type": "tex",
                "section_count": len(ordered_sections),
                "sentence_count": len(sentences)
            }
            
            log.info(f"Parsed LaTeX document: {len(sentences)} sentences, "
                    f"{len(ordered_sections)} sections")
            
            return DocPack(sentences, ordered_sections, metadata)
            
        except Exception as e:
            raise ProcessingError(f"Failed to parse LaTeX document: {e}")
    
    def _parse_pdf(self, document_path: str) -> DocPack:
        """
        Parse PDF document.
        
        Args:
            document_path: Path to PDF file
            
        Returns:
            Parsed DocPack
        """
        try:
            # Try to import PyMuPDF
            try:
                import fitz  # PyMuPDF
            except ImportError:
                raise ProcessingError("PDF parsing requires PyMuPDF (pip install PyMuPDF)")
            
            with fitz.open(document_path) as doc:
                page_count = len(doc)
                # Extract text from all pages
                full_text = ""
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text("text")
                        full_text += f"\n{page_text}"
                    except Exception as e:
                        log.warning(f"Failed to extract text from page {page_num}: {e}")
            
            if not full_text.strip():
                raise ProcessingError("No text could be extracted from PDF")
            
            # Extract sections and parse sentences
            sections = self._extract_pdf_sections(full_text)
            sentences, ordered_sections = self._parse_sections(full_text, sections)
            
            metadata = {
                "path": document_path,
                "type": "pdf",
                "section_count": len(ordered_sections),
                "sentence_count": len(sentences),
                "page_count": page_count
            }
            
            log.info(f"Parsed PDF document: {len(sentences)} sentences, "
                    f"{len(ordered_sections)} sections, {page_count} pages")
            
            return DocPack(sentences, ordered_sections, metadata)
            
        except Exception as e:
            raise ProcessingError(f"Failed to parse PDF document: {e}")
    
    def _extract_latex_sections(self, content: str) -> List[Tuple[int, str]]:
        """
        Extract section information from LaTeX content.
        
        Args:
            content: LaTeX document content
            
        Returns:
            List of (position, section_name) tuples
        """
        sections = []
        
        # Pattern for LaTeX sections
        section_pattern = re.compile(r'\\(section|subsection|subsubsection)\{([^}]+)\}')
        
        for match in section_pattern.finditer(content):
            position = match.start()
            section_name = match.group(2).strip()
            if section_name:
                sections.append((position, section_name))
        
        # Add end marker
        sections.append((len(content), "__END__"))
        sections.sort()
        
        return sections
    
    def _extract_pdf_sections(self, content: str) -> List[Tuple[int, str]]:
        """
        Extract section information from PDF content.
        
        Args:
            content: PDF text content
            
        Returns:
            List of (position, section_name) tuples
        """
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for section headers (numbered or all caps)
            if re.match(r'^(\d+\.|[A-Z][A-Z0-9\- ]{4,})$', line):
                position = sum(len(lines[j]) + 1 for j in range(i))
                section_name = line[:80].strip()
                if section_name:
                    sections.append((position, section_name))
        
        # Add end marker
        sections.append((len(content), "__END__"))
        sections.sort()
        
        return sections
    
    def _parse_sections(self, content: str, sections: List[Tuple[int, str]]) -> Tuple[List[Sentence], List[str]]:
        """
        Parse sentences from sectioned content.
        
        Args:
            content: Document content
            sections: List of (position, section_name) tuples
            
        Returns:
            Tuple of (sentences, ordered_sections)
        """
        sentences = []
        ordered_sections = []
        sentence_idx = 0
        
        # Sentence splitting pattern
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\\])')
        
        # Fallback: if no sections detected, treat entire document as a single section
        if not sections or len(sections) < 2:
            sections = [(0, "Document"), (len(content), "__END__")]
        else:
            # Ensure we cover any leading content before the first detected section
            first_pos, first_name = sections[0]
            if first_pos > 0:
                sections = [(0, "Document")] + sections

        for i in range(len(sections) - 1):
            start_pos, section_name = sections[i]
            end_pos = sections[i + 1][0]
            
            # Add section to ordered list if not already present
            if section_name not in ordered_sections and section_name != "__END__":
                ordered_sections.append(section_name)
            
            # Extract section content
            section_content = content[start_pos:end_pos]
            
            # Clean content based on document type
            if content.startswith('\\'):
                # LaTeX cleaning
                section_content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', section_content)
                section_content = re.sub(r'\\[a-zA-Z]+', ' ', section_content)
                section_content = re.sub(r'\{[^}]*\}', ' ', section_content)
            
            # Split into sentences
            sentence_texts = sentence_pattern.split(section_content)
            
            for sentence_text in sentence_texts:
                # Clean and normalize sentence
                sentence_text = self._clean_sentence(sentence_text)
                
                # Skip very short sentences
                min_len = 6
                try:
                    if self.config is not None:
                        proc = self.config.get_processing_config()
                        min_len = int(proc.get("min_sentence_length", min_len))
                except Exception:
                    pass
                if len(sentence_text.split()) < min_len:
                    continue
                
                sentence = Sentence(sentence_text.strip(), section_name, sentence_idx)
                sentences.append(sentence)
                sentence_idx += 1
        
        return sentences, ordered_sections
    
    def _clean_sentence(self, text: str) -> str:
        """
        Clean and normalize sentence text.
        
        Args:
            text: Raw sentence text
            
        Returns:
            Cleaned sentence text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List of file extensions
        """
        return ['.tex', '.pdf', '.docx']

    def _parse_docx(self, document_path: str) -> DocPack:
        try:
            import zipfile
            from xml.etree import ElementTree as ET
            z = zipfile.ZipFile(document_path)
            def read_xml(p):
                try:
                    with z.open(p) as f:
                        return f.read()
                except Exception:
                    return b""
            xml_main = read_xml('word/document.xml')
            xml_foot = read_xml('word/footnotes.xml')
            xml_end = read_xml('word/endnotes.xml')
            texts = []
            for data in [xml_main, xml_foot, xml_end]:
                if not data:
                    continue
                root = ET.fromstring(data)
                for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    texts.append(t.text or '')
            full_text = ' '.join(texts)
            sections = self._extract_pdf_sections(full_text)
            sentences, ordered_sections = self._parse_sections(full_text, sections)
            metadata = {
                "path": document_path,
                "type": "docx",
                "section_count": len(ordered_sections),
                "sentence_count": len(sentences)
            }
            return DocPack(sentences, ordered_sections, metadata)
        except Exception as e:
            raise ProcessingError(f"Failed to parse DOCX document: {e}")
    
    def validate_document(self, document_path: str) -> bool:
        """
        Validate if a document can be parsed.
        
        Args:
            document_path: Path to document file
            
        Returns:
            True if document is valid and parseable
        """
        try:
            path = Path(document_path)
            if not path.exists():
                return False
            
            if path.suffix.lower() not in self.get_supported_formats():
                return False
            
            # Try to parse the document
            doc_pack = self.parse_document(document_path)
            return len(doc_pack.sentences) > 0
            
        except Exception:
            return False
