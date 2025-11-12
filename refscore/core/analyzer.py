"""
Main analyzer class for RefScore academic application.

This module provides the core RefScoreAnalyzer class that orchestrates
the entire analysis pipeline from document parsing to score computation.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ..models.document import Document, Sentence, DocPack
from ..models.source import Source
from ..models.scoring import SourceScore, RefEvidence
from .document_parser import DocumentParser
from .source_loader import SourceLoader
from .scoring import ScoringEngine
from .quality_evaluator import QualityEvaluator
from ..utils.config import Config
from ..utils.exceptions import ProcessingError, ValidationError


log = logging.getLogger(__name__)


class RefScoreAnalyzer:
    """
    Main analyzer class for RefScore academic application.
    
    This class provides a high-level interface for analyzing academic
    documents against reference sources, computing scores, and generating reports.
    
    Attributes:
        config: Application configuration
        document_parser: Document parsing engine
        source_loader: Source loading engine
        scoring_engine: Scoring computation engine
    """
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the RefScore analyzer.
        
        Args:
            config: Optional configuration object. If not provided,
                   default configuration will be used.
        """
        self.config = config or Config()
        self.document_parser = DocumentParser(self.config)
        self.source_loader = SourceLoader(self.config)
        self.scoring_engine = ScoringEngine(self.config)
        self.quality_evaluator = QualityEvaluator(self.config)
        
        log.info("RefScore analyzer initialized")
    
    def load_document(self, document_path: Union[str, Path]) -> Document:
        """
        Load and parse an academic document.
        
        Args:
            document_path: Path to the document file (.tex or .pdf)
            
        Returns:
            Parsed Document object
            
        Raises:
            ValidationError: If document path is invalid
            ProcessingError: If document parsing fails
        """
        try:
            document_path = Path(document_path)
            if not document_path.exists():
                raise ValidationError(f"Document file not found: {document_path}")
            
            if not document_path.suffix.lower() in ['.tex', '.pdf']:
                raise ValidationError(f"Unsupported document format: {document_path.suffix}")
            
            log.info(f"Loading document: {document_path}")
            doc_pack = self.document_parser.parse_document(str(document_path))
            document = doc_pack.to_document()
            
            log.info(f"Document loaded: {len(document.sentences)} sentences, "
                    f"{len(document.sections)} sections")
            return document
            
        except ValidationError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to load document: {e}")
    
    def load_sources(self, source_paths: List[Union[str, Path]]) -> List[Source]:
        """
        Load and parse reference sources from multiple files.
        
        Args:
            source_paths: List of paths to source files (.bib, .json, .csv)
            
        Returns:
            List of parsed Source objects
            
        Raises:
            ValidationError: If source paths are invalid
            ProcessingError: If source loading fails
        """
        try:
            if not source_paths:
                raise ValidationError("At least one source file must be provided")
            
            log.info(f"Loading sources from {len(source_paths)} files")
            sources = []
            
            for path in source_paths:
                path = Path(path)
                if not path.exists():
                    log.warning(f"Source file not found: {path}")
                    continue
                
                try:
                    file_sources = self.source_loader.load_sources(str(path))
                    sources.extend(file_sources)
                    log.info(f"Loaded {len(file_sources)} sources from {path}")
                except Exception as e:
                    log.warning(f"Failed to load sources from {path}: {e}")
            
            # Remove duplicates based on DOI or title
            unique_sources = self._deduplicate_sources(sources)
            log.info(f"Total unique sources loaded: {len(unique_sources)}")
            return unique_sources
            
        except ValidationError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to load sources: {e}")
    
    def compute_scores(self, document: Document, sources: List[Source]) -> List[SourceScore]:
        """
        Compute RefScores for all sources against the document.
        
        Args:
            document: Parsed academic document
            sources: List of academic sources
            
        Returns:
            List of SourceScore objects sorted by score (descending)
            
        Raises:
            ProcessingError: If scoring computation fails
        """
        try:
            if not document.sentences:
                raise ValidationError("Document contains no sentences")
            if not sources:
                raise ValidationError("No sources provided for scoring")
            
            log.info(f"Computing scores for {len(sources)} sources against "
                    f"{len(document.sentences)} sentences")
            
            scores = self.scoring_engine.compute_refscores(document, sources)
            log.info(f"Scoring completed. Top score: {scores[0].refscore:.4f}")
            return scores
            
        except ValidationError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to compute scores: {e}")
    
    def get_coverage_report(self, document: Document, scores: List[SourceScore]) -> Dict[str, Any]:
        """
        Generate a coverage report showing section-wise support statistics.
        
        Args:
            document: Parsed academic document
            scores: List of source scores
            
        Returns:
            Dictionary containing coverage statistics
        """
        return self.scoring_engine.section_coverage_report(document, scores)
    
    def get_weak_sentences(self, document: Document, scores: List[SourceScore], 
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Identify sentences with the weakest source support.
        
        Args:
            document: Parsed academic document
            scores: List of source scores
            top_k: Number of weakest sentences to return
            
        Returns:
            List of dictionaries containing weak sentence information
        """
        return self.scoring_engine.weakest_sentences(document, scores, top_k)
    
    def generate_report(self, document: Document, sources: List[Source], 
                       scores: List[SourceScore], output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Generate comprehensive analysis reports.
        
        Args:
            document: Parsed academic document
            sources: List of academic sources
            scores: Computed source scores
            output_dir: Directory to save reports
            
        Returns:
            Dictionary mapping report types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reports = {}
        
        # Sources ranking report
        sources_file = output_dir / "sources_ranked.json"
        self._save_json(sources_file, self._scores_to_json(scores))
        reports["sources"] = str(sources_file)
        
        # Coverage report
        coverage = self.get_coverage_report(document, scores)
        coverage_file = output_dir / "section_coverage.json"
        self._save_json(coverage_file, coverage)
        reports["coverage"] = str(coverage_file)
        
        # Weak sentences report
        weak_sentences = self.get_weak_sentences(document, scores)
        gaps_file = output_dir / "gaps.json"
        self._save_json(gaps_file, weak_sentences)
        reports["gaps"] = str(gaps_file)

        quality = self.get_quality_assessment(document, scores)
        quality_file = output_dir / "quality_assessment.json"
        self._save_json(quality_file, quality)
        reports["quality"] = str(quality_file)
        
        log.info(f"Reports generated in {output_dir}")
        return reports

    def get_quality_assessment(self, document: Document, scores: List[SourceScore]) -> Dict[str, Any]:
        return self.quality_evaluator.assess(document, scores)
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources based on DOI or title."""
        seen = set()
        unique_sources = []
        
        for source in sources:
            key = source.doi.lower() if source.doi else source.title.lower()
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(source)
        
        return unique_sources
    
    def _scores_to_json(self, scores: List[SourceScore]) -> List[Dict[str, Any]]:
        """Convert scores to JSON-serializable format."""
        return [score.to_dict() for score in scores]
    
    def _save_json(self, file_path: Path, data: Any) -> None:
        """Save data to JSON file."""
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
