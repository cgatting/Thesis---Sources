"""
Test suite for RefScore core functionality.

This module contains tests for the core RefScore analyzer and related components.
"""

import pytest
import logging
from pathlib import Path

from refscore.core.analyzer import RefScoreAnalyzer
from refscore.core.document_parser import DocumentParser
from refscore.core.source_loader import SourceLoader
from refscore.core.scoring import ScoringEngine
from refscore.utils.config import Config
from refscore.utils.exceptions import ValidationError, ProcessingError


log = logging.getLogger(__name__)


class TestRefScoreAnalyzer:
    """Test cases for RefScoreAnalyzer."""
    
    def test_analyzer_initialization(self, sample_config):
        """Test analyzer initialization."""
        analyzer = RefScoreAnalyzer(sample_config)
        assert analyzer.config == sample_config
        assert analyzer.document_parser is not None
        assert analyzer.source_loader is not None
        assert analyzer.scoring_engine is not None
    
    def test_analyzer_default_config(self):
        """Test analyzer with default configuration."""
        analyzer = RefScoreAnalyzer()
        assert analyzer.config is not None
        assert isinstance(analyzer.config, Config)
    
    def test_load_document_success(self, sample_latex_file, sample_config):
        """Test successful document loading."""
        analyzer = RefScoreAnalyzer(sample_config)
        document = analyzer.load_document(sample_latex_file)
        
        assert document is not None
        assert len(document.sentences) > 0
        assert len(document.sections) > 0
        assert document.meta["type"] == "tex"
    
    def test_load_document_invalid_path(self, sample_config):
        """Test document loading with invalid path."""
        analyzer = RefScoreAnalyzer(sample_config)
        
        with pytest.raises(ValidationError):
            analyzer.load_document("nonexistent_file.tex")
    
    def test_load_document_unsupported_format(self, temp_dir, sample_config):
        """Test document loading with unsupported format."""
        analyzer = RefScoreAnalyzer(sample_config)
        invalid_file = temp_dir / "test.docx"
        invalid_file.write_text("invalid content")
        
        with pytest.raises(ValidationError):
            analyzer.load_document(invalid_file)
    
    def test_load_sources_success(self, sample_bibtex_file, sample_config):
        """Test successful source loading."""
        analyzer = RefScoreAnalyzer(sample_config)
        sources = analyzer.load_sources([sample_bibtex_file])
        
        assert len(sources) > 0
        assert all(source.is_valid for source in sources)
    
    def test_load_sources_multiple_files(self, sample_bibtex_file, sample_json_file, sample_config):
        """Test loading sources from multiple files."""
        analyzer = RefScoreAnalyzer(sample_config)
        sources = analyzer.load_sources([sample_bibtex_file, sample_json_file])
        
        assert len(sources) > 0
        assert len(sources) >= 2  # At least one from each file
    
    def test_load_sources_empty_list(self, sample_config):
        """Test loading sources with empty list."""
        analyzer = RefScoreAnalyzer(sample_config)
        
        with pytest.raises(ValidationError):
            analyzer.load_sources([])
    
    def test_compute_scores_success(self, sample_latex_file, sample_bibtex_file, sample_config):
        """Test successful score computation."""
        analyzer = RefScoreAnalyzer(sample_config)
        document = analyzer.load_document(sample_latex_file)
        sources = analyzer.load_sources([sample_bibtex_file])
        
        scores = analyzer.compute_scores(document, sources)
        
        assert len(scores) > 0
        assert all(0.0 <= score.refscore <= 1.0 for score in scores)
        assert scores[0].refscore >= scores[-1].refscore  # Should be sorted
    
    def test_compute_scores_empty_sources(self, sample_latex_file, sample_config):
        """Test score computation with empty sources."""
        analyzer = RefScoreAnalyzer(sample_config)
        document = analyzer.load_document(sample_latex_file)
        
        with pytest.raises(ValidationError):
            analyzer.compute_scores(document, [])
    
    def test_compute_scores_empty_document(self, sample_bibtex_file, sample_config):
        """Test score computation with empty document."""
        analyzer = RefScoreAnalyzer(sample_config)
        sources = analyzer.load_sources([sample_bibtex_file])
        
        # Create empty document and expect validation to fail
        from refscore.models.document import Document
        with pytest.raises(ValueError):
            empty_doc = Document(sentences=[], sections=[], meta={})
            analyzer.compute_scores(empty_doc, sources)
    
    def test_get_coverage_report(self, sample_latex_file, sample_bibtex_file, sample_config):
        """Test coverage report generation."""
        analyzer = RefScoreAnalyzer(sample_config)
        document = analyzer.load_document(sample_latex_file)
        sources = analyzer.load_sources([sample_bibtex_file])
        scores = analyzer.compute_scores(document, sources)
        
        coverage = analyzer.get_coverage_report(document, scores)
        
        assert "sections" in coverage
        assert "section_stats" in coverage
        assert "overall_coverage" in coverage
        assert len(coverage["sections"]) == len(document.sections)
    
    def test_get_weak_sentences(self, sample_latex_file, sample_bibtex_file, sample_config):
        """Test weak sentences identification."""
        analyzer = RefScoreAnalyzer(sample_config)
        document = analyzer.load_document(sample_latex_file)
        sources = analyzer.load_sources([sample_bibtex_file])
        scores = analyzer.compute_scores(document, sources)
        
        weak_sentences = analyzer.get_weak_sentences(document, scores, top_k=5)
        
        assert len(weak_sentences) <= 5
        assert all("idx" in ws for ws in weak_sentences)
        assert all("section" in ws for ws in weak_sentences)
        assert all("support" in ws for ws in weak_sentences)
        assert all("sentence" in ws for ws in weak_sentences)
    
    def test_generate_report(self, sample_latex_file, sample_bibtex_file, temp_dir, sample_config):
        """Test comprehensive report generation."""
        analyzer = RefScoreAnalyzer(sample_config)
        document = analyzer.load_document(sample_latex_file)
        sources = analyzer.load_sources([sample_bibtex_file])
        scores = analyzer.compute_scores(document, sources)
        
        reports = analyzer.generate_report(document, sources, scores, temp_dir)
        
        assert "sources" in reports
        assert "coverage" in reports
        assert "gaps" in reports
        
        # Check that files were created
        assert Path(reports["sources"]).exists()
        assert Path(reports["coverage"]).exists()
        assert Path(reports["gaps"]).exists()
    
    def test_deduplicate_sources(self, sample_config):
        """Test source deduplication."""
        analyzer = RefScoreAnalyzer(sample_config)
        
        # Create duplicate sources
        from refscore.models.source import Source
        sources = [
            Source(source_id="1", title="Test Paper", doi="10.1234/test.2023"),
            Source(source_id="2", title="Test Paper", doi="10.1234/test.2023"),  # Duplicate
            Source(source_id="3", title="Different Paper", doi="10.5678/other.2023"),
        ]
        
        unique_sources = analyzer._deduplicate_sources(sources)
        
        assert len(unique_sources) == 2  # Should remove one duplicate
        assert all(source.source_id in ["1", "3"] for source in unique_sources)


class TestDocumentParser:
    """Test cases for DocumentParser."""
    
    def test_parser_initialization(self, sample_config):
        """Test parser initialization."""
        parser = DocumentParser(sample_config)
        assert parser.config == sample_config
    
    def test_parse_latex_document(self, sample_latex_file, sample_config):
        """Test LaTeX document parsing."""
        parser = DocumentParser(sample_config)
        doc_pack = parser.parse_document(str(sample_latex_file))
        
        assert doc_pack is not None
        assert len(doc_pack.sentences) > 0
        assert len(doc_pack.sections) > 0
        assert doc_pack.meta["type"] == "tex"
        assert doc_pack.meta["path"] == str(sample_latex_file)
    
    def test_parse_pdf_document(self, sample_config):
        """Test PDF document parsing."""
        # This test requires PyMuPDF to be installed
        pytest.importorskip("fitz")
        
        # Create a simple PDF file for testing
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            # Create temporary PDF file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                pdf_path = Path(tmp.name)
            
            # Generate simple PDF
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "Test PDF Document")
            c.drawString(100, 700, "Introduction")
            c.drawString(100, 680, "This is a test PDF document for RefScore analysis.")
            c.drawString(100, 650, "Methods")
            c.drawString(100, 630, "We used various computational methods.")
            c.save()
            
            parser = DocumentParser(sample_config)
            doc_pack = parser.parse_document(str(pdf_path))
            
            assert doc_pack is not None
            assert len(doc_pack.sentences) > 0
            assert doc_pack.meta["type"] == "pdf"
            
            # Cleanup
            pdf_path.unlink()
            
        except ImportError:
            pytest.skip("reportlab not available for PDF creation")
    
    def test_parse_document_invalid_format(self, temp_dir, sample_config):
        """Test parsing document with invalid format."""
        parser = DocumentParser(sample_config)
        invalid_file = temp_dir / "test.txt"
        invalid_file.write_text("invalid content")
        
        with pytest.raises(ValidationError):
            parser.parse_document(str(invalid_file))
    
    def test_parse_document_nonexistent(self, sample_config):
        """Test parsing nonexistent document."""
        parser = DocumentParser(sample_config)
        
        with pytest.raises(ValidationError):
            parser.parse_document("nonexistent_file.tex")
    
    def test_get_supported_formats(self, sample_config):
        """Test getting supported formats."""
        parser = DocumentParser(sample_config)
        formats = parser.get_supported_formats()
        
        assert ".tex" in formats
        assert ".pdf" in formats
        assert ".docx" in formats
        assert len(formats) == 3
    
    def test_validate_document(self, sample_latex_file, sample_config):
        """Test document validation."""
        parser = DocumentParser(sample_config)
        
        # Valid document
        assert parser.validate_document(str(sample_latex_file))
        
        # Invalid document
        assert not parser.validate_document("nonexistent_file.tex")
        assert not parser.validate_document("test.odt")


class TestSourceLoader:
    """Test cases for SourceLoader."""
    
    def test_loader_initialization(self, sample_config):
        """Test loader initialization."""
        loader = SourceLoader(sample_config)
        assert loader.config == sample_config
    
    def test_load_bibtex_sources(self, sample_bibtex_file, sample_config):
        """Test loading sources from BibTeX file."""
        loader = SourceLoader(sample_config)
        sources = loader.load_sources(str(sample_bibtex_file))
        
        assert len(sources) > 0
        assert all(source.is_valid for source in sources)
        assert any(source.doi for source in sources)
        assert any(source.authors for source in sources)
    
    def test_load_json_sources(self, sample_json_file, sample_config):
        """Test loading sources from JSON file."""
        loader = SourceLoader(sample_config)
        sources = loader.load_sources(str(sample_json_file))
        
        assert len(sources) > 0
        assert all(source.is_valid for source in sources)
    
    def test_load_csv_sources(self, sample_csv_file, sample_config):
        """Test loading sources from CSV file."""
        loader = SourceLoader(sample_config)
        sources = loader.load_sources(str(sample_csv_file))
        
        assert len(sources) > 0
        assert all(source.is_valid for source in sources)
    
    def test_load_unsupported_format(self, temp_dir, sample_config):
        """Test loading sources from unsupported format."""
        loader = SourceLoader(sample_config)
        invalid_file = temp_dir / "test.xml"
        invalid_file.write_text("<xml>invalid</xml>")
        
        # Should try to load as DOI list
        sources = loader.load_sources(str(invalid_file))
        assert len(sources) == 0  # No valid DOIs in the file
    
    def test_get_supported_formats(self, sample_config):
        """Test getting supported formats."""
        loader = SourceLoader(sample_config)
        formats = loader.get_supported_formats()
        
        expected_formats = ['.bib', '.json', '.csv', '.txt']
        for fmt in expected_formats:
            assert fmt in formats
    
    def test_validate_source_file(self, sample_bibtex_file, sample_config):
        """Test source file validation."""
        loader = SourceLoader(sample_config)
        
        # Valid file
        assert loader.validate_source_file(str(sample_bibtex_file))
        
        # Invalid file
        assert not loader.validate_source_file("nonexistent_file.bib")


class TestScoringEngine:
    """Test cases for ScoringEngine."""
    
    def test_engine_initialization(self, sample_config):
        """Test engine initialization."""
        engine = ScoringEngine(sample_config)
        assert engine.config == sample_config
        assert engine.weights == ScoringEngine.DEFAULT_WEIGHTS
    
    def test_alignment_score(self, sample_config):
        """Test text alignment scoring."""
        engine = ScoringEngine(sample_config)
        # Force fallback to Jaccard for deterministic testing and to avoid model loading issues
        engine.embedder = None
        engine.cross_encoder = None
        engine.tfidf_vectorizer = None
        
        # Similar texts should have high alignment
        text1 = "machine learning algorithms for natural language processing"
        text2 = "machine learning methods for NLP applications"
        score = engine._alignment_score(text1, text2)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should have some similarity
        
        # Different texts should have low alignment
        text3 = "quantum physics and particle interactions"
        score2 = engine._alignment_score(text1, text3)
        assert score2 < 0.3  # Should have low similarity
    
    def test_entity_overlap(self, sample_config):
        """Test entity overlap scoring."""
        engine = ScoringEngine(sample_config)
        
        text1 = "Google and Microsoft are technology companies"
        text2 = "Apple and Google compete in the smartphone market"
        overlap = engine._entity_overlap(text1, text2)
        
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.0  # Should detect "Google" overlap
    
    def test_number_unit_match(self, sample_config):
        """Test number and unit matching."""
        engine = ScoringEngine(sample_config)
        
        sentence = "The algorithm achieved 95% accuracy"
        source = "Our method obtained 94% precision on the test set"
        score, reasons = engine._number_unit_match(sentence, source)
        
        assert 0.0 <= score <= 1.0
        assert len(reasons) >= 0
    
    def test_method_metric_overlap(self, sample_config):
        """Test method and metric overlap."""
        engine = ScoringEngine(sample_config)
        
        text1 = "We used SVM classification with cross-validation"
        text2 = "Our approach employed SVM with 5-fold cross-validation"
        overlap = engine._method_metric_overlap(text1, text2)
        
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.0  # Should detect SVM and cross-validation
    
    def test_recency_score(self, sample_config):
        """Test recency scoring."""
        engine = ScoringEngine(sample_config)
        
        # Recent year should have high score
        recent_score = engine._recency_score(2023)
        assert 0.0 <= recent_score <= 1.0
        assert recent_score > 0.5
        
        # Old year should have low score
        old_score = engine._recency_score(2000)
        assert old_score < 0.5
        
        # None year should have zero score
        none_score = engine._recency_score(None)
        assert none_score == 0.0
    
    def test_authority_score(self, sample_config):
        """Test authority scoring."""
        engine = ScoringEngine(sample_config)
        
        # High authority venue
        high_score = engine._authority_score("Nature")
        assert high_score == 1.0
        
        # Medium authority venue
        medium_score = engine._authority_score("Journal of Machine Learning Research")
        assert medium_score == 0.6
        
        # Low authority venue
        low_score = engine._authority_score("arXiv preprint")
        assert low_score == 0.35
        
        # Empty venue
        empty_score = engine._authority_score("")
        assert empty_score == 0.0
    
    def test_compute_refscores(self, sample_config):
        """Test comprehensive score computation."""
        engine = ScoringEngine(sample_config)
        
        # Create test document
        from refscore.models.document import Document, Sentence
        sentences = [
            Sentence("Machine learning algorithms are widely used", "Introduction", 0),
            Sentence("Deep learning has revolutionized computer vision", "Methods", 1),
        ]
        document = Document(sentences=sentences, sections=["Introduction", "Methods"], meta={})
        
        # Create test sources
        from refscore.models.source import Source
        sources = [
            Source(source_id="1", title="Machine Learning Fundamentals", 
                  abstract="This paper discusses machine learning algorithms and their applications"),
            Source(source_id="2", title="Deep Learning for Computer Vision",
                  abstract="Exploring deep learning techniques in computer vision tasks"),
        ]
        
        scores = engine.compute_refscores(document, sources)
        
        assert len(scores) == 2
        assert all(0.0 <= score.refscore <= 1.0 for score in scores)
        assert all(len(score.per_sentence) == 2 for score in scores)  # Should have scores for both sentences
    
    def test_section_coverage_report(self, sample_config):
        """Test section coverage report generation."""
        engine = ScoringEngine(sample_config)
        
        # Create test document and scores
        from refscore.models.document import Document, Sentence
        sentences = [
            Sentence("Introduction to machine learning", "Introduction", 0),
            Sentence("Deep learning methods", "Methods", 1),
            Sentence("Results and discussion", "Results", 2),
        ]
        document = Document(sentences=sentences, sections=["Introduction", "Methods", "Results"], meta={})
        
        from refscore.models.source import Source
        from refscore.models.scoring import SourceScore, RefEvidence
        
        source = Source(source_id="1", title="ML Paper", abstract="Machine learning and deep learning")
        evidence1 = RefEvidence(0.8, 0.5, 0.3, 0.4, 0.6, 0.7, ["good match"])
        evidence2 = RefEvidence(0.9, 0.6, 0.4, 0.5, 0.7, 0.8, ["excellent match"])
        evidence3 = RefEvidence(0.3, 0.2, 0.1, 0.2, 0.4, 0.5, ["poor match"])
        
        score = SourceScore(source, 0.7, {0: evidence1, 1: evidence2, 2: evidence3})
        
        coverage = engine.section_coverage_report(document, [score])
        
        assert "sections" in coverage
        assert "section_stats" in coverage
        assert "overall_coverage" in coverage
        assert len(coverage["sections"]) == 3
        assert "Introduction" in coverage["section_stats"]
        assert "Methods" in coverage["section_stats"]
        assert "Results" in coverage["section_stats"]
    
    def test_weakest_sentences(self, sample_config):
        """Test weak sentences identification."""
        engine = ScoringEngine(sample_config)
        
        # Create test document and scores
        from refscore.models.document import Document, Sentence
        sentences = [
            Sentence("This is a well-supported sentence about machine learning", "Introduction", 0),
            Sentence("This sentence has poor source support", "Methods", 1),
            Sentence("Another weak sentence with no good matches", "Results", 2),
        ]
        document = Document(sentences=sentences, sections=["Introduction", "Methods", "Results"], meta={})
        
        from refscore.models.source import Source
        from refscore.models.scoring import SourceScore, RefEvidence
        
        source = Source(source_id="1", title="ML Sources", abstract="Machine learning references")
        evidence1 = RefEvidence(0.8, 0.6, 0.4, 0.5, 0.7, 0.8, ["good support"])
        evidence2 = RefEvidence(0.2, 0.1, 0.1, 0.1, 0.3, 0.4, ["poor support"])
        evidence3 = RefEvidence(0.1, 0.1, 0.0, 0.1, 0.2, 0.3, ["very poor support"])
        
        score = SourceScore(source, 0.4, {0: evidence1, 1: evidence2, 2: evidence3})
        
        weak_sentences = engine.weakest_sentences(document, [score], top_k=2)
        
        assert len(weak_sentences) == 2
        assert weak_sentences[0]["idx"] == 2  # Should be sorted by support score
        assert weak_sentences[1]["idx"] == 1
        assert all("sentence" in ws for ws in weak_sentences)
        assert all("support" in ws for ws in weak_sentences)
