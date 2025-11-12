"""
Test suite for RefScore models.

This module contains tests for data models used throughout the application.
"""

import pytest
from pathlib import Path

from refscore.models.document import Document, Sentence, DocPack
from refscore.models.source import Source
from refscore.models.scoring import SourceScore, RefEvidence
from refscore.utils.exceptions import ValidationError


class TestDocumentModels:
    """Test cases for document models."""
    
    def test_sentence_creation(self):
        """Test Sentence model creation."""
        sentence = Sentence(
            text="This is a test sentence.",
            section="Introduction",
            idx=0
        )
        
        assert sentence.text == "This is a test sentence."
        assert sentence.section == "Introduction"
        assert sentence.idx == 0
    
    def test_sentence_validation_empty_text(self):
        """Test sentence validation with empty text."""
        with pytest.raises(ValueError, match="Sentence text cannot be empty"):
            Sentence(text="", section="Introduction", idx=0)
    
    def test_sentence_validation_negative_index(self):
        """Test sentence validation with negative index."""
        with pytest.raises(ValueError, match="Sentence index must be non-negative"):
            Sentence(text="Test sentence.", section="Introduction", idx=-1)
    
    def test_sentence_validation_empty_section(self):
        """Test sentence validation with empty section."""
        with pytest.raises(ValueError, match="Section name cannot be empty"):
            Sentence(text="Test sentence.", section="", idx=0)
    
    def test_document_creation(self):
        """Test Document model creation."""
        sentences = [
            Sentence("Sentence 1", "Introduction", 0),
            Sentence("Sentence 2", "Methods", 1),
        ]
        
        document = Document(
            sentences=sentences,
            sections=["Introduction", "Methods"],
            meta={"path": "test.doc", "type": "test"}
        )
        
        assert len(document.sentences) == 2
        assert len(document.sections) == 2
        assert document.meta["path"] == "test.doc"
    
    def test_document_validation_empty_sentences(self):
        """Test document validation with empty sentences."""
        with pytest.raises(ValueError, match="Document must contain at least one sentence"):
            Document(sentences=[], sections=["Introduction"], meta={})
    
    def test_document_validation_empty_sections(self):
        """Test document validation with empty sections."""
        sentences = [Sentence("Test sentence.", "Introduction", 0)]
        with pytest.raises(ValueError, match="Document must have at least one section"):
            Document(sentences=sentences, sections=[], meta={})
    
    def test_document_validation_invalid_meta(self):
        """Test document validation with invalid metadata."""
        sentences = [Sentence("Test sentence.", "Introduction", 0)]
        with pytest.raises(ValueError, match="Document metadata must be a dictionary"):
            Document(sentences=sentences, sections=["Introduction"], meta="invalid")
    
    def test_get_sentences_by_section(self):
        """Test getting sentences by section."""
        sentences = [
            Sentence("Intro sentence.", "Introduction", 0),
            Sentence("Methods sentence.", "Methods", 1),
            Sentence("Another intro sentence.", "Introduction", 2),
        ]
        
        document = Document(sentences=sentences, sections=["Introduction", "Methods"], meta={})
        
        intro_sentences = document.get_sentences_by_section("Introduction")
        assert len(intro_sentences) == 2
        assert all(s.section == "Introduction" for s in intro_sentences)
        
        methods_sentences = document.get_sentences_by_section("Methods")
        assert len(methods_sentences) == 1
        assert methods_sentences[0].section == "Methods"
    
    def test_get_section_stats(self):
        """Test getting section statistics."""
        sentences = [
            Sentence("Intro 1", "Introduction", 0),
            Sentence("Intro 2", "Introduction", 1),
            Sentence("Methods 1", "Methods", 2),
            Sentence("Results 1", "Results", 3),
            Sentence("Results 2", "Results", 4),
            Sentence("Results 3", "Results", 5),
        ]
        
        document = Document(sentences=sentences, sections=["Introduction", "Methods", "Results"], meta={})
        stats = document.get_section_stats()
        
        assert stats["Introduction"] == 2
        assert stats["Methods"] == 1
        assert stats["Results"] == 3
    
    def test_docpack_compatibility(self):
        """Test DocPack compatibility with Document."""
        sentences = [Sentence("Test sentence.", "Introduction", 0)]
        docpack = DocPack(sentences=sentences, sections=["Introduction"], meta={"type": "test"})
        
        # Test conversion to Document
        document = docpack.to_document()
        assert isinstance(document, Document)
        assert len(document.sentences) == 1
        assert document.sections == ["Introduction"]
        
        # Test conversion back to DocPack
        new_docpack = DocPack.from_document(document)
        assert isinstance(new_docpack, DocPack)
        assert len(new_docpack.sentences) == 1
        assert new_docpack.sections == ["Introduction"]


class TestSourceModels:
    """Test cases for source models."""
    
    def test_source_creation(self):
        """Test Source model creation."""
        source = Source(
            source_id="test_2023",
            title="Test Research Paper",
            abstract="This is a test abstract.",
            year=2023,
            venue="Test Journal",
            doi="10.1234/test.2023",
            authors=["John Doe", "Jane Smith"],
            extra={"note": "test note"}
        )
        
        assert source.source_id == "test_2023"
        assert source.title == "Test Research Paper"
        assert source.abstract == "This is a test abstract."
        assert source.year == 2023
        assert source.venue == "Test Journal"
        assert source.doi == "10.1234/test.2023"
        assert len(source.authors) == 2
        assert source.extra["note"] == "test note"
    
    def test_source_validation_empty_id(self):
        """Test source validation with empty ID."""
        with pytest.raises(ValueError, match="Source ID cannot be empty"):
            Source(source_id="", title="Test Paper")
    
    def test_source_is_valid(self):
        """Test source validity check."""
        # Valid source with title
        source1 = Source(source_id="1", title="Test Paper")
        assert source1.is_valid
        
        # Valid source with abstract
        source2 = Source(source_id="2", abstract="Test abstract")
        assert source2.is_valid
        
        # Invalid source (empty title and abstract)
        source3 = Source(source_id="3")
        assert not source3.is_valid
    
    def test_author_string_single_author(self):
        """Test author string with single author."""
        source = Source(source_id="1", authors=["John Doe"])
        assert source.author_string == "John Doe"
    
    def test_author_string_two_authors(self):
        """Test author string with two authors."""
        source = Source(source_id="1", authors=["John Doe", "Jane Smith"])
        assert source.author_string == "John Doe and Jane Smith"
    
    def test_author_string_multiple_authors(self):
        """Test author string with multiple authors."""
        source = Source(source_id="1", authors=["John Doe", "Jane Smith", "Bob Johnson"])
        assert source.author_string == "John Doe et al."
    
    def test_author_string_no_authors(self):
        """Test author string with no authors."""
        source = Source(source_id="1")
        assert source.author_string == "Unknown authors"
    
    def test_citation_string_complete(self):
        """Test citation string with complete information."""
        source = Source(
            source_id="1",
            authors=["John Doe", "Jane Smith"],
            year=2023,
            title="Test Paper",
            venue="Test Journal",
            doi="10.1234/test.2023"
        )
        
        citation = source.citation_string
        assert "John Doe and Jane Smith" in citation
        assert "(2023)" in citation
        assert '"Test Paper"' in citation
        assert "Test Journal" in citation
        assert "DOI: 10.1234/test.2023" in citation
    
    def test_citation_string_partial(self):
        """Test citation string with partial information."""
        source = Source(
            source_id="1",
            title="Test Paper",
            year=2023
        )
        
        citation = source.citation_string
        assert "Unknown authors" in citation
        assert "(2023)" in citation
        assert '"Test Paper"' in citation
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        source = Source(
            source_id="test_1",
            title="Test Paper",
            authors=["John Doe"],
            year=2023
        )
        
        source_dict = source.to_dict()
        
        assert source_dict["source_id"] == "test_1"
        assert source_dict["title"] == "Test Paper"
        assert source_dict["authors"] == ["John Doe"]
        assert source_dict["year"] == 2023
        assert source_dict["abstract"] == ""
        assert source_dict["venue"] == ""
        assert source_dict["doi"] == ""
        assert source_dict["extra"] == {}
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        source_dict = {
            "source_id": "test_1",
            "title": "Test Paper",
            "authors": ["John Doe", "Jane Smith"],
            "year": 2023,
            "venue": "Test Journal",
            "doi": "10.1234/test.2023",
            "abstract": "Test abstract",
            "extra": {"note": "test"}
        }
        
        source = Source.from_dict(source_dict)
        
        assert source.source_id == "test_1"
        assert source.title == "Test Paper"
        assert source.authors == ["John Doe", "Jane Smith"]
        assert source.year == 2023
        assert source.venue == "Test Journal"
        assert source.doi == "10.1234/test.2023"
        assert source.abstract == "Test abstract"
        assert source.extra["note"] == "test"


class TestScoringModels:
    """Test cases for scoring models."""
    
    def test_ref_evidence_creation(self):
        """Test RefEvidence model creation."""
        evidence = RefEvidence(
            alignment=0.8,
            entities=0.6,
            number_unit=0.4,
            method_metric=0.5,
            recency=0.7,
            authority=0.8,
            reasons=["good semantic match", "entity overlap"]
        )
        
        assert evidence.alignment == 0.8
        assert evidence.entities == 0.6
        assert evidence.number_unit == 0.4
        assert evidence.method_metric == 0.5
        assert evidence.recency == 0.7
        assert evidence.authority == 0.8
        assert len(evidence.reasons) == 2
    
    def test_ref_evidence_validation_out_of_range(self):
        """Test RefEvidence validation with out-of-range values."""
        with pytest.raises(ValueError, match="alignment must be between 0.0 and 1.0"):
            RefEvidence(
                alignment=1.5,  # Out of range
                entities=0.6,
                number_unit=0.4,
                method_metric=0.5,
                recency=0.7,
                authority=0.8,
                reasons=[]
            )
    
    def test_ref_evidence_validation_non_numeric(self):
        """Test RefEvidence validation with non-numeric values."""
        with pytest.raises(ValueError, match="alignment must be a number"):
            RefEvidence(
                alignment="invalid",  # Non-numeric
                entities=0.6,
                number_unit=0.4,
                method_metric=0.5,
                recency=0.7,
                authority=0.8,
                reasons=[]
            )
    
    def test_weighted_score_default_weights(self):
        """Test weighted score calculation with default weights."""
        evidence = RefEvidence(
            alignment=0.8,
            entities=0.6,
            number_unit=0.4,
            method_metric=0.5,
            recency=0.7,
            authority=0.8,
            reasons=[]
        )
        
        weighted_score = evidence.weighted_score()
        
        # Manual calculation with default weights
        expected = (0.45 * 0.8 + 0.15 * 0.6 + 0.20 * 0.4 + 0.10 * 0.5 + 0.07 * 0.7 + 0.03 * 0.8)
        assert abs(weighted_score - expected) < 0.001
    
    def test_weighted_score_custom_weights(self):
        """Test weighted score calculation with custom weights."""
        evidence = RefEvidence(
            alignment=0.8,
            entities=0.6,
            number_unit=0.4,
            method_metric=0.5,
            recency=0.7,
            authority=0.8,
            reasons=[]
        )
        
        custom_weights = {
            "alignment": 0.5,
            "entities": 0.3,
            "number_unit": 0.1,
            "method_metric": 0.05,
            "recency": 0.03,
            "authority": 0.02,
        }
        
        weighted_score = evidence.weighted_score(custom_weights)
        
        # Manual calculation with custom weights
        expected = (0.5 * 0.8 + 0.3 * 0.6 + 0.1 * 0.4 + 0.05 * 0.5 + 0.03 * 0.7 + 0.02 * 0.8)
        assert abs(weighted_score - expected) < 0.001
    
    def test_weighted_score_zero_weights(self):
        """Test weighted score calculation with zero total weights."""
        evidence = RefEvidence(
            alignment=0.8,
            entities=0.6,
            number_unit=0.4,
            method_metric=0.5,
            recency=0.7,
            authority=0.8,
            reasons=[]
        )
        
        zero_weights = {
            "alignment": 0.0,
            "entities": 0.0,
            "number_unit": 0.0,
            "method_metric": 0.0,
            "recency": 0.0,
            "authority": 0.0,
        }
        
        weighted_score = evidence.weighted_score(zero_weights)
        assert weighted_score == 0.0
    
    def test_to_dict(self):
        """Test RefEvidence conversion to dictionary."""
        evidence = RefEvidence(
            alignment=0.8,
            entities=0.6,
            number_unit=0.4,
            method_metric=0.5,
            recency=0.7,
            authority=0.8,
            reasons=["reason1", "reason2"]
        )
        
        evidence_dict = evidence.to_dict()
        
        assert evidence_dict["alignment"] == 0.8
        assert evidence_dict["entities"] == 0.6
        assert evidence_dict["number_unit"] == 0.4
        assert evidence_dict["method_metric"] == 0.5
        assert evidence_dict["recency"] == 0.7
        assert evidence_dict["authority"] == 0.8
        assert evidence_dict["reasons"] == ["reason1", "reason2"]
        assert "weighted_score" in evidence_dict
    
    def test_source_score_creation(self):
        """Test SourceScore model creation."""
        source = Source(source_id="test_1", title="Test Paper")
        evidence = RefEvidence(
            alignment=0.8,
            entities=0.6,
            number_unit=0.4,
            method_metric=0.5,
            recency=0.7,
            authority=0.8,
            reasons=[]
        )
        
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={0: evidence, 1: evidence}
        )
        
        assert source_score.source == source
        assert source_score.refscore == 0.75
        assert len(source_score.per_sentence) == 2
    
    def test_source_score_validation_invalid_refscore(self):
        """Test SourceScore validation with invalid refscore."""
        source = Source(source_id="test_1")
        
        with pytest.raises(ValueError, match="refscore must be between 0.0 and 1.0"):
            SourceScore(
                source=source,
                refscore=1.5,  # Out of range
                per_sentence={}
            )
    
    def test_source_score_validation_non_numeric_refscore(self):
        """Test SourceScore validation with non-numeric refscore."""
        source = Source(source_id="test_1")
        
        with pytest.raises(ValueError, match="refscore must be a number"):
            SourceScore(
                source=source,
                refscore="invalid",  # Non-numeric
                per_sentence={}
            )
    
    def test_evidence_count(self):
        """Test evidence count property."""
        source = Source(source_id="test_1")
        evidence = RefEvidence(0.8, 0.6, 0.4, 0.5, 0.7, 0.8, [])
        
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={0: evidence, 1: evidence, 2: evidence}
        )
        
        assert source_score.evidence_count == 3
    
    def test_average_evidence_score(self):
        """Test average evidence score calculation."""
        source = Source(source_id="test_1")
        evidence1 = RefEvidence(0.8, 0.6, 0.4, 0.5, 0.7, 0.8, [])
        evidence2 = RefEvidence(0.6, 0.4, 0.2, 0.3, 0.5, 0.6, [])
        evidence3 = RefEvidence(0.9, 0.7, 0.5, 0.6, 0.8, 0.9, [])
        
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={0: evidence1, 1: evidence2, 2: evidence3}
        )
        
        # Expected average: (0.8 + 0.6 + 0.9) / 3 = 0.766...
        expected = (0.8 + 0.6 + 0.9) / 3
        assert abs(source_score.average_evidence_score - expected) < 0.001
    
    def test_average_evidence_score_empty(self):
        """Test average evidence score with no evidence."""
        source = Source(source_id="test_1")
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={}
        )
        
        assert source_score.average_evidence_score == 0.0
    
    def test_get_top_reasons(self):
        """Test getting top reasons from evidence."""
        source = Source(source_id="test_1")
        
        evidence1 = RefEvidence(0.8, 0.6, 0.4, 0.5, 0.7, 0.8, ["reason1", "reason2"])
        evidence2 = RefEvidence(0.6, 0.4, 0.2, 0.3, 0.5, 0.6, ["reason2", "reason3"])
        evidence3 = RefEvidence(0.9, 0.7, 0.5, 0.6, 0.8, 0.9, ["reason1", "reason3"])
        
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={0: evidence1, 1: evidence2, 2: evidence3}
        )
        
        top_reasons = source_score.get_top_reasons(top_n=2)
        
        # reason1 appears 2 times, reason2 appears 2 times, reason3 appears 2 times
        # The exact order depends on the implementation, but we should get 2 reasons
        assert len(top_reasons) == 2
        assert all(reason in ["reason1", "reason2", "reason3"] for reason in top_reasons)
    
    def test_get_sentence_scores(self):
        """Test getting sentence scores mapping."""
        source = Source(source_id="test_1")
        
        evidence1 = RefEvidence(0.8, 0.6, 0.4, 0.5, 0.7, 0.8, [])
        evidence2 = RefEvidence(0.6, 0.4, 0.2, 0.3, 0.5, 0.6, [])
        
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={0: evidence1, 1: evidence2}
        )
        
        sentence_scores = source_score.get_sentence_scores()
        
        assert len(sentence_scores) == 2
        assert 0 in sentence_scores
        assert 1 in sentence_scores
        assert sentence_scores[0] == evidence1.weighted_score()
        assert sentence_scores[1] == evidence2.weighted_score()
    
    def test_to_dict(self):
        """Test SourceScore conversion to dictionary."""
        source = Source(
            source_id="test_1",
            title="Test Paper",
            authors=["John Doe"],
            year=2023
        )
        
        evidence = RefEvidence(0.8, 0.6, 0.4, 0.5, 0.7, 0.8, ["reason1"])
        
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={0: evidence, 1: evidence}
        )
        
        score_dict = source_score.to_dict()
        
        assert score_dict["refscore"] == 0.75
        assert score_dict["evidence_count"] == 2
        assert score_dict["source"]["source_id"] == "test_1"
        assert score_dict["source"]["title"] == "Test Paper"
        assert "per_sentence" in score_dict
        assert "top_reasons" in score_dict