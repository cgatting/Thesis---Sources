#!/usr/bin/env python3
"""
Test script to verify RefScore academic application installation and functionality.

This script runs basic tests to ensure the application is properly installed
and all core functionality works as expected.
"""

import sys
import os
import tempfile
from pathlib import Path


def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic imports...")
    
    try:
        import refscore
        print(f"✓ RefScore package imported successfully (version {refscore.__version__})")
        
        from refscore.core import RefScoreAnalyzer
        from refscore.utils.config import Config
        from refscore.models import Document, Source, SourceScore
        
        print("✓ All core modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from refscore.utils.config import Config
        
        config = Config()
        weights = config.get_scoring_weights()
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Default alignment weight: {weights['alignment']}")
        print(f"  - Default entities weight: {weights['entities']}")
        
        # Test custom weights
        custom_weights = weights.copy()
        custom_weights['alignment'] = 0.6
        config.set_scoring_weights(custom_weights)
        
        new_weights = config.get_scoring_weights()
        assert new_weights['alignment'] == 0.6
        
        print("✓ Custom weights applied successfully")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_document_models():
    """Test document and sentence models."""
    print("\nTesting document models...")
    
    try:
        from refscore.models.document import Document, Sentence
        
        # Create test sentences
        sentences = [
            Sentence("This is a test sentence.", "Introduction", 0),
            Sentence("Another test sentence here.", "Methods", 1),
            Sentence("Final test sentence.", "Results", 2)
        ]
        
        # Create document
        document = Document(
            sentences=sentences,
            sections=["Introduction", "Methods", "Results"],
            meta={"type": "test", "path": "test.txt"}
        )
        
        print(f"✓ Document created successfully")
        print(f"  - Total sentences: {len(document.sentences)}")
        print(f"  - Total sections: {len(document.sections)}")
        
        # Test section stats
        stats = document.get_section_stats()
        assert stats["Introduction"] == 1
        assert stats["Methods"] == 1
        assert stats["Results"] == 1
        
        print("✓ Section statistics calculated correctly")
        return True
        
    except Exception as e:
        print(f"✗ Document model test failed: {e}")
        return False


def test_source_models():
    """Test source models."""
    print("\nTesting source models...")
    
    try:
        from refscore.models.source import Source
        
        # Create test sources
        sources = [
            Source(
                source_id="test_1",
                title="Test Paper 1",
                authors=["John Doe", "Jane Smith"],
                year=2023,
                venue="Test Journal",
                doi="10.1234/test.2023"
            ),
            Source(
                source_id="test_2",
                title="Test Paper 2",
                authors=["Alice Johnson"],
                year=2022,
                venue="Another Journal",
                doi="10.5678/test.2022"
            )
        ]
        
        print(f"✓ Sources created successfully")
        print(f"  - Total sources: {len(sources)}")
        
        # Test source properties
        for source in sources:
            assert source.is_valid
            print(f"  - {source.title}: {source.author_string} ({source.year})")
        
        print("✓ Source properties work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Source model test failed: {e}")
        return False


def test_scoring_models():
    """Test scoring models."""
    print("\nTesting scoring models...")
    
    try:
        from refscore.models.scoring import RefEvidence, SourceScore
        from refscore.models.source import Source
        
        # Create test evidence
        evidence = RefEvidence(
            alignment=0.8,
            entities=0.6,
            number_unit=0.4,
            method_metric=0.5,
            recency=0.7,
            authority=0.8,
            reasons=["good semantic match", "entity overlap"]
        )
        
        # Create test source score
        source = Source(source_id="test", title="Test Paper")
        source_score = SourceScore(
            source=source,
            refscore=0.75,
            per_sentence={0: evidence, 1: evidence}
        )
        
        print(f"✓ Scoring models created successfully")
        print(f"  - Evidence count: {source_score.evidence_count}")
        print(f"  - Average evidence score: {source_score.average_evidence_score:.3f}")
        
        # Test weighted score calculation
        weighted_score = evidence.weighted_score()
        assert 0.0 <= weighted_score <= 1.0
        
        print(f"✓ Weighted score calculated: {weighted_score:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ Scoring model test failed: {e}")
        return False


def test_basic_scoring():
    """Test basic scoring functionality."""
    print("\nTesting basic scoring functionality...")
    
    try:
        from refscore.core.scoring import ScoringEngine
        from refscore.utils.config import Config
        
        config = Config()
        engine = ScoringEngine(config)
        
        # Test alignment scoring
        text1 = "machine learning algorithms for natural language processing"
        text2 = "machine learning methods for NLP applications"
        
        alignment_score = engine._alignment_score(text1, text2)
        
        print(f"✓ Alignment score calculated: {alignment_score:.3f}")
        assert 0.0 <= alignment_score <= 1.0
        
        # Test entity overlap
        entities_score = engine._entity_overlap(text1, text2)
        
        print(f"✓ Entity overlap score calculated: {entities_score:.3f}")
        assert 0.0 <= entities_score <= 1.0
        
        return True
        
    except Exception as e:
        print(f"✗ Basic scoring test failed: {e}")
        return False


def test_file_creation():
    """Test file creation and basic I/O."""
    print("\nTesting file creation and I/O...")
    
    try:
        import tempfile
        from pathlib import Path
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create test LaTeX file
            latex_file = tmp_path / "test.tex"
            latex_content = r"""
\documentclass{article}
\title{Test Paper}
\begin{document}
\section{Introduction}
This is a test paper about machine learning.
\section{Methods}
We used deep learning techniques.
\end{document}
"""
            latex_file.write_text(latex_content.strip())
            
            # Create test BibTeX file
            bibtex_file = tmp_path / "test.bib"
            bibtex_content = """
@article{test2023,
  title={Machine Learning Fundamentals},
  author={Test Author},
  journal={Test Journal},
  year={2023}
}
"""
            bibtex_file.write_text(bibtex_content.strip())
            
            print(f"✓ Test files created successfully")
            print(f"  - LaTeX file: {latex_file.name}")
            print(f"  - BibTeX file: {bibtex_file.name}")
            
            # Test file existence
            assert latex_file.exists()
            assert bibtex_file.exists()
            
            print("✓ File I/O works correctly")
            return True
            
    except Exception as e:
        print(f"✗ File creation test failed: {e}")
        return False


def test_analyzer_integration():
    """Test complete analyzer integration."""
    print("\nTesting analyzer integration...")
    
    try:
        from refscore.core.analyzer import RefScoreAnalyzer
        from refscore.utils.config import Config
        import tempfile
        from pathlib import Path
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create test LaTeX file
            latex_file = tmp_path / "test.tex"
            latex_content = r"""
\documentclass{article}
\title{Test Paper}
\begin{document}
\section{Introduction}
Machine learning algorithms are widely used in research.
Deep learning has revolutionized computer vision.
\section{Methods}
We employed neural networks with attention mechanisms.
The training process utilized optimization techniques.
\end{document}
"""
            latex_file.write_text(latex_content.strip())
            
            # Create test BibTeX file
            bibtex_file = tmp_path / "test.bib"
            bibtex_content = """
@article{ml2023,
  title={Machine Learning Algorithms},
  author={Smith, John},
  journal={ML Journal},
  year={2023},
  abstract={This paper discusses machine learning algorithms and their applications.}
}

@article{dl2022,
  title={Deep Learning for Computer Vision},
  author={Doe, Jane},
  journal={CV Journal},
  year={2022},
  abstract={Exploring deep learning techniques in computer vision tasks.}
}
"""
            bibtex_file.write_text(bibtex_content.strip())
            
            # Initialize analyzer
            config = Config()
            analyzer = RefScoreAnalyzer(config)
            
            # Load document and sources
            document = analyzer.load_document(latex_file)
            sources = analyzer.load_sources([bibtex_file])
            
            print(f"✓ Document loaded: {len(document.sentences)} sentences")
            print(f"✓ Sources loaded: {len(sources)} sources")
            
            # Compute scores
            scores = analyzer.compute_scores(document, sources)
            
            print(f"✓ Scores computed: {len(scores)} source scores")
            
            if scores:
                print(f"  - Top score: {scores[0].refscore:.4f}")
                print(f"  - Average score: {sum(s.refscore for s in scores) / len(scores):.4f}")
            
            # Generate coverage report
            coverage = analyzer.get_coverage_report(document, scores)
            
            print(f"✓ Coverage report generated")
            print(f"  - Overall coverage: {coverage.get('overall_coverage', 0):.3f}")
            
            return True
            
    except Exception as e:
        print(f"✗ Analyzer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RefScore Academic Application - Installation Test")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("Document Models", test_document_models),
        ("Source Models", test_source_models),
        ("Scoring Models", test_scoring_models),
        ("Basic Scoring", test_basic_scoring),
        ("File Creation", test_file_creation),
        ("Analyzer Integration", test_analyzer_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! RefScore is ready to use.")
        print("\nNext steps:")
        print("1. Run the GUI: python -m refscore.main")
        print("2. Run the CLI: python -m refscore.cli --help")
        print("3. Check sample files in: ~/Documents/RefScoreSamples/")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check the output above for details.")
        print("\nThe application may still work, but some features might be limited.")
        return 1


if __name__ == "__main__":
    sys.exit(main())