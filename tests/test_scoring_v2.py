import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import math
from refscore.core.scoring import ScoringEngine
from refscore.utils.config import Config

class TestScoringV2:
    
    @pytest.fixture
    def engine(self):
        config = Config()
        # Initialize engine but mock heavy components if needed
        # We will override them in tests anyway
        engine = ScoringEngine(config)
        return engine

    def test_weighted_method_matching(self, engine):
        # "random forest" is a phrase in METHOD_TOKENS (assuming it is, need to verify or mock)
        # "svm" is a single token
        
        # Force METHOD_TOKENS for this test to be sure
        engine.METHOD_TOKENS = {"random forest", "svm"}
        
        # Case 1: Single token match
        # svm match -> 0.5 points
        score_svm = engine._method_metric_overlap("We used svm here", "svm is great")
        # 0.5 / 6.0 = 0.0833
        assert score_svm == pytest.approx(0.5 / 6.0)

        # Case 2: Phrase match
        # random forest match -> 2.0 points
        score_rf = engine._method_metric_overlap("We used random forest", "random forest is used")
        # 2.0 / 6.0 = 0.333
        assert score_rf == pytest.approx(2.0 / 6.0)
        
        # Case 3: Both match
        # svm + random forest -> 2.5 points
        score_both = engine._method_metric_overlap("svm and random forest", "random forest and svm")
        # 2.5 / 6.0 = 0.4166
        assert score_both == pytest.approx(2.5 / 6.0)
        
        # Case 4: Partial match (forest) should not trigger phrase match
        # But 'forest' is not in our mock tokens, so 0
        score_partial = engine._method_metric_overlap("forest", "forest")
        assert score_partial == 0.0

    def test_max_sim_alignment(self, engine):
        # Mock embedder
        mock_embedder = MagicMock()
        engine.embedder = mock_embedder
        
        # Mock _split_into_sentences to ensure we have 2 sentences
        engine._split_into_sentences = MagicMock(return_value=["Sentence 1", "Sentence 2"])
        
        # Mock embeddings
        # doc_sent = "A"
        # source_sents = ["Sentence 1", "Sentence 2"]
        # embedding(A) = [1, 0]
        # embedding(S1) = [0, 1] (sim=0)
        # embedding(S2) = [0.9, 0.1] (sim=0.9 approx if normalized? No, dot product)
        # 1*0.9 + 0*0.1 = 0.9
        
        def encode_side_effect(text, **kwargs):
            if isinstance(text, str):
                return np.array([1.0, 0.0])
            elif isinstance(text, list):
                # source sentences
                return np.array([[0.0, 1.0], [0.9, 0.1]])
            return np.array([0.0, 0.0])
            
        mock_embedder.encode.side_effect = encode_side_effect
        
        # Mock cross encoder to be None to test just Max-Sim first
        engine.cross_encoder = None
        
        score = engine._alignment_score("A", "B. C.")
        
        # Should pick max sim which is [1,0] dot [0.9, 0.1] = 0.9
        assert score == pytest.approx(0.9)

    def test_cross_encoder_reranking(self, engine):
        # Mock embedder to return high sim > 0.25
        mock_embedder = MagicMock()
        engine.embedder = mock_embedder
        
        # Doc: [1, 0]
        # Source: [0.9, 0.1] -> Sim 0.9 -> Triggers re-ranking
        mock_embedder.encode.side_effect = lambda text, **kwargs: \
            np.array([1.0, 0.0]) if isinstance(text, str) else np.array([[0.9, 0.1]])
            
        # Mock Cross Encoder
        mock_cross = MagicMock()
        engine.cross_encoder = mock_cross
        engine.rerank_threshold = 0.25
        
        # Mock predict output (logits)
        # Logit 2.0 -> Sigmoid(2.0) ~= 0.88
        mock_cross.predict.return_value = np.array([2.0])
        
        score = engine._alignment_score("Doc", "Source")
        
        # Check if cross encoder was called
        mock_cross.predict.assert_called_once()
        
        # Check score is sigmoid(2.0)
        expected = 1 / (1 + math.exp(-2.0))
        assert score == pytest.approx(expected)

    def test_cross_encoder_not_triggered(self, engine):
        # Mock embedder to return low sim < 0.25
        mock_embedder = MagicMock()
        engine.embedder = mock_embedder
        
        # Doc: [1, 0]
        # Source: [0.0, 1.0] -> Sim 0.0
        mock_embedder.encode.side_effect = lambda text, **kwargs: \
            np.array([1.0, 0.0]) if isinstance(text, str) else np.array([[0.0, 1.0]])
            
        # Mock Cross Encoder
        mock_cross = MagicMock()
        engine.cross_encoder = mock_cross
        engine.rerank_threshold = 0.25
        
        score = engine._alignment_score("Doc", "Source")
        
        # Check if cross encoder was NOT called
        mock_cross.predict.assert_not_called()
        
        assert score == 0.0
