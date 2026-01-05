import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch
import numpy as np
try:
    import sklearn.metrics.pairwise
except ImportError:
    pass

# Add parent directory to path to import refscore
ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from refscore.core.source_ranking import SourceRankingEngine

class TestSourceRankingEngine(unittest.TestCase):
    def setUp(self):
        # Patch SentenceTransformer to avoid loading real model
        self.st_patcher = patch('sentence_transformers.SentenceTransformer')
        self.mock_st_class = self.st_patcher.start()
        self.mock_model = MagicMock()
        self.mock_st_class.return_value = self.mock_model
        
        # Patch requests
        self.requests_patcher = patch('refscore.core.source_ranking.requests')
        self.mock_requests = self.requests_patcher.start()

    def tearDown(self):
        self.st_patcher.stop()
        self.requests_patcher.stop()

    def test_extract_terms(self):
        engine = SourceRankingEngine()
        # SourceRankingEngine.extract_terms uses regex r"[A-Za-z]{4,}"
        text = "Deep neural network learning algorithm"
        terms = engine.extract_terms(text)
        
        self.assertIn("neural", terms)
        self.assertIn("network", terms)
        self.assertIn("learning", terms)
        self.assertIn("algorithm", terms)
        self.assertIn("deep", terms) 
        
    def test_search_and_rank(self):
        engine = SourceRankingEngine()
        
        # Mock requests response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'message': {
                'items': [
                    {
                        'title': ['Paper A'],
                        'abstract': 'Abstract A',
                        'DOI': '10.1000/a',
                        'container-title': ['Journal A']
                    },
                    {
                        'title': ['Paper B'],
                        'abstract': 'Abstract B',
                        'DOI': '10.1000/b',
                        'container-title': ['Journal B']
                    }
                ]
            }
        }
        self.mock_requests.get.return_value = mock_response
        
        # Mock embedding behavior (just needs to return something iterable/numpy-like)
        self.mock_model.encode.return_value = np.array([1.0])
        
        # Mock _similarity to avoid sklearn issues and control ranking
        # We want Paper A (first item) to have higher score than Paper B (second item)
        # The rank loop calls _similarity for each item.
        engine._similarity = MagicMock(side_effect=[0.9, 0.5])
        
        results = engine.rank("doc text", rows=10)
        
        self.assertEqual(len(results), 2)
        # Results should be sorted by similarity (A > B)
        self.assertEqual(results[0]['title'], 'Paper A')
        self.assertEqual(results[1]['title'], 'Paper B')

    def test_handles_empty_input(self):
        engine = SourceRankingEngine()
        results = engine.rank("", rows=10)
        self.assertEqual(results, [])

    def test_caching(self):
        engine = SourceRankingEngine()
        
        # Mock requests response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'message': { 'items': [{'title': ['A'], 'container-title':['J'], 'DOI':'1'}] }
        }
        self.mock_requests.get.return_value = mock_response
        self.mock_model.encode.return_value = np.array([1.0])
        
        # Mock _similarity
        engine._similarity = MagicMock(return_value=0.9)
         
        # First call
        engine.rank("doc", rows=10)
        self.assertEqual(self.mock_requests.get.call_count, 1)
         
        # Second call (should be cached)
        engine.rank("doc", rows=10)
        self.assertEqual(self.mock_requests.get.call_count, 1)
