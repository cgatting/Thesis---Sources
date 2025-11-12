import unittest
from types import SimpleNamespace

# Load SourceRankingEngine from DEEPSEARCH.py
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# Inject lightweight stubs for optional heavy dependencies to allow import
class _StubYake:
    class KeywordExtractor:
        def __init__(self, **kwargs):
            pass
        def extract_keywords(self, text):
            return [("stub", 0.1)]

sys.modules.setdefault('yake', _StubYake())

class _StubST:
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, device):
            return self
        def encode(self, text, convert_to_tensor=True):
            class _T:
                def cpu(self):
                    return self
                def numpy(self):
                    import numpy as np
                    return np.zeros((384,), dtype=float)
                def reshape(self, *a, **k):
                    return self
            return _T()

sys.modules.setdefault('sentence_transformers', _StubST())

class _StubTF:
    def pipeline(*args, **kwargs):
        def _run(x, **k):
            return [{"summary_text": str(x)[:60]}]
        return _run

sys.modules.setdefault('transformers', _StubTF())

class _StubTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False
    class multiprocessing:
        @staticmethod
        def set_start_method(*a, **k):
            pass
    class _Dev:
        def __init__(self, *a, **k):
            pass
    def device(*a, **k):
        return _StubTorch._Dev()
    class _NoGrad:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False
    def no_grad():
        return _StubTorch._NoGrad()

sys.modules.setdefault('torch', _StubTorch())

import types
_sk = types.ModuleType('sklearn')
_sk_metrics = types.ModuleType('sklearn.metrics')
_sk_pairwise = types.ModuleType('sklearn.metrics.pairwise')
def _cosine_similarity(a, b):
    import numpy as np
    return np.array([[0.5]])
_sk_pairwise.cosine_similarity = _cosine_similarity
sys.modules.setdefault('sklearn', _sk)
sys.modules.setdefault('sklearn.metrics', _sk_metrics)
sys.modules.setdefault('sklearn.metrics.pairwise', _sk_pairwise)

from DEEPSEARCH import SourceRankingEngine


class FakeKeywordExtractor:
    def extract_keywords(self, text):
        return [("neural", 0.1), ("network", 0.2), ("learning", 0.3), ("deep", 0.4)]


class FakeNLP:
    def __init__(self):
        self.keyword_extractor = FakeKeywordExtractor()
        self._scores = {}

    def refine_query(self, sentence: str) -> str:
        return "refined query"

    def calculate_similarity(self, text1: str, text2: str) -> float:
        return self._scores.get(text2, 0.5)

    def set_similarity(self, combined_text: str, score: float):
        self._scores[combined_text] = score

    def summarizer(self, text, max_length=32, min_length=8, do_sample=False):
        return [{"summary_text": text[:60]}]


class FakeSearchEngine:
    def __init__(self, items):
        self._items = items

    def search_papers(self, query: str, limit: int = 50):
        return self._items[:limit]


class FakeCache:
    def __init__(self):
        self._store = {}

    def get_cached_result(self, key: str):
        return self._store.get(key)

    def cache_result(self, key: str, data: dict):
        self._store[key] = data


class TestSourceRankingEngine(unittest.TestCase):
    def setUp(self):
        self.settings = {
            'model_settings': {'max_length': 20, 'min_length': 2},
            'search_settings': {'max_results': 50},
            'similarity_threshold': 0.7,
        }
        self.nlp = FakeNLP()

    def test_extract_key_terms(self):
        engine = SourceRankingEngine(self.settings, self.nlp, FakeSearchEngine([]), FakeCache())
        terms = engine.extract_key_terms("Deep neural network learning")
        self.assertIn("neural", terms)
        self.assertIn("network", terms)

    def test_rank_top_sources_orders_by_similarity(self):
        items = [
            {
                'title': ['Paper A'], 'abstract': 'alpha', 'DOI': '10.1/a',
                'container-title': ['Journal'], 'URL': 'http://example.com/a',
                'author': [{'family': 'Smith', 'given': 'J'}],
                'published-print': {'date-parts': [[2020]]}
            },
            {
                'title': ['Paper B'], 'abstract': 'beta', 'DOI': '10.1/b',
                'container-title': ['Journal'], 'URL': 'http://example.com/b',
                'author': [{'family': 'Lee', 'given': 'K'}],
                'published-print': {'date-parts': [[2021]]}
            }
        ]
        search = FakeSearchEngine(items)
        engine = SourceRankingEngine(self.settings, self.nlp, search, FakeCache())
        self.nlp.set_similarity("Paper A alpha", 0.9)
        self.nlp.set_similarity("Paper B beta", 0.6)
        results = engine.rank_top_sources("doc", rows=50, threshold=0.0, use_refine=False)
        self.assertEqual(results[0]['title'], 'Paper A')
        self.assertGreaterEqual(results[0]['score'], results[1]['score'])

    def test_handles_empty_input_and_no_results(self):
        search = FakeSearchEngine([])
        engine = SourceRankingEngine(self.settings, self.nlp, search, FakeCache())
        results = engine.rank_top_sources("", rows=10, threshold=0.0, use_refine=True)
        self.assertEqual(results, [])

    def test_caching_works(self):
        items = [{
            'title': ['Cached'], 'abstract': 'text', 'DOI': '10.1/x',
            'container-title': ['J'], 'URL': 'http://example.com/x',
        }]
        cache = FakeCache()
        search = FakeSearchEngine(items)
        engine = SourceRankingEngine(self.settings, self.nlp, search, cache)
        r1 = engine.rank_top_sources("doc", rows=10, threshold=0.0, use_refine=False)
        # mutate search to prove caching returns same
        search._items = []
        r2 = engine.rank_top_sources("doc", rows=10, threshold=0.0, use_refine=False)
        self.assertEqual(r1, r2)

    def test_link_generation_and_summary_truncation(self):
        items = [{
            'title': ['Link Only'], 'abstract': '', 'DOI': '10.1/doi',
            'container-title': ['J'],
        }]
        search = FakeSearchEngine(items)
        engine = SourceRankingEngine(self.settings, self.nlp, search, FakeCache())
        self.nlp.set_similarity("Link Only ", 0.8)
        res = engine.rank_top_sources("doc", rows=10, threshold=0.0, use_refine=False)
        self.assertTrue(res[0]['link'].startswith('https://doi.org/'))
        self.assertIsInstance(res[0]['summary'], str)


if __name__ == '__main__':
    unittest.main()
