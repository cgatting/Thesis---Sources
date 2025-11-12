import unittest
import sys
import os
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# Stub heavy dependencies used in SourceRankingEngine
_st = types.ModuleType('sentence_transformers')
class _M:
    def __init__(self, name):
        pass
    def encode(self, text):
        import numpy as np
        return np.random.RandomState(abs(hash(text)) % (2**32)).rand(384)
_st.SentenceTransformer = _M
sys.modules.setdefault('sentence_transformers', _st)

_skp = types.ModuleType('sklearn.metrics.pairwise')
def _cos(a,b):
    import numpy as np
    return np.array([[0.5]])
_skp.cosine_similarity = _cos
sys.modules.setdefault('sklearn.metrics.pairwise', _skp)

from refscore.core.source_ranking import SourceRankingEngine

class TestRelevantSources(unittest.TestCase):
    def test_rank_returns_top_10(self):
        eng = SourceRankingEngine()
        # Monkeypatch search to avoid network
        def fake_search(query, rows):
            items = []
            for i in range(15):
                items.append({
                    'title': [f'Title {i}'],
                    'abstract': 'example abstract',
                    'container-title': ['Journal'],
                    'DOI': f'10.1/{i}',
                    'URL': f'https://example.com/{i}',
                    'author': [{'family': 'Doe', 'given': 'J'}],
                    'published-print': {'date-parts': [[2020]]}
                })
            return items
        eng.search = fake_search
        res = eng.rank('sample document text', rows=100, threshold=0.0, use_refine=True)
        self.assertEqual(len(res), 10)
        self.assertTrue(all('title' in r for r in res))

    def test_empty_text_returns_empty(self):
        eng = SourceRankingEngine()
        res = eng.rank('', rows=50, threshold=0.0, use_refine=False)
        self.assertEqual(res, [])

    def test_link_generation_prefers_url(self):
        eng = SourceRankingEngine()
        def fake_search(query, rows):
            return [{
                'title': ['X'], 'abstract': '', 'container-title': ['J'],
                'DOI': '10.1/x', 'URL': 'http://example.com/x'
            }]
        eng.search = fake_search
        res = eng.rank('doc', rows=1, threshold=0.0, use_refine=False)
        self.assertTrue(res[0]['link'].startswith('http'))

if __name__ == '__main__':
    unittest.main()

