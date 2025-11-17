import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from refscore.core.scoring import ScoringEngine
from refscore.core.document_parser import DocumentParser
from refscore.utils.config import Config

cfg = Config()
cfg.set_scoring_weights({
    'alignment': 0.7,
    'entities': 0.1,
    'number_unit': 0.1,
    'method_metric': 0.05,
    'recency': 0.03,
    'authority': 0.02,
})
proc = cfg.get_processing_config()
proc['min_sentence_length'] = 4
proc['numeric_rel_tol'] = 0.15
cfg.settings.processing = proc

engine = ScoringEngine(cfg)
print('WEIGHTS', json.dumps(engine.weights, sort_keys=True))

parser = DocumentParser(cfg)
content = 'Introduction\nOne two three four. Short. This is a longer sentence that exceeds threshold.'
sections = [(0, 'Intro'), (len(content), '__END__')]
sentences, ordered_sections = parser._parse_sections(content, sections)
print('SENTENCE_COUNT', len(sentences))
print('SENTENCES', [s.text for s in sentences])

score, reasons = engine._number_unit_match('accuracy 95%', 'we achieved 94% accuracy')
print('NUM_UNIT_SCORE', score, reasons)