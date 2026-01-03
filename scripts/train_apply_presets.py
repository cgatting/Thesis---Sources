import json
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from refscore.training.weights_trainer import WeightsTrainer

doc = os.environ.get('REFSCORE_DOC', '')
sources = os.environ.get('REFSCORE_SOURCES', '')
if not doc or not sources:
    print('SET REFSCORE_DOC and REFSCORE_SOURCES env vars')
    raise SystemExit(1)

jobs = [(doc, sources.split(','))]
wt = WeightsTrainer()
best_w, metrics = wt.train(jobs, {
    'alignment': (0.5, 0.8, 0.1),
    'entities': (0.05, 0.2, 0.05),
    'number_unit': (0.05, 0.2, 0.05),
    'method_metric': (0.05, 0.15, 0.05),
    'recency': (0.02, 0.1, 0.02),
    'authority': (0.02, 0.1, 0.02),
})
print('BEST', json.dumps(best_w))
print('METRICS', json.dumps(metrics))
pid = wt.save_preset('auto', best_w, metrics, {'grid': 'default'})
print('SAVED', pid)
assert wt.apply_preset(pid)
print('APPLIED', pid)
print('LIST', json.dumps(wt.list_presets(), indent=2))
prev = wt.rollback()
print('ROLLED_BACK_TO', prev)