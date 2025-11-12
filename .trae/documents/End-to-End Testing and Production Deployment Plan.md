## Scope and Context

* Application type: Python desktop GUI (PyQt5) with CLI; core modules: `refscore/core/analyzer.py`, `refscore/core/scoring.py`, `refscore/core/source_loader.py`, `refscore/core/document_parser.py`, models in `refscore/models`, GUI in `refscore/gui`.

* External integrations: optional Crossref API (HTTP), spaCy model `en_core_web_sm`, PDF parsing via `PyMuPDF`.

* No built-in database layer; “database operations” are simulated via persistent config and file I/O.

## Unit Testing

* Goal: Verify functions/classes in isolation with ≥90% coverage; include edge cases.

* Targets:

  * `RefScoreAnalyzer` orchestration (refscore/core/analyzer.py:41–55, 131–160).

  * `ScoringEngine` methods: `_alignment_score`, `_entity_overlap`, `_number_unit_match`, `_weighted_score` (refscore/core/scoring.py:208–217, 219–241, 277–286, 319–319+).

  * `SourceLoader` format loaders and Crossref parsing (refscore/core/source\_loader.py:57–91, 92–127, 226–267, 469–601).

  * Models validation/serialization: `RefEvidence`, `SourceScore` (refscore/models/scoring.py:38–46, 111–118, 152–171).

  * Utilities: configuration, validators, exceptions (refscore/utils/\*).

* Approach:

  * Use `pytest` + `pytest-cov`, target `--cov=refscore --cov-report=term-missing --cov-report=html`.

  * Edge cases: empty inputs, extreme weights (zeros/ones), malformed BibTeX/JSON/CSV, large numbers/units, missing spaCy, missing transformers, Crossref timeouts.

  * Determinism: mock external calls (`requests` via `responses/requests-mock`), mock heavy ML paths when needed.

  * Property-based tests for tokenization and Jaccard calculations.

## Integration Testing

* Goal: Validate component interactions, data flows, and external API behavior.

* Scenarios:

  * Analyzer end-to-end from document load → sources load → scoring → reports (refscore/core/analyzer.py:161–227).

  * Crossref API integration: DOI and title lookups with success/failure paths (refscore/core/source\_loader.py:469–531).

  * CLI pipelines: run typical commands and validate outputs (`refscore/cli.py`).

* Data Flows:

  * Validate object contracts between parsers → models → scoring (e.g., `DocPack.to_document()` → `SourceScore.to_dict()`).

* Techniques:

  * Spin up tests with temporary directories and files; stub network via mock; verify generated report files.

## System Testing

* Goal: Full application validation against user stories and requirements.

* Coverage:

  * GUI workflows: open document, load sources, adjust weights, run analysis, save outputs (refscore/gui/main\_window\.py).

  * CLI workflows: batch process documents, export JSON/CSV.

  * Config persistence and reset flows (refscore/utils/config.py).

* Environments:

  * Windows baseline (Python 3.13); optionally validate macOS/Linux if in scope.

* Load Conditions:

  * Large LaTeX/PDF with ≥10k sentences; sources lists of ≥1k entries; verify UI responsiveness and background worker stability.

## User Acceptance Testing (UAT)

* Goal: Real-user sessions validating business requirements.

* Plan:

  * Define test scripts per user story: document parsing accuracy, ranking quality, evidence explanations, export formats.

  * Facilitate sessions with prebuilt datasets; capture feedback via forms and logs.

  * Acceptance criteria: all critical workflows succeed; output correctness validated by domain reviewers.

## Performance Testing

* Goal: Assess throughput, memory, and responsiveness under expected peak loads.

* Metrics:

  * Scoring throughput (sources/sec), total analysis time, CPU/memory footprint.

  * GUI latency: progress updates, tab switching, result rendering.

* Method:

  * Synthetic large datasets; repeat runs to warm caches; isolate GPU/CPU variances (transformers disabled vs enabled).

  * Record timings with Python `time/perf_counter`; aggregate into CSV; plot with `matplotlib`.

* Targets:

  * Establish baseline SLAs (e.g., 1k sources × 2k sentences ≤ X minutes without critical memory pressure).

## Security Testing

* Goal: Identify vulnerabilities and verify auth/data protections.

* Scope:

  * Dependency audits: `pip-audit`, `safety`; static analysis: `bandit`.

  * Input validation: path traversal, arbitrary code in BibTeX/JSON/CSV, PDF robustness.

  * Network security: Crossref calls – HTTPS only, timeouts, exception handling, no sensitive data logged.

* Tests:

  * Fuzz inputs for loaders; ensure exceptions are raised safely (`ProcessingError`, `ValidationError`).

  * Verify configuration files do not leak secrets; sanitize logs.

## Tooling & Automation

* CI pipeline:

  * Run `pytest` with coverage gate at 90% (fail if below).

  * Lint/type checks: `flake8`, `black --check`, `mypy`.

  * Security checks: `pip-audit`, `bandit`.

* Reports:

  * Publish `htmlcov`, `junitxml` (`pytest --junitxml=reports/junit.xml`), performance CSV/plots, security audit outputs.

## Defect Management

* Severity classification; “zero critical issues” required before deploy.

* Triaging workflow; retest and regression suites.

## Deployment Plan

* Package Preparation:

  * Pin runtime deps; include spaCy model as optional post-install step.

  * Build artifacts: PyInstaller executable for Windows and `sdist/wheel` via `python -m build`.

* Production Environment:

  * Define supported OS versions, Python runtime (if using wheel), file permissions for output directories.

* Monitoring & Alerting:

  * Structured logging to rotating files; optional crash reporting (e.g., Sentry) if acceptable.

  * Health indicators: CLI exit codes, GUI error dialogs surfaced with context (`refscore/utils/exceptions.py:1–164, 202–322`).

* Phased Rollout:

  * Internal pilot → limited external group → full release; maintain rollback-ready previous version.

* Rollback Plan:

  * Versioned installers; documented reversion steps; artifacts retained.

## Documentation & Evidence

* Test Artefacts:

  * Unit/integration/system test cases, datasets, expected results.

  * Coverage reports, performance graphs, security audit logs.

* Deployment Procedures:

  * Step-by-step install and configuration; rollout/rollback instructions.

* UAT Records:

  * Session notes, defects found, acceptance sign-offs.

## Milestones & Exit Criteria

* Complete each phase with documented results and no critical defects.

* Coverage ≥90%; performance targets met; security audits with no high/critical findings.

* Final sign-off by stakeholders and release owner.

## Next Actions (upon approval)

* Set up CI with coverage/lint/type/security gates.

* Author missing tests to reach coverage targets, focusing on edge cases and integration paths.

* Prepare packaging scripts and draft rollout checklist.

