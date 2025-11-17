## Why Scores Are Low

* Alignment uses embeddings if available; otherwise TF‑IDF, else Jaccard. On minimal libraries/metadata, Jaccard produces low scores (e.g., \~0.10). See `refscore/core/scoring.py:219-247`.

* Source text for scoring is `title + abstract`. If abstracts are missing, alignment and entities underperform. See `refscore/core/scoring.py:182-191`.

* NER and number/unit matching degrade if spaCy and quantulum3 aren’t installed; regex fallbacks are weaker. See `refscore/core/scoring.py:293-331` and `refscore/core/scoring.py:356-395`.

## Quick Wins (No Code Changes)

* Increase alignment weight via CLI to emphasize semantic match: `--weights alignment=0.7 entities=0.1 number_unit=0.1 method_metric=0.05 recency=0.03 authority=0.02` (cli supports `--weights`; see `refscore/cli.py:121-153`).

* Provide richer sources: add Zotero JSON exports (with abstracts) and comprehensive `.bib` files. Loader supports `.bib, .json, .csv, .txt` (DOI lists). See `refscore/core/source_loader.py:71-75` and `refscore/utils/validators.py:36-39`.

* Feed a `.txt` DOI list to fetch metadata via Crossref when `requests` is present. See `refscore/core/source_loader.py:428-463`.

## Environment Enhancements

* Install libraries to unlock stronger scoring:

  * `sentence-transformers` for semantic alignment (MiniLM) `refscore/core/scoring.py:97-104`.

  * `scikit-learn` for TF‑IDF fallback `refscore/core/scoring.py:102-107`.

  * `spacy` + `en_core_web_sm` for NER `refscore/core/scoring.py:113-121`.

  * `quantulum3` for numeric/units `refscore/core/scoring.py:126-131`.

  * `requests` for Crossref `refscore/core/source_loader.py:48-55`.

* After install, verify logs show: “Loaded SentenceTransformer…”, “Loaded spaCy…”, “Loaded quantulum3…”. CLI logging is configured; see `refscore/cli.py:20-33`.

## Data Enrichment

* Ensure abstracts are present in `.bib` or Zotero JSON. BibTeX parsing respects `abstract` field; see `refscore/core/source_loader.py:162-171`.

* Use Crossref by DOI or title to backfill missing metadata; see `refscore/core/source_loader.py:464-491` and `refscore/core/source_loader.py:493-526`.

* Optionally extend to fetch publisher page text via `URL` from Crossref to enrich `source_text` (kept optional for compliance). Currently link is built in ranking; see `refscore/core/source_ranking.py:123-170`.

## Scoring Tuning (Code Changes After Approval)

* Lower sentence length threshold from 6→4 words to include more document sentences: `refscore/core/document_parser.py:264`.

* Relax numeric relative tolerance `rel_tol` from 0.07→0.15 for more number/unit matches: `refscore/core/scoring.py:325`.

* Expand `UNITS` list with domain units and `METHOD_TOKENS` with domain terminology to improve overlaps: `refscore/core/scoring.py:59-67`, `refscore/core/scoring.py:50-56`.

* Make weights configurable via `Config` so GUI remembers custom weights; wire `ScoringEngine` to read them (currently it copies defaults; see `refscore/core/scoring.py:82-87`).

## More Results

* Add more source files in GUI Sources tab or via CLI `--sources` (multiple paths supported). Deduplication preserves unique entries by DOI/title; see `refscore/core/analyzer.py:127-131` and `refscore/core/analyzer.py:244-255`.

## Verification

* Run analysis with verbose logging to confirm improved libraries and weights; see `refscore/cli.py:100-118` and `refscore/cli.py:176-183`.

* Check top scores printed by CLI `refscore/cli.py:190-193` and review “Relevant Sources” UI which shows authors/year/journal/DOI/score/link `refscore/gui/widgets/relevant_sources_tab.py:255-257`.

* Review weakest sentences and coverage reports to ensure section support rises: `refscore/core/scoring.py:491-536` and `refscore/core/scoring.py:444-489`.

## Next Steps

1. Apply CLI weight tuning and add richer sources (no code changes).
2. Install recommended libraries to unlock embeddings/NER/numeric parsing.
3. If needed, implement the small code adjustments (thresholds, tokens, units, config-driven weights).
4. Re‑run and compare top refscore and coverage; iterate on domain tokens/units as needed.

