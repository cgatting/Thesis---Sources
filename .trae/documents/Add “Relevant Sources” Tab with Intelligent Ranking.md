## Overview
- Add a new tab in `DeepResearchToolApp` to analyze an input document and surface the top 10 most relevant sources from programmatic search results.
- Reuse existing components: `NLPProcessor` (keywords, embeddings), `ResearchSearchEngine` (CrossRef), `DocumentCache`, async event loop, and global progress/console.
- Implement a lightweight `SourceRankingEngine` inside `DEEPSEARCH.py` to orchestrate term extraction, query building, searching, scoring, and ranking.

## UI Additions
- Extend `create_notebook_frame` (the later definition using `self.tabview`) to add a tab `"Relevant Sources"`.
- In `create_sources_tab()`:
  - Add `CTkTextbox` `self.sources_input_text` for the document content.
  - Add search parameter controls bound to existing settings:
    - `CTkComboBox` for `rows` (e.g., 50, 100, 200, 500) reading from `settings['search_settings']['max_results']`.
    - `CTkSlider` or `CTkEntry` for similarity threshold, default `settings['similarity_threshold']`.
    - `CTkSwitch` for “use refined query” (leverages `NLPProcessor.refine_query`).
  - Add `CTkButton` `self.sources_run_button` to execute.
  - Show progress via existing global `self.progress_bar` and `self.status_label`; also show tab-local small `CTkProgressBar` if desired.
  - Display results in a table using `ttk.Treeview` embedded in the tab with columns: `Title`, `Authors`, `Year`, `Journal`, `DOI`, `Score`, `Summary`, `Link`.
  - Bind double-click on a row to open the `Link` in the default browser (`webbrowser.open`).

## Processing Logic
- Add class `SourceRankingEngine` (in `DEEPSEARCH.py`) that:
  - `extract_key_terms(text)` using `yake.KeywordExtractor` (already configured) + simple NLTK token heuristics for concepts.
  - `build_query(text, use_refine)` → if `use_refine`, call `NLPProcessor.refine_query(text)`; otherwise join top keywords.
  - `search(query, rows)` → call `ResearchSearchEngine.search_papers(query, limit=rows)`, handling rate limits and empty responses.
  - `score_result(doc_text, item)` → compute similarity between `doc_text` and `combined_text = title + abstract (if present)` using `NLPProcessor.calculate_similarity`.
  - `summarize_item(item)` → generate a brief summary from `title/abstract` using `pipeline` summarizer, restricted to top 10 to preserve performance.
  - `rank_top_sources(doc_text, params)` → end-to-end pipeline that returns a list of dicts with the fields required for the table.
  - Use `DocumentCache` with key `hash(doc_text + json.dumps(params, sort_keys=True))` to cache results.

## App Integration
- In `DeepResearchToolApp`:
  - Initialize `self.source_ranking = SourceRankingEngine(self.settings, self.doc_refiner.nlp_processor, self.search_engine)` in `__init__` after existing components.
  - Add handlers:
    - `start_source_ranking()` → reads input + params, disables run button, logs, schedules `run_source_ranking_async()` on `self.async_loop` with `asyncio.run_coroutine_threadsafe`.
    - `run_source_ranking_async()` → async coroutine performing ranking, updates progress during stages (extract, search, score, summarize, render), handles errors, re-enables UI.
  - Add `update_sources_results(results)` → fills the `ttk.Treeview`, formats scores, stores raw results for future actions.

## Ranking Details
- Similarity scoring: use `NLPProcessor.calculate_similarity` on full text vs `title + abstract` to keep behavior consistent with existing citation logic.
- Secondary signals (if present): boost items that include overlapping authors or keywords; penalize missing `DOI`/`container-title` (already filtered in refiner).
- Final score: weighted combination (e.g., `0.85 * similarity + 0.15 * keyword_overlap`), normalized to 0–1; sort descending and take top 10.

## Result Display
- Populate `ttk.Treeview` with:
  - `Title` (`item['title'][0]`), `Authors` (joined family+given), `Year`, `Journal` (`container-title[0]`), `DOI`, `Score` (rounded), `Summary` (short), `Link` (`item.get('URL')` or DOI URL).
- On double-click, open `Link` in browser.
- Provide an “Export JSON” button (optional) to save ranked results.

## Error Handling & Feedback
- Wrap each stage in try/except; call `self.show_error(...)` and `self.log_to_console(...)` on failures.
- Respect `MAX_DOCUMENT_SIZE` and show a friendly message for oversized input.
- Handle empty input, no results, or rate-limit responses; show status text accordingly.
- Avoid blocking UI: all heavy work runs on the existing async loop.

## Performance & Architecture
- Reuse existing models; only summarize the final top 10.
- Use caching to prevent repeated work on unchanged input/params.
- Leverage GPU if available via existing `NLPProcessor` device handling.
- Follow current style (class-based, methods on `DeepResearchToolApp`, use CustomTkinter + `ttk.Treeview`).

## Testing
- Add unit tests (Python `unittest`) in `tests/test_source_ranking.py`:
  - `test_extract_key_terms` verifies keyword extraction stability on sample text.
  - `test_rank_top_sources_orders_by_similarity` with mocked results to ensure correct sorting.
  - `test_handles_empty_input_and_no_results` ensures graceful behavior.
  - `test_caching_works` confirms cached retrieval reduces calls.
  - `test_link_generation_and_summary_truncation` checks display fields.
- Add lightweight mocks for `ResearchSearchEngine.search_papers` and `NLPProcessor.calculate_similarity` to avoid network/model in tests.

## Acceptance Criteria
- New tab visible and functional; runs end-to-end without blocking UI.
- Displays top 10 ranked sources with accurate scores and metadata; links open.
- Handles error cases with clear feedback; no regressions in existing features.
- Tests pass locally; ranking logic verified with mocks.

## Rollout Steps
1. Implement `SourceRankingEngine` class.
2. Add UI: `create_sources_tab()` and integrate into `create_notebook_frame`.
3. Wire handlers and async execution.
4. Render results and linking.
5. Add tests and mocks; run test suite.
6. Manual verification with a sample `.tex` or `.txt` document.

Please confirm, and I’ll implement the tab, engine, integration, and tests accordingly.