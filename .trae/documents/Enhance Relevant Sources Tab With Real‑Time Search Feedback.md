## UI Additions
- Add `QProgressBar` (search progress) and `QLabel`s for: phase status, processed count/total, and ETA.
- Add `QPushButton` “Cancel” to stop an in‑progress search.
- Add lightweight loading animation via a `QTimer` that animates status text (ellipsis) while running; use indeterminate `QProgressBar` (`setRange(0,0)`) during non‑quantifiable phases.
- Keep visual style consistent with existing tabs: group boxes, header stretch, accessible names/tooltips.

## Localization
- Use `QCoreApplication.translate("RelevantSourcesTab", text)` for all user‑visible status strings.
- Provide a small helper `tr(key)` in the tab that maps phase identifiers to translated strings: "Indexing documents", "Analyzing content", "Matching results", "Completed", "Cancelled", and error messages.

## Worker Refactor (Non‑Blocking)
- Create `SearchProgressWorker(QThread)` to perform incremental ranking with real‑time updates:
  - Phase 1 Indexing: call `engine.extract_terms` and `engine.build_query`; emit status and set progress to ~10%.
  - Phase 2 Searching: call `engine.search(query, rows)`; emit status and indeterminate progress until results list is ready.
  - Phase 3 Analyzing: compute `doc_vec = engine._embed(doc_text)`; emit status.
  - Phase 4 Matching: iterate over `items` to score each using `engine._embed` and `engine._similarity`, applying threshold; after each item:
    - Emit `progress(processed/total * phase_weight + base)`
    - Emit processed count (`processed`, `total`) and updated ETA computed from average per‑item time since the start of matching.
    - Check `self.cancelled` to support cancellation and abort early.
  - On completion: emit `completed(results)` and set progress to 100%, status “Completed”.
  - On error: emit `failed(message)` with localized text.

## Tab Wiring
- Replace current direct use of `SourceSearchWorker` with `SearchProgressWorker` in `refscore/gui/widgets/relevant_sources_tab.py`.
- Keep existing parameters (rows, threshold, refine) and table population; reuse `populate_table`.
- Add new labels: `self.phase_label`, `self.count_label`, `self.eta_label`; update them via worker signals.
- Add `self.cancel_btn` to set `worker.cancelled = True` and disable itself during cancellation; re‑enable Run when idle.
- Maintain responsiveness: all heavy work runs inside `QThread`; UI updated via signals only.

## Accurate Progress Strategy
- Phase weights: Indexing 10%, Searching 20% (indeterminate until item count known), Analyzing 10%, Matching 60% (computed per item).
- If `items` < 1: emit graceful message and complete early.
- Progress bar switches between indeterminate during unknown durations and determinate when totals are known.

## Error Handling
- Gracefully handle network errors from CrossRef, empty content, and embed failures.
- Show `QMessageBox` for fatal errors; update status text and reset progress bar/state.
- Ensure cancellation produces “Cancelled” state and resets UI controls.

## Accessibility & Consistency
- Add accessible names to new controls (`progress-bar`, `phase-label`, `eta-label`, `cancel-button`).
- Keyboard mnemonics for Run and Cancel; tooltips for indicators.
- Follow existing group box structure and header stretch on tables.

## Files to Update
- `refscore/gui/widgets/relevant_sources_tab.py`: UI additions, new worker class, signal wiring, localization helper, cancellation control.
- No changes needed to core engine API (we’ll reuse existing `SourceRankingEngine` internals inside the worker).

## Validation Plan
- Manual run via `python -m refscore.main`; trigger a search and observe:
  - Progress bar percentages and phase text change in real‑time.
  - Count shows processed/total; ETA updates; spinner animates.
  - Cancel button stops matching promptly and restores UI.
  - Completion shows “Completed” with progress at 100%.
- Add a lightweight unit test for `SearchProgressWorker` that simulates items with a fake engine to verify progress and cancellation signals.

Please confirm and I’ll implement these changes.