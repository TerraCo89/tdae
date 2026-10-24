# Changelog

## 0.2.0 — 2026-10-24

Makes TDAE usable outside freight. Fully backwards compatible: existing code,
tests and `tdae-demo` output are unchanged, and there are still no runtime
dependencies.

### Added

- **Pluggable scoring.** `Scorer` protocol (`score(scenario, agent_output) -> dict`
  with `overall_score` and `passed`) and `TDAEOrchestrator(scorer=...)`.
  `FreightScorer` wraps the existing weighted validation report and stays the
  default; its `pass_threshold` defaults to 0.85.
- **Scoring helpers** for writing domain scorers: `within_tolerance`
  (relative/absolute tolerance), `fact_check` (dotted-path facts with
  per-fact tolerances and subset matching for lists) and `precision_recall`
  (one-to-one maximum matching with a custom `match`, e.g. time windows).
- **Run metadata and usage.** `TestExecutionRecord` gains optional `run_id`,
  `suite`, `model`, `input_tokens`, `output_tokens`, `cost_usd` and
  `metadata`, plus `to_dict` / `from_dict`. The orchestrator takes `run_id` and
  `suite`, reads an optional `_usage` key from agent outputs (stripped before
  scoring) and falls back to an agent's `model` attribute.
- **Usage in metrics and summaries.** `AgentPerformanceMetrics` gains token
  totals and total / average cost (also under `to_dict()["usage"]`); suite
  summaries total tokens and cost when agents report them.
- **Run-scoped analytics.** `records(...)` and `metrics_for(...)` filter by
  `run_id` and `suite`; `run_ids()` lists runs; `compare(...)` takes
  `run_id_a` / `run_id_b` to compare two runs of the same agent.
- **`JsonlResultStore(path)`**: an append-only JSON Lines ledger with the same
  interface as `InMemoryResultStore`. It reloads on open, keeps non-ASCII text
  readable, ignores unknown keys, and reports malformed lines with their line number.

### Changed

- Suite summaries report `average_field_accuracy` / `average_price_accuracy`
  as `None` (and `print_summary` omits them) when the scorer doesn't produce
  those freight fields. Freight results are unchanged.
- `print_summary` shows the suite, run ID and usage line when present.
- `run_suite(scenario_filter=...)` accepts scenarios without a `scenario_type`.

## 0.1.0 — 2026-07-10

Initial release: 36 synthetic freight scenarios, weighted validation scoring,
per-agent metrics, regression detection, mock/scripted LLM providers and an
offline demo.
