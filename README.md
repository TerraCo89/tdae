# TDAE — Test-Driven Agent Evolution

![CI](https://github.com/TerraCo89/tdae/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC.svg)
![Dependencies: none](https://img.shields.io/badge/runtime%20deps-zero-brightgreen.svg)

**TDAE is a ground-truth evaluation harness for LLM agents.** You maintain a
suite of scenarios with known-correct outputs; TDAE runs your agents against
them, scores each run with tolerance-aware accuracy metrics, aggregates
per-agent performance (pass rate, accuracy, latency, error rate), and fails
loudly when a prompt, model, or config change regresses the agent.

LLM agents don't have unit tests in the classical sense — outputs are fuzzy,
partially correct, and non-binary. TDAE's answer:

- **Ground-truth suites** — 36 synthetic freight-forwarding scenarios
  (FCL/LCL quotes, air freight with dangerous goods, multimodal routing,
  customer matching, booking confirmations), each with an expected output and
  per-scenario validation metrics (required fields, price tolerance, critical
  validations).
- **Accuracy scoring** — a weighted validation report per run: field accuracy
  (30%), price accuracy within tolerance (30%), output completeness (20%),
  and critical-field validations (20%). A run passes at ≥ 85% overall.
- **Per-agent metrics** — pass rate, average accuracy, latency statistics,
  and error rate, aggregated from every recorded execution.
- **Regression tracking** — compare agent configurations head-to-head and
  diff a current run against a stored baseline; wire `detect_regressions`
  into CI to block degrading changes.

## Architecture

```mermaid
graph TB
    subgraph "Ground Truth"
        GT[ground_truth.py<br/>36 synthetic scenarios<br/>+ ValidationMetrics]
    end

    subgraph "Evaluation Loop"
        ORCH[TDAEOrchestrator<br/>batch runner]
        SCORE[scoring.py<br/>field / price / completeness /<br/>critical validations]
        STORE[InMemoryResultStore<br/>execution records]
        METRICS[AgentPerformanceMetrics<br/>pass rate · accuracy · latency · errors]
    end

    subgraph "Agents Under Test"
        AGENT[CustomerMatchingAgent<br/>LLM extraction + layered matching]
        PROV[LLMProvider protocol]
        MOCK[MockLLMProvider<br/>deterministic, offline]
        REAL[Real provider<br/>Anthropic / OpenAI / local]
    end

    GT -->|scenario| ORCH
    ORCH -->|email input| AGENT
    AGENT -->|structured output| ORCH
    ORCH --> SCORE
    SCORE -->|validation report| STORE
    STORE --> METRICS
    METRICS -->|detect_regressions| CI[CI gate /<br/>refinement loop]
    AGENT --> PROV
    PROV -.-> MOCK
    PROV -.-> REAL
```

The refinement cycle this enables: **run suite → score → analyze gaps →
adjust agent config/prompt → re-run → compare against baseline → deploy or
roll back.**

## Quick start

No API keys needed — the demo and the full test suite run offline against a
deterministic `MockLLMProvider`.

```bash
git clone https://github.com/TerraCo89/tdae && cd tdae
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"

pytest        # 38 tests, all offline, < 1 second
tdae-demo     # end-to-end demo: baseline run + regression detection
```

The demo runs the customer-matching agent over its 6 ground-truth scenarios
(100% pass), then re-runs with a deliberately broken configuration and shows
`detect_regressions` catching the drop in pass rate and accuracy:

```text
TDAE GROUND TRUTH TEST SUMMARY
==============================
Agent:              customer_matching
Total Scenarios:    6
Passed:             6 (100.0%)
Average Score:      100.00%    (field 100% · price 100%)

Agent performance metrics (baseline):
  pass_rate: 100.00%   avg_accuracy: 100.00%   error_rate: 0.00%

Re-running suite with a broken configuration (empty customer
directory) to demonstrate regression detection...
  broken config pass rate: 0% (baseline was 100%)
  Regressions detected:
    - accuracy dropped 30.0% (1.00 -> 0.70)
    - pass rate dropped 100.0% (1.00 -> 0.00)

Demo result: OK
```

## Using the harness

```python
from tdae import GroundTruthTestData, TDAEOrchestrator, MockLLMProvider, detect_regressions
from agents.customer_matching import CustomerMatchingAgent, CustomerDirectory
from tdae.ground_truth import CUSTOMER_FIXTURES

agent = CustomerMatchingAgent(
    provider=MockLLMProvider.for_ground_truth(),   # swap for a real provider in prod
    directory=CustomerDirectory.from_fixtures(CUSTOMER_FIXTURES),
)

orchestrator = TDAEOrchestrator(verbose=True)
summary = orchestrator.run_suite(
    agent,
    GroundTruthTestData.get_scenarios_by_type("CUSTOMER_MATCHING"),
)
print(summary["summary"])            # pass rate, avg score, per-difficulty breakdown

metrics = orchestrator.store.metrics_for(agent.name)
print(metrics.to_dict())             # pass_rate, avg_accuracy, latency stats, error_rate
```

**Evaluate your own agent** by implementing the two-member `ScenarioAgent`
protocol — a `name` attribute and `run(scenario) -> dict` — and returning
output shaped like the scenario's `expected_output`. **Add scenarios** with
`GroundTruthTestData.register_scenario({...})` or by extending the variant
tables in `src/tdae/ground_truth.py`.

**Gate CI on regressions:**

```python
regressions = detect_regressions(baseline_metrics, current_metrics)
assert not regressions, f"Agent regressed: {regressions}"
```

## Repository layout

```
src/tdae/                    # the harness
  ground_truth.py            # 36 synthetic scenarios + ValidationMetrics
  scoring.py                 # accuracy scoring + weighted validation reports
  metrics.py                 # result store, per-agent metrics, regression detection
  orchestrator.py            # batch evaluation loop
  providers.py               # LLMProvider protocol + Mock/Scripted providers
  demo.py                    # offline end-to-end demo (tdae-demo)
src/agents/customer_matching # example agent under evaluation
tests/                       # pytest suite (offline, deterministic)
```

## Provenance and scope

Extracted from a production multi-agent system built for a freight-forwarding
company (2024–2026), where TDAE evaluated customer-matching, quoting, and
email-processing agents. **All scenarios in this repository are synthetic** —
every company, contact, email address, rate, vessel, and booking reference is
fabricated (customer domains use the reserved `.example` TLD; scenario lookup
addresses use `@example-freight.test`). A test in the suite
(`test_no_production_identifiers_in_scenario_data`) enforces this.

To keep the showcase runnable with `pip install -e . && pytest`, the
production integrations were intentionally replaced:

| Production | Showcase |
|---|---|
| Postgres result tables | `InMemoryResultStore` (same analytical surface) |
| Redis batch-progress channel | not needed for in-process runs |
| Agents as MCP servers called over HTTP (JSON-RPC) | in-process `ScenarioAgent` protocol |
| Vendor LLM SDK (async) | `LLMProvider` protocol + `MockLLMProvider` (sync) |
| LangGraph orchestrator test runner with JWT auth | `TDAEOrchestrator.run_suite` |
| Auto-refinement / config rollback tools | out of scope (the metrics and comparison APIs they build on are included) |

## License

MIT © 2024–2026 Kris Cernjavic
