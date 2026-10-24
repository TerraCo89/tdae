"""Tests for usage reporting, run metadata, run-scoped metrics and the JSONL ledger."""

import json

import pytest

from tdae.metrics import (
    InMemoryResultStore,
    JsonlResultStore,
    TestExecutionRecord,
    detect_regressions,
)
from tdae.orchestrator import TDAEOrchestrator, print_summary


class PassScorer:
    def score(self, scenario, agent_output):
        ok = agent_output.get("answer") == scenario["expected_output"]["answer"]
        return {"overall_score": 1.0 if ok else 0.0, "passed": ok}


class UsageAgent:
    """Answers correctly and reports token usage, like an LLM-backed agent would."""

    name = "ask_agent"

    def __init__(self, model="claude-test", wrong=(), cost=0.002):
        self.model = model
        self.wrong = set(wrong)
        self.cost = cost

    def run(self, scenario):
        answer = "wrong" if scenario["scenario_id"] in self.wrong else "right"
        return {
            "answer": answer,
            "_usage": {
                "input_tokens": 1200,
                "output_tokens": 300,
                "cost_usd": self.cost,
                "tool_calls": 3,                    # unknown key -> metadata
            },
        }


SCENARIOS = [
    {"scenario_id": f"ASK_{i:03d}", "scenario_type": "ASK", "expected_output": {"answer": "right"}}
    for i in range(1, 5)
]


def _record(agent="a", run_id=None, result="pass", accuracy=1.0, cost=None):
    return TestExecutionRecord(
        scenario_id="S1",
        agent_name=agent,
        result=result,
        accuracy_score=accuracy,
        execution_time_ms=10.0,
        run_id=run_id,
        cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# Usage reporting through the orchestrator
# ---------------------------------------------------------------------------

def test_usage_is_recorded_and_stripped_before_scoring():
    seen = []

    class SpyScorer(PassScorer):
        def score(self, scenario, agent_output):
            seen.append(dict(agent_output))
            return super().score(scenario, agent_output)

    orchestrator = TDAEOrchestrator(scorer=SpyScorer(), run_id="r1", suite="ask")
    report = orchestrator.run_scenario(UsageAgent(), SCENARIOS[0])

    assert "_usage" not in seen[0]
    assert report["usage"] == {
        "model": "claude-test",                    # taken from the agent attribute
        "input_tokens": 1200,
        "output_tokens": 300,
        "cost_usd": 0.002,
    }
    record = orchestrator.store.records()[0]
    assert record.model == "claude-test"
    assert (record.input_tokens, record.output_tokens, record.cost_usd) == (1200, 300, 0.002)
    assert record.metadata == {"tool_calls": 3}


def test_model_in_usage_overrides_agent_attribute():
    class NamedModelAgent(UsageAgent):
        def run(self, scenario):
            output = super().run(scenario)
            output["_usage"]["model"] = "ollama:qwen27b-64k"
            return output

    orchestrator = TDAEOrchestrator(scorer=PassScorer())
    orchestrator.run_scenario(NamedModelAgent(), SCENARIOS[0])
    assert orchestrator.store.records()[0].model == "ollama:qwen27b-64k"


def test_suite_summary_totals_usage(capsys):
    orchestrator = TDAEOrchestrator(scorer=PassScorer(), run_id="r1")
    summary = orchestrator.run_suite(UsageAgent(), SCENARIOS)
    s = summary["summary"]
    assert s["total_input_tokens"] == 4 * 1200
    assert s["total_output_tokens"] == 4 * 300
    assert s["total_cost_usd"] == pytest.approx(4 * 0.002)

    print_summary(summary)
    assert "4800 in / 1200 out tokens, $0.0080" in capsys.readouterr().out

    metrics = orchestrator.store.metrics_for("ask_agent")
    assert metrics.total_cost_usd == pytest.approx(0.008)
    assert metrics.avg_cost_usd == pytest.approx(0.002)
    assert metrics.to_dict()["usage"]["total_input_tokens"] == 4800


def test_agent_error_records_run_metadata():
    class CrashingAgent:
        name = "crashy"
        model = "claude-test"

        def run(self, scenario):
            raise RuntimeError("tool timed out")

    orchestrator = TDAEOrchestrator(scorer=PassScorer(), run_id="r9", suite="ask")
    report = orchestrator.run_scenario(CrashingAgent(), SCENARIOS[0])
    assert report["error"] is True
    record = orchestrator.store.records()[0]
    assert (record.result, record.run_id, record.suite, record.model) == (
        "error", "r9", "ask", "claude-test",
    )


# ---------------------------------------------------------------------------
# Run-scoped metrics, comparison and regressions
# ---------------------------------------------------------------------------

def test_metrics_for_filters_by_run_id():
    store = InMemoryResultStore()
    store.add(_record(run_id="base", result="pass", accuracy=1.0, cost=0.01))
    store.add(_record(run_id="base", result="pass", accuracy=1.0, cost=0.03))
    store.add(_record(run_id="cand", result="fail", accuracy=0.2))

    assert store.run_ids() == ["base", "cand"]
    base = store.metrics_for("a", run_id="base")
    assert base.total_tests == 2
    assert base.pass_rate == 1.0
    assert base.avg_cost_usd == pytest.approx(0.02)
    assert store.metrics_for("a", run_id="cand").pass_rate == 0.0
    assert store.metrics_for("a").total_tests == 3        # no filter: every run
    assert store.metrics_for("a", run_id="cand").total_cost_usd == 0.0  # no usage reported


def test_compare_two_runs_of_the_same_agent():
    store = InMemoryResultStore()
    for _ in range(3):
        store.add(_record(run_id="base", result="pass", accuracy=0.95))
        store.add(_record(run_id="cand", result="fail", accuracy=0.40))

    comparison = store.compare("a", "a", run_id_a="base", run_id_b="cand")
    assert set(comparison["agents"]) == {"a@base", "a@cand"}
    assert comparison["comparison"]["accuracy"]["a@base"] == pytest.approx(0.95)
    assert "a@base" in " ".join(comparison["recommendations"])


def test_regression_gate_against_a_stored_baseline(tmp_path):
    ledger = tmp_path / "runs.jsonl"
    TDAEOrchestrator(store=JsonlResultStore(ledger), scorer=PassScorer(), run_id="base").run_suite(
        UsageAgent(), SCENARIOS
    )

    # A later process: reopen the ledger, run a worse candidate, gate on the baseline.
    store = JsonlResultStore(ledger)
    TDAEOrchestrator(store=store, scorer=PassScorer(), run_id="cand").run_suite(
        UsageAgent(wrong={"ASK_001", "ASK_002"}), SCENARIOS
    )
    regressions = detect_regressions(
        store.metrics_for("ask_agent", run_id="base"),
        store.metrics_for("ask_agent", run_id="cand"),
    )
    assert any("pass rate dropped" in r for r in regressions)


# ---------------------------------------------------------------------------
# JsonlResultStore
# ---------------------------------------------------------------------------

def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "nested" / "runs.jsonl"     # parent folder is created
    store = JsonlResultStore(path)
    original = TestExecutionRecord(
        scenario_id="ASK_JA_001",
        agent_name="ask_agent",
        result="pass",
        accuracy_score=0.9,
        execution_time_ms=1234.5,
        run_id="r1",
        suite="ask",
        model="claude-test",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        metadata={"question": "昨日のラビ周波数はなぜ変わった？"},
    )
    store.add(original)

    line = path.read_text(encoding="utf-8").strip()
    assert "ラビ周波数" in line                     # non-ASCII stays readable
    assert json.loads(line)["run_id"] == "r1"

    reloaded = JsonlResultStore(path)
    assert len(reloaded) == 1
    assert reloaded.records()[0] == original


def test_jsonl_appends_across_instances_and_skips_blank_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    JsonlResultStore(path).add(_record(run_id="r1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    JsonlResultStore(path).add(_record(run_id="r2"))

    store = JsonlResultStore(path)
    assert [r.run_id for r in store.records()] == ["r1", "r2"]


def test_jsonl_ignores_unknown_keys(tmp_path):
    path = tmp_path / "runs.jsonl"
    data = _record(run_id="r1").to_dict()
    data["added_in_a_future_version"] = True
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    assert JsonlResultStore(path).records()[0].run_id == "r1"


def test_jsonl_reports_bad_line_with_its_number(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(_record().to_dict()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"runs\.jsonl:2"):
        JsonlResultStore(path)


def test_old_style_records_still_work():
    # v0.1.0-style construction: no run metadata at all.
    record = TestExecutionRecord(
        scenario_id="S1", agent_name="a", result="pass", accuracy_score=1.0, execution_time_ms=5.0
    )
    assert record.run_id is None and record.metadata == {}
    assert TestExecutionRecord.from_dict(record.to_dict()) == record
