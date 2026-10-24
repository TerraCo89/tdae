"""Tests for pluggable scoring: FreightScorer parity and a non-freight scorer end to end."""

import pytest

from agents.customer_matching import CustomerDirectory, CustomerMatchingAgent
from tdae.ground_truth import CUSTOMER_FIXTURES, GroundTruthTestData
from tdae.orchestrator import TDAEOrchestrator, print_summary
from tdae.providers import MockLLMProvider
from tdae.scoring import (
    FreightScorer,
    Scorer,
    TestResultValidator,
    fact_check,
    within_tolerance,
)


# ---------------------------------------------------------------------------
# FreightScorer (the default) is the v0.1.0 behaviour
# ---------------------------------------------------------------------------

def _healthy_agent():
    return CustomerMatchingAgent(
        provider=MockLLMProvider.for_ground_truth(),
        directory=CustomerDirectory.from_fixtures(CUSTOMER_FIXTURES),
    )


def test_freight_scorer_matches_validation_report():
    scenario = GroundTruthTestData.get_all_scenarios()[0]
    output = scenario["expected_output"]
    assert FreightScorer().score(scenario, output) == (
        TestResultValidator.generate_validation_report(scenario, output)
    )


def test_freight_scorer_is_the_default():
    orchestrator = TDAEOrchestrator()
    assert isinstance(orchestrator.scorer, FreightScorer)
    assert isinstance(orchestrator.scorer, Scorer)


def test_freight_scorer_threshold_only_changes_the_verdict():
    scenario = GroundTruthTestData.get_scenarios_by_type("CUSTOMER_MATCHING")[0]
    lenient = FreightScorer(pass_threshold=0.0).score(scenario, {})
    strict = FreightScorer().score(scenario, {})
    assert lenient["overall_score"] == strict["overall_score"]
    assert lenient["passed"] is True
    assert strict["passed"] is False


def test_explicit_freight_scorer_gives_same_suite_result():
    scenarios = GroundTruthTestData.get_scenarios_by_type("CUSTOMER_MATCHING")
    default = TDAEOrchestrator().run_suite(_healthy_agent(), scenarios)
    explicit = TDAEOrchestrator(scorer=FreightScorer()).run_suite(_healthy_agent(), scenarios)
    assert default["summary"]["pass_rate"] == explicit["summary"]["pass_rate"] == 1.0
    assert default["summary"]["average_score"] == explicit["summary"]["average_score"]


# ---------------------------------------------------------------------------
# A non-freight domain: a curve-fitting "agent" scored on fitted parameters
# ---------------------------------------------------------------------------

SCAN_SCENARIOS = [
    {
        "scenario_id": "RABI_001",
        "scenario_type": "RABI_SCAN",
        "difficulty": "easy",
        "expected_output": {"rabi_freq_khz": 85.0, "suspect": False},
    },
    {
        "scenario_id": "RABI_002",
        "scenario_type": "RABI_SCAN",
        "difficulty": "hard",
        "expected_output": {"rabi_freq_khz": 60.0, "suspect": True},
    },
]


class FitScorer:
    """Pass when every fitted parameter is within 5% and the suspect flag is right."""

    def score(self, scenario, agent_output):
        expected = scenario["expected_output"]
        verdicts = fact_check(
            expected, agent_output, tolerances={"rabi_freq_khz": {"rel_tol": 0.05}}
        )
        overall = sum(verdicts.values()) / len(verdicts)
        return {"overall_score": overall, "passed": all(verdicts.values()), "verdicts": verdicts}


class FakeFitAgent:
    name = "fit_agent"
    model = "deterministic"

    def __init__(self, fits):
        self.fits = fits

    def run(self, scenario):
        return dict(self.fits[scenario["scenario_id"]])


def test_custom_scorer_end_to_end():
    agent = FakeFitAgent({
        "RABI_001": {"rabi_freq_khz": 84.1, "suspect": False},   # within 5%
        "RABI_002": {"rabi_freq_khz": 66.0, "suspect": True},    # 10% off -> fails
    })
    orchestrator = TDAEOrchestrator(scorer=FitScorer(), suite="fits", run_id="r1")
    summary = orchestrator.run_suite(agent, SCAN_SCENARIOS)

    assert summary["summary"]["passed"] == 1
    assert summary["summary"]["pass_rate"] == 0.5
    assert summary["summary"]["average_score"] == pytest.approx(0.75)
    # Freight-only averages are absent for a non-freight scorer.
    assert summary["summary"]["average_field_accuracy"] is None
    assert summary["summary"]["average_price_accuracy"] is None
    assert summary["run_id"] == "r1"
    assert summary["suite"] == "fits"

    # Scorer detail survives into the per-scenario report.
    second = summary["results"][1]
    assert second["verdicts"] == {"rabi_freq_khz": False, "suspect": True}
    assert second["scenario_type"] == "RABI_SCAN"

    record = orchestrator.store.records()[0]
    assert (record.run_id, record.suite, record.model) == ("r1", "fits", "deterministic")
    assert within_tolerance(85.0, 84.1, rel_tol=0.05)


def test_custom_scorer_summary_prints_without_freight_lines(capsys):
    agent = FakeFitAgent({
        "RABI_001": {"rabi_freq_khz": 85.0, "suspect": False},
        "RABI_002": {"rabi_freq_khz": 60.0, "suspect": True},
    })
    summary = TDAEOrchestrator(scorer=FitScorer(), suite="fits").run_suite(agent, SCAN_SCENARIOS)
    print_summary(summary)
    out = capsys.readouterr().out
    assert "Suite:              fits" in out
    assert "Field Accuracy" not in out
    assert "Price Accuracy" not in out


def test_scenarios_without_type_can_be_filtered():
    scenarios = [{"scenario_id": "Q_001", "expected_output": {"x": 1}}]
    agent = FakeFitAgent({"Q_001": {"x": 1}})
    summary = TDAEOrchestrator(scorer=FitScorer()).run_suite(agent, scenarios, scenario_filter="q_")
    assert summary["summary"]["total_scenarios"] == 1


def test_scorer_missing_required_keys_fails_loudly():
    class BrokenScorer:
        def score(self, scenario, agent_output):
            return {"overall_score": 1.0}

    agent = FakeFitAgent({"RABI_001": {}})
    orchestrator = TDAEOrchestrator(scorer=BrokenScorer())
    with pytest.raises(ValueError, match="returned no passed"):
        orchestrator.run_scenario(agent, SCAN_SCENARIOS[0])
