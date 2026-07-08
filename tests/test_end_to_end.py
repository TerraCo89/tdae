"""End-to-end tests: customer-matching agent + mock provider + orchestrator."""

import pytest

from agents.customer_matching import CustomerDirectory, CustomerMatchingAgent
from tdae.ground_truth import CUSTOMER_FIXTURES, GroundTruthTestData
from tdae.metrics import detect_regressions
from tdae.orchestrator import TDAEOrchestrator
from tdae.providers import MockLLMProvider, ScriptedLLMProvider


@pytest.fixture
def scenarios():
    return GroundTruthTestData.get_scenarios_by_type("CUSTOMER_MATCHING")


@pytest.fixture
def healthy_agent():
    return CustomerMatchingAgent(
        provider=MockLLMProvider.for_ground_truth(),
        directory=CustomerDirectory.from_fixtures(CUSTOMER_FIXTURES),
    )


def test_full_suite_passes_with_mock_provider(scenarios, healthy_agent):
    orchestrator = TDAEOrchestrator()
    summary = orchestrator.run_suite(healthy_agent, scenarios)

    assert summary["summary"]["total_scenarios"] == 6
    assert summary["summary"]["passed"] == 6
    assert summary["summary"]["pass_rate"] == 1.0
    assert summary["summary"]["errors"] == 0

    metrics = orchestrator.store.metrics_for("customer_matching")
    assert metrics.total_tests == 6
    assert metrics.pass_rate == 1.0
    assert metrics.error_rate == 0.0
    assert metrics.avg_accuracy == pytest.approx(1.0)


def test_exact_match_scenario_uses_exact_strategy(healthy_agent):
    scenario = GroundTruthTestData.get_scenario_by_id("CUST_MATCH_002")
    output = healthy_agent.run(scenario)
    assert output["customer_match"]["match_method"] == "exact_match"
    assert output["customer_match"]["confidence_score"] == 1.0


def test_domain_match_scenario_uses_domain_strategy(healthy_agent):
    scenario = GroundTruthTestData.get_scenario_by_id("CUST_MATCH_001")
    output = healthy_agent.run(scenario)
    assert output["customer_match"]["match_method"] == "domain_match"
    assert output["customer_match"]["customer_id"] == "CUST-2024-0042"


def test_broken_directory_fails_and_regression_is_detected(scenarios, healthy_agent):
    # Baseline run with the healthy agent.
    baseline_orchestrator = TDAEOrchestrator()
    baseline_orchestrator.run_suite(healthy_agent, scenarios)
    baseline = baseline_orchestrator.store.metrics_for("customer_matching")

    # Candidate run with an empty customer directory: everything should fail.
    broken_agent = CustomerMatchingAgent(
        provider=MockLLMProvider.for_ground_truth(),
        directory=CustomerDirectory(),
    )
    candidate_orchestrator = TDAEOrchestrator()
    summary = candidate_orchestrator.run_suite(broken_agent, scenarios)
    assert summary["summary"]["pass_rate"] == 0.0

    candidate = candidate_orchestrator.store.metrics_for("customer_matching")
    regressions = detect_regressions(baseline, candidate)
    assert regressions, "regression in pass rate/accuracy should be detected"


def test_agent_crash_is_recorded_as_error(scenarios):
    class ExplodingAgent:
        name = "exploder"

        def run(self, scenario):
            raise RuntimeError("boom")

    orchestrator = TDAEOrchestrator()
    summary = orchestrator.run_suite(ExplodingAgent(), scenarios[:2])
    assert summary["summary"]["errors"] == 2
    metrics = orchestrator.store.metrics_for("exploder")
    assert metrics.error_rate == 1.0


def test_scenario_filter(scenarios, healthy_agent):
    orchestrator = TDAEOrchestrator()
    summary = orchestrator.run_suite(
        healthy_agent,
        GroundTruthTestData.get_all_scenarios(),
        scenario_filter="CUST_MATCH_001",
    )
    assert summary["summary"]["total_scenarios"] == 1


def test_regex_fallback_when_llm_finds_nothing():
    """With a provider that always answers NOT_FOUND, the regex fallback fires."""
    agent = CustomerMatchingAgent(
        provider=ScriptedLLMProvider(script=["NOT_FOUND"]),
        directory=CustomerDirectory.from_fixtures(CUSTOMER_FIXTURES),
    )
    info = agent.extract_company_info(
        "Hello,\n\nPlease quote Shanghai to LA.\n\nRegards,\nJane Doe\n"
        "TechCorp International Ltd\n",
        sender_email="jane@techcorp-intl.example",
    )
    assert info["extraction_method"] == "regex"
    assert "TechCorp International Ltd" in info["company_name"]


def test_demo_entry_point_returns_zero(capsys):
    from tdae.demo import main

    assert main() == 0
    captured = capsys.readouterr()
    assert "Regressions detected" in captured.out
