"""TDAE end-to-end demo (no API keys required).

Runs the customer-matching agent against the CUSTOMER_MATCHING ground-truth
scenarios using the deterministic MockLLMProvider, prints the suite summary
and the agent's performance metrics, and demonstrates regression detection
by re-running the suite with a deliberately broken configuration (an empty
customer directory).

Run it with:

    tdae-demo            # console script installed by pip
    python -m tdae.demo  # equivalent
"""

from __future__ import annotations

import sys

from agents.customer_matching import CustomerDirectory, CustomerMatchingAgent
from tdae.ground_truth import CUSTOMER_FIXTURES, GroundTruthTestData
from tdae.metrics import detect_regressions
from tdae.orchestrator import TDAEOrchestrator, print_summary
from tdae.providers import MockLLMProvider


def main() -> int:
    print("TDAE Demo - Test-Driven Agent Evolution")
    print("Provider: MockLLMProvider (deterministic, offline, no API keys)")

    scenarios = GroundTruthTestData.get_scenarios_by_type("CUSTOMER_MATCHING")
    orchestrator = TDAEOrchestrator(verbose=False)

    # --- Baseline run: healthy agent -------------------------------------
    provider = MockLLMProvider.for_ground_truth()
    directory = CustomerDirectory.from_fixtures(CUSTOMER_FIXTURES)
    agent = CustomerMatchingAgent(provider=provider, directory=directory)

    summary = orchestrator.run_suite(agent, scenarios)
    print_summary(summary)

    baseline_metrics = orchestrator.store.metrics_for(agent.name)
    print("\nAgent performance metrics (baseline):")
    print(f"  pass_rate:            {baseline_metrics.pass_rate:.2%}")
    print(f"  avg_accuracy:         {baseline_metrics.avg_accuracy:.2%}")
    print(f"  avg_response_time_ms: {baseline_metrics.avg_response_time_ms:.1f}")
    print(f"  error_rate:           {baseline_metrics.error_rate:.2%}")

    # --- Regression demo: broken configuration ----------------------------
    print("\nRe-running suite with a broken configuration (empty customer")
    print("directory) to demonstrate regression detection...")

    broken_orchestrator = TDAEOrchestrator(verbose=False)
    broken_agent = CustomerMatchingAgent(
        provider=MockLLMProvider.for_ground_truth(),
        directory=CustomerDirectory(),  # nobody home
    )
    broken_agent_name = "customer_matching_broken"
    broken_agent.name = broken_agent_name  # type: ignore[misc]
    broken_summary = broken_orchestrator.run_suite(broken_agent, scenarios)

    broken_metrics = broken_orchestrator.store.metrics_for(broken_agent_name)
    regressions = detect_regressions(baseline_metrics, broken_metrics)

    print(
        f"  broken config pass rate: "
        f"{broken_summary['summary']['pass_rate']:.0%} "
        f"(baseline was {summary['summary']['pass_rate']:.0%})"
    )
    if regressions:
        print("  Regressions detected:")
        for regression in regressions:
            print(f"    - {regression}")
    else:
        print("  No regressions detected (unexpected!)")

    # The demo succeeds when the baseline passes AND the regression is caught.
    ok = summary["summary"]["pass_rate"] == 1.0 and len(regressions) > 0
    print(f"\nDemo result: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
