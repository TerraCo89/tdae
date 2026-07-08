"""TDAE orchestrator: the core evaluation loop.

Runs agents against ground-truth scenarios, scores each run with
:class:`~tdae.scoring.TestResultValidator`, records executions in a result
store, and produces suite summaries.

The production orchestrator dispatched JSON-RPC calls to agent MCP servers
over HTTP, persisted results to Postgres, and streamed batch progress via
Redis. The showcase keeps the same evaluation semantics but calls agents
in-process through the :class:`ScenarioAgent` protocol and stores results in
:class:`~tdae.metrics.InMemoryResultStore`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from tdae.metrics import InMemoryResultStore, TestExecutionRecord
from tdae.scoring import TestResultValidator


@runtime_checkable
class ScenarioAgent(Protocol):
    """Anything that can process a ground-truth scenario.

    ``run`` receives the full scenario dict (agents typically only read
    ``email_content``) and returns the agent's structured output, which is
    scored against ``scenario["expected_output"]``.
    """

    name: str

    def run(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        ...


class TDAEOrchestrator:
    """Coordinates scenario execution, scoring, and result recording."""

    def __init__(self, store: Optional[InMemoryResultStore] = None, verbose: bool = False):
        self.store = store or InMemoryResultStore()
        self.verbose = verbose

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[tdae] {message}")

    # -- single scenario ------------------------------------------------------

    def run_scenario(self, agent: ScenarioAgent, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one scenario, score it, and record the execution."""
        scenario_id = scenario["scenario_id"]
        start = time.perf_counter()

        try:
            agent_output = agent.run(scenario)
        except Exception as exc:  # agent crashed: record an error execution
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.store.add(
                TestExecutionRecord(
                    scenario_id=scenario_id,
                    agent_name=agent.name,
                    result="error",
                    accuracy_score=0.0,
                    execution_time_ms=elapsed_ms,
                    status="failed",
                    error_message=str(exc),
                )
            )
            self.log(f"{scenario_id}: ERROR ({exc})")
            return {
                "scenario_id": scenario_id,
                "scenario_type": scenario.get("scenario_type"),
                "difficulty": scenario.get("difficulty"),
                "passed": False,
                "error": True,
                "error_message": str(exc),
                "overall_score": 0.0,
                "execution_time_ms": elapsed_ms,
            }

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        report = TestResultValidator.generate_validation_report(scenario, agent_output)
        report["scenario_type"] = scenario.get("scenario_type")
        report["difficulty"] = scenario.get("difficulty")
        report["execution_time_ms"] = elapsed_ms

        self.store.add(
            TestExecutionRecord(
                scenario_id=scenario_id,
                agent_name=agent.name,
                result="pass" if report["passed"] else "fail",
                accuracy_score=report["overall_score"],
                execution_time_ms=elapsed_ms,
            )
        )

        self.log(
            f"{scenario_id}: score={report['overall_score']:.2%} "
            f"({'PASSED' if report['passed'] else 'FAILED'})"
        )
        return report

    # -- batch execution -------------------------------------------------------

    def run_suite(
        self,
        agent: ScenarioAgent,
        scenarios: List[Dict[str, Any]],
        scenario_filter: Optional[str] = None,
        stop_on_failure: bool = False,
    ) -> Dict[str, Any]:
        """Run a batch of scenarios and return a summary report.

        Args:
            agent: The agent under evaluation.
            scenarios: Ground-truth scenarios to run.
            scenario_filter: Optional case-insensitive filter matched against
                scenario type and ID (e.g. ``"FCL"`` or ``"CUSTOMER"``).
            stop_on_failure: Abort the batch on the first failure.
        """
        if scenario_filter:
            wanted = scenario_filter.upper()
            scenarios = [
                s
                for s in scenarios
                if wanted in s["scenario_type"].upper() or wanted in s["scenario_id"].upper()
            ]

        if not scenarios:
            return {"error": "No scenarios match the filter", "results": []}

        self.log(f"Running {len(scenarios)} scenarios with agent '{agent.name}'...")

        results: List[Dict[str, Any]] = []
        for scenario in scenarios:
            report = self.run_scenario(agent, scenario)
            results.append(report)
            if stop_on_failure and not report.get("passed", False):
                self.log("Stopping batch on first failure")
                break

        return self._summarize(agent.name, results)

    # -- reporting ---------------------------------------------------------------

    @staticmethod
    def _summarize(agent_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        errors = sum(1 for r in results if r.get("error", False))
        failed = total - passed

        valid = [r for r in results if not r.get("error")]
        avg_score = (
            sum(r.get("overall_score", 0.0) for r in valid) / len(valid) if valid else 0.0
        )
        avg_field_accuracy = (
            sum(r.get("field_accuracy", 0.0) for r in valid) / len(valid) if valid else 0.0
        )
        avg_price_accuracy = (
            sum(r.get("price_accuracy", 0.0) for r in valid) / len(valid) if valid else 0.0
        )

        by_difficulty: Dict[str, Dict[str, int]] = {}
        for result in results:
            difficulty = result.get("difficulty") or "unknown"
            bucket = by_difficulty.setdefault(difficulty, {"total": 0, "passed": 0})
            bucket["total"] += 1
            if result.get("passed"):
                bucket["passed"] += 1

        total_time_ms = sum(r.get("execution_time_ms", 0.0) for r in results)

        return {
            "agent_name": agent_name,
            "summary": {
                "total_scenarios": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": passed / total if total > 0 else 0.0,
                "average_score": avg_score,
                "average_field_accuracy": avg_field_accuracy,
                "average_price_accuracy": avg_price_accuracy,
                "total_execution_time_ms": total_time_ms,
            },
            "by_difficulty": by_difficulty,
            "results": results,
        }


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted suite summary to the console."""
    print("\n" + "=" * 60)
    print("TDAE GROUND TRUTH TEST SUMMARY")
    print("=" * 60)

    s = summary["summary"]
    print(f"\nAgent:              {summary.get('agent_name', '-')}")
    print(f"Total Scenarios:    {s['total_scenarios']}")
    print(f"Passed:             {s['passed']} ({s['pass_rate']:.1%})")
    print(f"Failed:             {s['failed']}")
    print(f"Errors:             {s['errors']}")
    print(f"\nAverage Score:      {s['average_score']:.2%}")
    print(f"Field Accuracy:     {s['average_field_accuracy']:.2%}")
    print(f"Price Accuracy:     {s['average_price_accuracy']:.2%}")
    print(f"Total Time:         {s['total_execution_time_ms']:.0f}ms")

    print("\nResults by Difficulty:")
    for difficulty, stats in summary.get("by_difficulty", {}).items():
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {difficulty:12s}: {stats['passed']}/{stats['total']} ({rate:.0%})")

    print("\nDetailed Results:")
    print("-" * 60)
    for result in summary.get("results", []):
        status = "PASS" if result.get("passed") else "FAIL"
        if result.get("error"):
            status = "ERR "
        score = result.get("overall_score", 0.0)
        print(f"  [{status}] {result['scenario_id']:30s} Score: {score:.2%}")

    print("=" * 60)
