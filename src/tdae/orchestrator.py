"""TDAE orchestrator: the core evaluation loop.

Runs agents against ground-truth scenarios, scores each run with a
:class:`~tdae.scoring.Scorer` (by default the freight
:class:`~tdae.scoring.FreightScorer`), records executions in a result store,
and produces suite summaries.

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
from tdae.scoring import FreightScorer, Scorer

# Optional key an agent output may carry to report model usage. It is
# stripped before scoring and recorded on the execution record.
USAGE_KEY = "_usage"
_USAGE_FIELDS = ("model", "input_tokens", "output_tokens", "cost_usd")


@runtime_checkable
class ScenarioAgent(Protocol):
    """Anything that can process a ground-truth scenario.

    ``run`` receives the full scenario dict (agents typically only read
    ``email_content``) and returns the agent's structured output, which is
    scored against ``scenario["expected_output"]``.

    To report model usage, include ``"_usage": {"model": ..., "input_tokens":
    ..., "output_tokens": ..., "cost_usd": ...}`` in the output (any other
    keys land in the record's ``metadata``). An optional ``model`` attribute
    on the agent is used when the output does not name one.
    """

    name: str

    def run(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        ...


class TDAEOrchestrator:
    """Coordinates scenario execution, scoring, and result recording."""

    def __init__(
        self,
        store: Optional[InMemoryResultStore] = None,
        verbose: bool = False,
        scorer: Optional[Scorer] = None,
        run_id: Optional[str] = None,
        suite: Optional[str] = None,
    ):
        """
        Args:
            store: Where execution records go (default: in memory).
            verbose: Log each scenario result.
            scorer: How outputs are scored (default: :class:`FreightScorer`).
            run_id: Tags every record, so runs can be told apart in a ledger.
            suite: Tags every record with the suite name.
        """
        self.store = store if store is not None else InMemoryResultStore()
        self.verbose = verbose
        self.scorer: Scorer = scorer if scorer is not None else FreightScorer()
        self.run_id = run_id
        self.suite = suite

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
                    run_id=self.run_id,
                    suite=self.suite,
                    model=getattr(agent, "model", None),
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

        usage: Dict[str, Any] = {}
        if isinstance(agent_output, dict) and USAGE_KEY in agent_output:
            agent_output = dict(agent_output)
            usage = dict(agent_output.pop(USAGE_KEY) or {})
        usage.setdefault("model", getattr(agent, "model", None))

        report = self.scorer.score(scenario, agent_output)
        missing = [key for key in ("overall_score", "passed") if key not in report]
        if missing:
            raise ValueError(
                f"{type(self.scorer).__name__}.score() for {scenario_id} "
                f"returned no {', '.join(missing)}"
            )
        report.setdefault("scenario_id", scenario_id)
        report["scenario_type"] = scenario.get("scenario_type")
        report["difficulty"] = scenario.get("difficulty")
        report["execution_time_ms"] = elapsed_ms
        if any(usage.get(key) is not None for key in _USAGE_FIELDS[1:]):
            report["usage"] = {key: usage.get(key) for key in _USAGE_FIELDS}

        self.store.add(
            TestExecutionRecord(
                scenario_id=scenario_id,
                agent_name=agent.name,
                result="pass" if report["passed"] else "fail",
                accuracy_score=report["overall_score"],
                execution_time_ms=elapsed_ms,
                run_id=self.run_id,
                suite=self.suite,
                model=usage.get("model"),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cost_usd=usage.get("cost_usd"),
                metadata={k: v for k, v in usage.items() if k not in _USAGE_FIELDS},
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
                if wanted in s.get("scenario_type", "").upper()
                or wanted in s["scenario_id"].upper()
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

        summary = self._summarize(agent.name, results)
        if self.run_id is not None:
            summary["run_id"] = self.run_id
        if self.suite is not None:
            summary["suite"] = self.suite
        return summary

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
        # Freight-report averages; None when the scorer doesn't produce them.
        avg_field_accuracy = TDAEOrchestrator._average_of(valid, "field_accuracy")
        avg_price_accuracy = TDAEOrchestrator._average_of(valid, "price_accuracy")

        by_difficulty: Dict[str, Dict[str, int]] = {}
        for result in results:
            difficulty = result.get("difficulty") or "unknown"
            bucket = by_difficulty.setdefault(difficulty, {"total": 0, "passed": 0})
            bucket["total"] += 1
            if result.get("passed"):
                bucket["passed"] += 1

        total_time_ms = sum(r.get("execution_time_ms", 0.0) for r in results)

        summary: Dict[str, Any] = {
            "total_scenarios": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0.0,
            "average_score": avg_score,
            "average_field_accuracy": avg_field_accuracy,
            "average_price_accuracy": avg_price_accuracy,
            "total_execution_time_ms": total_time_ms,
        }

        usages = [r["usage"] for r in results if "usage" in r]
        if usages:
            summary["total_input_tokens"] = sum(u.get("input_tokens") or 0 for u in usages)
            summary["total_output_tokens"] = sum(u.get("output_tokens") or 0 for u in usages)
            summary["total_cost_usd"] = sum(u.get("cost_usd") or 0.0 for u in usages)

        return {
            "agent_name": agent_name,
            "summary": summary,
            "by_difficulty": by_difficulty,
            "results": results,
        }

    @staticmethod
    def _average_of(results: List[Dict[str, Any]], key: str) -> Optional[float]:
        """Mean of ``key`` over results; 0.0 for none, None if no result has it."""
        if not results:
            return 0.0
        values = [r[key] for r in results if key in r]
        return sum(values) / len(values) if values else None


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted suite summary to the console."""
    print("\n" + "=" * 60)
    print("TDAE GROUND TRUTH TEST SUMMARY")
    print("=" * 60)

    s = summary["summary"]
    print(f"\nAgent:              {summary.get('agent_name', '-')}")
    if summary.get("suite"):
        print(f"Suite:              {summary['suite']}")
    if summary.get("run_id"):
        print(f"Run:                {summary['run_id']}")
    print(f"Total Scenarios:    {s['total_scenarios']}")
    print(f"Passed:             {s['passed']} ({s['pass_rate']:.1%})")
    print(f"Failed:             {s['failed']}")
    print(f"Errors:             {s['errors']}")
    print(f"\nAverage Score:      {s['average_score']:.2%}")
    if s.get("average_field_accuracy") is not None:
        print(f"Field Accuracy:     {s['average_field_accuracy']:.2%}")
    if s.get("average_price_accuracy") is not None:
        print(f"Price Accuracy:     {s['average_price_accuracy']:.2%}")
    print(f"Total Time:         {s['total_execution_time_ms']:.0f}ms")
    if "total_cost_usd" in s:
        print(
            f"Usage:              {s['total_input_tokens']} in / "
            f"{s['total_output_tokens']} out tokens, ${s['total_cost_usd']:.4f}"
        )

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
