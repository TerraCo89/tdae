"""Per-agent performance metrics, result storage, and regression tracking.

The production system persisted execution records to Postgres tables and
broadcast batch progress over Redis. The showcase replaces both with an
:class:`InMemoryResultStore` exposing the same analytical surface:

* per-agent :class:`AgentPerformanceMetrics` (pass rate, accuracy, latency,
  error rate)
* configuration / agent comparison with recommendations
* regression detection against a baseline
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TestExecutionRecord:
    """One scenario execution by one agent (or agent configuration)."""

    __test__ = False  # not a pytest test class, despite the name

    scenario_id: str
    agent_name: str
    result: str  # "pass" | "fail" | "error"
    accuracy_score: float
    execution_time_ms: float
    status: str = "completed"  # "completed" | "failed"
    error_message: Optional[str] = None
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    executed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class AgentPerformanceMetrics:
    """Aggregated performance metrics for one agent."""

    agent_name: str
    total_tests: int
    pass_rate: float
    avg_accuracy: float
    avg_response_time_ms: float
    error_rate: float
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    stddev_response_time_ms: float = 0.0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_tests": self.total_tests,
            "pass_rate": self.pass_rate,
            "avg_accuracy": self.avg_accuracy,
            "avg_response_time_ms": self.avg_response_time_ms,
            "error_rate": self.error_rate,
            "response_time_stats": {
                "min": self.min_response_time_ms,
                "max": self.max_response_time_ms,
                "stddev": self.stddev_response_time_ms,
            },
            "last_updated": self.last_updated,
        }


class InMemoryResultStore:
    """In-memory replacement for the production TDAE result tables."""

    def __init__(self) -> None:
        self._records: List[TestExecutionRecord] = []

    # -- write -------------------------------------------------------------

    def add(self, record: TestExecutionRecord) -> None:
        self._records.append(record)

    # -- read --------------------------------------------------------------

    def records(
        self,
        agent_name: Optional[str] = None,
        scenario_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[TestExecutionRecord]:
        results = [
            r
            for r in self._records
            if (agent_name is None or r.agent_name == agent_name)
            and (scenario_id is None or r.scenario_id == scenario_id)
        ]
        if limit is not None:
            results = results[-limit:]
        return results

    def __len__(self) -> int:
        return len(self._records)

    # -- analytics ----------------------------------------------------------

    def metrics_for(self, agent_name: str) -> AgentPerformanceMetrics:
        """Compute aggregated performance metrics for one agent."""
        records = self.records(agent_name=agent_name)
        total = len(records)
        if total == 0:
            return AgentPerformanceMetrics(
                agent_name=agent_name,
                total_tests=0,
                pass_rate=0.0,
                avg_accuracy=0.0,
                avg_response_time_ms=0.0,
                error_rate=0.0,
            )

        passed = sum(1 for r in records if r.result == "pass")
        errors = sum(1 for r in records if r.result == "error")
        latencies = [r.execution_time_ms for r in records]

        return AgentPerformanceMetrics(
            agent_name=agent_name,
            total_tests=total,
            pass_rate=passed / total,
            avg_accuracy=sum(r.accuracy_score for r in records) / total,
            avg_response_time_ms=sum(latencies) / total,
            error_rate=errors / total,
            min_response_time_ms=min(latencies),
            max_response_time_ms=max(latencies),
            stddev_response_time_ms=(
                statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
            ),
        )

    def compare(
        self,
        agent_a: str,
        agent_b: str,
        accuracy_threshold: float = 0.05,
        latency_threshold_ms: float = 50.0,
        pass_rate_threshold: float = 0.10,
    ) -> Dict[str, Any]:
        """Compare two agents/configurations and generate recommendations.

        Mirrors the production ``compare_configurations`` tool: differences
        below the thresholds are treated as noise.
        """
        metrics_a = self.metrics_for(agent_a)
        metrics_b = self.metrics_for(agent_b)

        comparison: Dict[str, Any] = {}
        recommendations: List[str] = []

        accuracy_diff = metrics_b.avg_accuracy - metrics_a.avg_accuracy
        comparison["accuracy"] = {
            agent_a: metrics_a.avg_accuracy,
            agent_b: metrics_b.avg_accuracy,
            "difference": accuracy_diff,
            "improvement": accuracy_diff > accuracy_threshold,
        }
        if abs(accuracy_diff) > accuracy_threshold:
            better = agent_b if accuracy_diff > 0 else agent_a
            recommendations.append(
                f"'{better}' shows {abs(accuracy_diff) * 100:.1f}% better accuracy"
            )

        latency_diff = metrics_a.avg_response_time_ms - metrics_b.avg_response_time_ms
        comparison["response_time"] = {
            agent_a: metrics_a.avg_response_time_ms,
            agent_b: metrics_b.avg_response_time_ms,
            "difference": -latency_diff,  # negative because lower is better
            "improvement": latency_diff > latency_threshold_ms,
        }
        if abs(latency_diff) > latency_threshold_ms:
            better = agent_b if latency_diff > 0 else agent_a
            recommendations.append(
                f"'{better}' is {abs(latency_diff):.0f}ms faster on average"
            )

        pass_rate_diff = metrics_b.pass_rate - metrics_a.pass_rate
        comparison["pass_rate"] = {
            agent_a: metrics_a.pass_rate,
            agent_b: metrics_b.pass_rate,
            "difference": pass_rate_diff,
            "improvement": pass_rate_diff > pass_rate_threshold,
        }
        if abs(pass_rate_diff) > pass_rate_threshold:
            better = agent_b if pass_rate_diff > 0 else agent_a
            recommendations.append(
                f"'{better}' has {abs(pass_rate_diff) * 100:.1f}% better pass rate"
            )

        return {
            "agents": {
                agent_a: metrics_a.to_dict(),
                agent_b: metrics_b.to_dict(),
            },
            "comparison": comparison,
            "recommendations": recommendations,
        }


def detect_regressions(
    baseline: AgentPerformanceMetrics,
    current: AgentPerformanceMetrics,
    max_accuracy_drop: float = 0.05,
    max_pass_rate_drop: float = 0.05,
    max_latency_increase_ms: float = 250.0,
    max_error_rate_increase: float = 0.05,
) -> List[str]:
    """Compare a current run against a stored baseline; return regressions.

    An empty list means no regression was detected. This is the hook used in
    CI to fail a build when an agent, prompt, or model change degrades
    performance on the ground-truth suite.
    """
    regressions: List[str] = []

    accuracy_drop = baseline.avg_accuracy - current.avg_accuracy
    if accuracy_drop > max_accuracy_drop:
        regressions.append(
            f"accuracy dropped {accuracy_drop * 100:.1f}% "
            f"({baseline.avg_accuracy:.2f} -> {current.avg_accuracy:.2f})"
        )

    pass_rate_drop = baseline.pass_rate - current.pass_rate
    if pass_rate_drop > max_pass_rate_drop:
        regressions.append(
            f"pass rate dropped {pass_rate_drop * 100:.1f}% "
            f"({baseline.pass_rate:.2f} -> {current.pass_rate:.2f})"
        )

    latency_increase = current.avg_response_time_ms - baseline.avg_response_time_ms
    if latency_increase > max_latency_increase_ms:
        regressions.append(
            f"avg latency increased {latency_increase:.0f}ms "
            f"({baseline.avg_response_time_ms:.0f}ms -> "
            f"{current.avg_response_time_ms:.0f}ms)"
        )

    error_rate_increase = current.error_rate - baseline.error_rate
    if error_rate_increase > max_error_rate_increase:
        regressions.append(
            f"error rate increased {error_rate_increase * 100:.1f}% "
            f"({baseline.error_rate:.2f} -> {current.error_rate:.2f})"
        )

    return regressions
