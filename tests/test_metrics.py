"""Tests for the result store, performance metrics, and regression tracking."""

import pytest

from tdae.metrics import (
    AgentPerformanceMetrics,
    InMemoryResultStore,
    TestExecutionRecord,
    detect_regressions,
)


def _record(agent, result="pass", accuracy=1.0, latency=10.0, scenario="S1"):
    return TestExecutionRecord(
        scenario_id=scenario,
        agent_name=agent,
        result=result,
        accuracy_score=accuracy,
        execution_time_ms=latency,
    )


def test_metrics_for_empty_store():
    store = InMemoryResultStore()
    metrics = store.metrics_for("nobody")
    assert metrics.total_tests == 0
    assert metrics.pass_rate == 0.0
    assert metrics.error_rate == 0.0


def test_metrics_aggregation():
    store = InMemoryResultStore()
    store.add(_record("a", "pass", accuracy=0.9, latency=10.0))
    store.add(_record("a", "fail", accuracy=0.5, latency=30.0))
    store.add(_record("a", "error", accuracy=0.0, latency=20.0))
    store.add(_record("other", "pass", accuracy=1.0, latency=5.0))  # ignored

    metrics = store.metrics_for("a")
    assert metrics.total_tests == 3
    assert metrics.pass_rate == pytest.approx(1 / 3)
    assert metrics.error_rate == pytest.approx(1 / 3)
    assert metrics.avg_accuracy == pytest.approx((0.9 + 0.5 + 0.0) / 3)
    assert metrics.avg_response_time_ms == pytest.approx(20.0)
    assert metrics.min_response_time_ms == 10.0
    assert metrics.max_response_time_ms == 30.0


def test_record_filters():
    store = InMemoryResultStore()
    store.add(_record("a", scenario="S1"))
    store.add(_record("a", scenario="S2"))
    store.add(_record("b", scenario="S1"))
    assert len(store.records(agent_name="a")) == 2
    assert len(store.records(scenario_id="S1")) == 2
    assert len(store.records(agent_name="a", scenario_id="S1")) == 1
    assert len(store.records(limit=1)) == 1


def test_compare_generates_recommendations():
    store = InMemoryResultStore()
    for _ in range(4):
        store.add(_record("baseline", "pass", accuracy=0.95, latency=10.0))
    for _ in range(4):
        store.add(_record("candidate", "fail", accuracy=0.60, latency=200.0))

    comparison = store.compare("baseline", "candidate")
    assert comparison["comparison"]["accuracy"]["improvement"] is False
    recommendations = " ".join(comparison["recommendations"])
    assert "baseline" in recommendations  # baseline wins on accuracy/pass rate/latency


def test_compare_similar_configs_yields_no_recommendations():
    store = InMemoryResultStore()
    for _ in range(4):
        store.add(_record("a", "pass", accuracy=0.90, latency=10.0))
        store.add(_record("b", "pass", accuracy=0.91, latency=12.0))
    comparison = store.compare("a", "b")
    assert comparison["recommendations"] == []


def test_detect_regressions_flags_drops():
    baseline = AgentPerformanceMetrics(
        agent_name="a", total_tests=10, pass_rate=1.0,
        avg_accuracy=0.95, avg_response_time_ms=50.0, error_rate=0.0,
    )
    degraded = AgentPerformanceMetrics(
        agent_name="a", total_tests=10, pass_rate=0.5,
        avg_accuracy=0.70, avg_response_time_ms=900.0, error_rate=0.2,
    )
    regressions = detect_regressions(baseline, degraded)
    assert len(regressions) == 4  # accuracy, pass rate, latency, error rate


def test_detect_regressions_clean_run():
    baseline = AgentPerformanceMetrics(
        agent_name="a", total_tests=10, pass_rate=0.9,
        avg_accuracy=0.90, avg_response_time_ms=50.0, error_rate=0.0,
    )
    improved = AgentPerformanceMetrics(
        agent_name="a", total_tests=10, pass_rate=0.95,
        avg_accuracy=0.92, avg_response_time_ms=45.0, error_rate=0.0,
    )
    assert detect_regressions(baseline, improved) == []
