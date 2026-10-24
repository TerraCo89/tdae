"""TDAE - Test-Driven Agent Evolution.

Ground-truth evaluation, accuracy scoring, per-agent performance metrics,
and regression tracking for LLM agents.
"""

from tdae.ground_truth import (
    CUSTOMER_FIXTURES,
    GroundTruthTestData,
    ValidationMetrics,
)
from tdae.metrics import (
    AgentPerformanceMetrics,
    InMemoryResultStore,
    JsonlResultStore,
    TestExecutionRecord,
    detect_regressions,
)
from tdae.orchestrator import ScenarioAgent, TDAEOrchestrator, print_summary
from tdae.providers import LLMProvider, MockLLMProvider, ScriptedLLMProvider
from tdae.scoring import (
    PASS_THRESHOLD,
    FreightScorer,
    Scorer,
    TestResultValidator,
    calculate_accuracy_score,
    fact_check,
    precision_recall,
    within_tolerance,
)

__version__ = "0.2.0"

__all__ = [
    "AgentPerformanceMetrics",
    "CUSTOMER_FIXTURES",
    "FreightScorer",
    "GroundTruthTestData",
    "InMemoryResultStore",
    "JsonlResultStore",
    "LLMProvider",
    "MockLLMProvider",
    "PASS_THRESHOLD",
    "ScenarioAgent",
    "Scorer",
    "ScriptedLLMProvider",
    "TDAEOrchestrator",
    "TestExecutionRecord",
    "TestResultValidator",
    "ValidationMetrics",
    "calculate_accuracy_score",
    "detect_regressions",
    "fact_check",
    "precision_recall",
    "print_summary",
    "within_tolerance",
    "__version__",
]
