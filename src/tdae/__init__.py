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
    TestExecutionRecord,
    detect_regressions,
)
from tdae.orchestrator import ScenarioAgent, TDAEOrchestrator, print_summary
from tdae.providers import LLMProvider, MockLLMProvider, ScriptedLLMProvider
from tdae.scoring import (
    PASS_THRESHOLD,
    TestResultValidator,
    calculate_accuracy_score,
)

__version__ = "0.1.0"

__all__ = [
    "AgentPerformanceMetrics",
    "CUSTOMER_FIXTURES",
    "GroundTruthTestData",
    "InMemoryResultStore",
    "LLMProvider",
    "MockLLMProvider",
    "PASS_THRESHOLD",
    "ScenarioAgent",
    "ScriptedLLMProvider",
    "TDAEOrchestrator",
    "TestExecutionRecord",
    "TestResultValidator",
    "ValidationMetrics",
    "calculate_accuracy_score",
    "detect_regressions",
    "print_summary",
    "__version__",
]
