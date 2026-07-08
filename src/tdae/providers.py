"""LLM provider abstraction.

Agents under evaluation depend only on the :class:`LLMProvider` protocol, so
the harness can swap a real provider (Anthropic, OpenAI, a local model, ...)
for the deterministic :class:`MockLLMProvider` in tests and CI. The mock needs
no API keys and returns canned responses, which keeps the quick-start and the
regression suite fully offline and reproducible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal completion interface an agent needs from an LLM.

    Production implementations wrap a vendor SDK/HTTP client and are async;
    the showcase keeps the interface synchronous for a dependency-free run.
    """

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's text completion for ``prompt``."""
        ...


@dataclass
class MockLLMProvider:
    """Deterministic, offline LLM stand-in.

    ``responses`` is an ordered list of ``(needle, response)`` pairs. The
    first ``needle`` found (case-insensitively) in the prompt wins; otherwise
    ``default_response`` is returned. Every prompt is recorded in ``calls``
    so tests can assert on what the agent asked.
    """

    responses: List[Tuple[str, str]] = field(default_factory=list)
    default_response: str = "NOT_FOUND"
    latency_ms: float = 0.0
    calls: List[str] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(prompt)
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        lowered = prompt.lower()
        for needle, response in self.responses:
            if needle.lower() in lowered:
                return response
        return self.default_response

    @classmethod
    def for_ground_truth(cls, latency_ms: float = 0.0) -> "MockLLMProvider":
        """Mock pre-loaded with canned answers for the built-in scenarios.

        Maps each synthetic customer's email domain (which appears in the
        extraction prompt) to that customer's canonical company name, so the
        customer-matching demo runs end-to-end without any API key.
        """
        from tdae.ground_truth import CUSTOMER_FIXTURES

        responses: List[Tuple[str, str]] = [
            (fixture["email_domain"], fixture["company_name"])
            for fixture in CUSTOMER_FIXTURES
        ]
        return cls(responses=responses, latency_ms=latency_ms)


@dataclass
class ScriptedLLMProvider:
    """Returns a fixed sequence of responses, one per call.

    Useful for scripting multi-step agent behaviour in tests. Raises
    ``IndexError`` when the script is exhausted, which surfaces unexpected
    extra LLM calls as loud test failures.
    """

    script: Sequence[str] = ()
    calls: List[str] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        index = len(self.calls)
        self.calls.append(prompt)
        return self.script[index]
