"""Accuracy scoring for agent outputs against ground-truth scenarios.

Scoring layers:

* :func:`calculate_accuracy_score` - a generic per-field accuracy score with
  numeric tolerance and fuzzy string matching (used by the batch runner for
  free-form outputs).
* :class:`TestResultValidator` - the scenario-aware validator that produces a
  weighted validation report (field accuracy, price accuracy, completeness,
  critical validations) and the final pass/fail verdict.
* :class:`Scorer` - the protocol the orchestrator scores through.
  :class:`FreightScorer` (the default) wraps :class:`TestResultValidator`;
  other domains plug in their own scorer.
* Domain-neutral helpers for writing scorers: :func:`within_tolerance`,
  :func:`fact_check` and :func:`precision_recall`.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

# Overall score at or above this threshold counts as a pass.
PASS_THRESHOLD = 0.85

# Weights for the overall score.
WEIGHT_FIELD_ACCURACY = 0.3
WEIGHT_PRICE_ACCURACY = 0.3
WEIGHT_COMPLETENESS = 0.2
WEIGHT_CRITICAL = 0.2


def calculate_accuracy_score(
    expected_output: Dict[str, Any],
    actual_output: Dict[str, Any],
    numeric_tolerance: float = 0.10,
) -> float:
    """Generic per-field accuracy between two flat dicts.

    * exact equality scores 1 point per field
    * numeric values within ``numeric_tolerance`` (fractional) score 1 point
    * strings that contain each other score 0.8 points (partial credit)
    """
    if not expected_output:
        return 0.0

    total_fields = 0
    correct_fields = 0.0

    for key, expected_value in expected_output.items():
        total_fields += 1
        actual_value = actual_output.get(key)

        if actual_value == expected_value:
            correct_fields += 1
        elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            tolerance = abs(expected_value) * numeric_tolerance
            if abs(expected_value - actual_value) <= tolerance:
                correct_fields += 1
        elif isinstance(expected_value, str) and isinstance(actual_value, str):
            if (
                expected_value.lower() in actual_value.lower()
                or actual_value.lower() in expected_value.lower()
            ):
                correct_fields += 0.8  # partial credit for fuzzy string match

    return correct_fields / total_fields if total_fields > 0 else 0.0


class TestResultValidator:
    """Validates agent outputs against ground-truth scenario expectations."""

    @staticmethod
    def calculate_field_accuracy(
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        required_fields: List[str],
    ) -> float:
        """Fraction of required fields extracted correctly."""
        if not actual:
            return 0.0

        correct_fields = 0
        total_fields = len(required_fields)

        for field_name in required_fields:
            if field_name in actual and field_name in expected:
                if TestResultValidator._compare_values(expected[field_name], actual[field_name]):
                    correct_fields += 1

        return correct_fields / total_fields if total_fields > 0 else 0.0

    @staticmethod
    def calculate_price_accuracy(expected: float, actual: float, tolerance: float) -> float:
        """Price accuracy scaled within tolerance (1.0 = exact, 0.0 = outside)."""
        if not actual or not expected:
            return 0.0

        difference = abs(expected - actual) / expected
        if difference <= tolerance:
            return 1.0 - (difference / tolerance)
        return 0.0

    @staticmethod
    def calculate_completeness_score(expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
        """Fraction of expected (nested) keys present in the actual output."""
        if not actual:
            return 0.0

        expected_keys = TestResultValidator._get_all_keys(expected)
        actual_keys = TestResultValidator._get_all_keys(actual)

        if not expected_keys:
            return 0.0

        matched_keys = expected_keys.intersection(actual_keys)
        return len(matched_keys) / len(expected_keys)

    @staticmethod
    def validate_critical_fields(
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        critical_fields: List[str],
    ) -> Dict[str, bool]:
        """Per-field verdicts for critical validations.

        Critical fields not present in the expected data are skipped (True).
        """
        validation_results: Dict[str, bool] = {}

        for field_name in critical_fields:
            if field_name in expected:
                validation_results[field_name] = field_name in actual and (
                    TestResultValidator._compare_values(expected[field_name], actual[field_name])
                )
            else:
                validation_results[field_name] = True  # skip if not in expected

        return validation_results

    @staticmethod
    def generate_validation_report(
        scenario: Dict[str, Any],
        agent_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce the weighted validation report for one scenario run."""
        expected = scenario["expected_output"]
        metrics = scenario.get("validation_metrics", {})

        # Field accuracy - check parsed_data first, then customer_match, then booking_data.
        required_fields = metrics.get("required_fields", [])
        expected_data = (
            expected.get("parsed_data")
            or expected.get("customer_match")
            or expected.get("booking_data", {})
        )
        actual_data = (
            agent_output.get("parsed_data")
            or agent_output.get("customer_match")
            or agent_output.get("booking_data", {})
        )
        field_accuracy = TestResultValidator.calculate_field_accuracy(
            expected_data, actual_data, required_fields
        )

        # Price accuracy. Non-pricing scenarios default to 100%.
        price_accuracy = 1.0
        if "calculated_quote" in expected:
            if "calculated_quote" in agent_output:
                exp_quote = expected["calculated_quote"]
                act_quote = agent_output.get("calculated_quote", {})

                exp_no_rates = exp_quote.get("no_rates_found", False)
                act_no_rates = act_quote.get("no_rates_found", False)

                if exp_no_rates and act_no_rates:
                    # Both correctly identified that no rates were available.
                    price_accuracy = 1.0
                elif exp_no_rates and not act_no_rates:
                    price_accuracy = 0.5  # partial credit
                elif not exp_no_rates and act_no_rates:
                    price_accuracy = 0.0
                else:
                    expected_price = exp_quote.get("total_cost", 0)
                    actual_price = act_quote.get("total_cost", 0)
                    price_tolerance = metrics.get("price_tolerance", 0.05)
                    price_accuracy = TestResultValidator.calculate_price_accuracy(
                        expected_price, actual_price, price_tolerance
                    )
            else:
                price_accuracy = 0.0  # expected a quote but got none

        completeness = TestResultValidator.calculate_completeness_score(expected, agent_output)

        critical_fields = metrics.get("critical_validations", [])
        critical_validation = TestResultValidator.validate_critical_fields(
            expected_data, actual_data, critical_fields
        )

        critical_score = (
            sum(critical_validation.values()) / len(critical_validation)
            if critical_validation
            else 0.0
        )

        overall_score = (
            field_accuracy * WEIGHT_FIELD_ACCURACY
            + price_accuracy * WEIGHT_PRICE_ACCURACY
            + completeness * WEIGHT_COMPLETENESS
            + critical_score * WEIGHT_CRITICAL
        )

        return {
            "scenario_id": scenario["scenario_id"],
            "field_accuracy": field_accuracy,
            "price_accuracy": price_accuracy,
            "completeness_score": completeness,
            "critical_validations": critical_validation,
            "overall_score": overall_score,
            "passed": overall_score >= PASS_THRESHOLD,
            "details": {
                "expected_fields": len(required_fields),
                "matched_fields": int(field_accuracy * len(required_fields)),
                "critical_passed": sum(critical_validation.values()),
                "critical_total": len(critical_validation),
            },
        }

    @staticmethod
    def _compare_values(expected: Any, actual: Any) -> bool:
        """Compare values with type flexibility."""
        if expected == actual:
            return True

        # Case-insensitive string comparison.
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower().strip() == actual.lower().strip()

        # Numeric comparison with a small absolute tolerance.
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(expected - actual) < 0.01

        # Order-independent list comparison.
        if isinstance(expected, list) and isinstance(actual, list):
            try:
                return set(expected) == set(actual)
            except TypeError:
                return expected == actual

        return False

    @staticmethod
    def _get_all_keys(d: Dict[str, Any], parent_key: str = "") -> set:
        """Recursively collect all keys from a nested dictionary."""
        keys: set = set()
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            keys.add(new_key)
            if isinstance(v, dict):
                keys.update(TestResultValidator._get_all_keys(v, new_key))
        return keys


# ---------------------------------------------------------------------------
# Pluggable scoring
# ---------------------------------------------------------------------------


@runtime_checkable
class Scorer(Protocol):
    """Scores one agent output against one scenario.

    ``score`` must return a dict with at least ``overall_score`` (0.0-1.0) and
    ``passed`` (bool). Any other keys are kept in the scenario report, so a
    scorer can return whatever detail its domain needs.
    """

    def score(self, scenario: Dict[str, Any], agent_output: Dict[str, Any]) -> Dict[str, Any]:
        ...


class FreightScorer:
    """The default scorer: the weighted freight validation report.

    Behaves exactly like :meth:`TestResultValidator.generate_validation_report`;
    ``pass_threshold`` only changes the pass/fail verdict.
    """

    def __init__(self, pass_threshold: float = PASS_THRESHOLD) -> None:
        self.pass_threshold = pass_threshold

    def score(self, scenario: Dict[str, Any], agent_output: Dict[str, Any]) -> Dict[str, Any]:
        report = TestResultValidator.generate_validation_report(scenario, agent_output)
        report["passed"] = report["overall_score"] >= self.pass_threshold
        return report


# ---------------------------------------------------------------------------
# Domain-neutral helpers for writing scorers
# ---------------------------------------------------------------------------


def within_tolerance(
    expected: float,
    actual: Any,
    *,
    rel_tol: Optional[float] = None,
    abs_tol: Optional[float] = None,
) -> bool:
    """True if ``actual`` is within tolerance of ``expected``.

    The allowed difference is the larger of ``rel_tol * |expected|`` and
    ``abs_tol`` (the same rule as :func:`math.isclose`). With neither given,
    the values must be equal. Non-numeric ``actual`` values never match.
    """
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    allowed = max(
        (rel_tol or 0.0) * abs(expected),
        abs_tol or 0.0,
    )
    return abs(expected - actual) <= allowed


def _lookup(data: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path (``"fit.center"``) in nested dicts; raises KeyError."""
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def fact_check(
    expected_facts: Mapping[str, Any],
    output: Mapping[str, Any],
    tolerances: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, bool]:
    """Check each expected fact against an agent output; one verdict per fact.

    Keys of ``expected_facts`` are dotted paths into ``output``
    (``"facts.rabi_freq_khz"``). How each fact is compared:

    * numbers - :func:`within_tolerance`, using ``tolerances[path]``
      (``{"rel_tol": ..., "abs_tol": ...}``) when given, else exact
    * lists / sets - every expected item must be present (extras are allowed,
      so "cites runs 112 and 140" passes when a third run is also cited)
    * strings - case- and whitespace-insensitive equality
    * anything else - equality

    A fact whose path is missing from ``output`` fails.
    """
    tolerances = tolerances or {}
    verdicts: Dict[str, bool] = {}
    for path, expected in expected_facts.items():
        try:
            actual = _lookup(output, path)
        except KeyError:
            verdicts[path] = False
            continue

        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            verdicts[path] = within_tolerance(expected, actual, **dict(tolerances.get(path, {})))
        elif isinstance(expected, (list, tuple, set, frozenset)):
            if not isinstance(actual, (list, tuple, set, frozenset)):
                verdicts[path] = False
            else:
                verdicts[path] = all(item in actual for item in expected)
        elif isinstance(expected, str) and isinstance(actual, str):
            verdicts[path] = expected.strip().lower() == actual.strip().lower()
        else:
            verdicts[path] = expected == actual
    return verdicts


def precision_recall(
    expected: Sequence[Any],
    found: Sequence[Any],
    match: Optional[Callable[[Any, Any], bool]] = None,
) -> Dict[str, Any]:
    """Precision / recall of ``found`` items against ``expected`` items.

    Each expected item can be matched by at most one found item and vice
    versa; the pairing maximises the number of matches. ``match(expected,
    found)`` decides whether two items correspond (default: equality), e.g.
    "alert of the right kind within 30 minutes of the injected event".

    Conventions: precision is 1.0 when nothing was found, recall is 1.0 when
    nothing was expected (an empty window with no alerts is a perfect score).

    Returns ``precision``, ``recall``, ``f1``, the ``true_positives`` /
    ``false_positives`` / ``false_negatives`` counts and ``matches`` as a
    list of ``(expected_index, found_index)`` pairs.
    """
    match = match or (lambda e, f: e == f)
    candidates = [
        [j for j, f in enumerate(found) if match(e, f)] for e in expected
    ]
    found_to_expected: Dict[int, int] = {}

    def assign(i: int, seen: set) -> bool:
        # Augmenting-path step of Kuhn's bipartite matching algorithm.
        for j in candidates[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in found_to_expected or assign(found_to_expected[j], seen):
                found_to_expected[j] = i
                return True
        return False

    for i in range(len(expected)):
        assign(i, set())

    tp = len(found_to_expected)
    fp = len(found) - tp
    fn = len(expected) - tp
    precision = tp / len(found) if found else 1.0
    recall = tp / len(expected) if expected else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "matches": sorted((i, j) for j, i in found_to_expected.items()),
    }
