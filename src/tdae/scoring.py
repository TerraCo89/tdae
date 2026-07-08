"""Accuracy scoring for agent outputs against ground-truth scenarios.

Two scoring layers:

* :func:`calculate_accuracy_score` - a generic per-field accuracy score with
  numeric tolerance and fuzzy string matching (used by the batch runner for
  free-form outputs).
* :class:`TestResultValidator` - the scenario-aware validator that produces a
  weighted validation report (field accuracy, price accuracy, completeness,
  critical validations) and the final pass/fail verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
