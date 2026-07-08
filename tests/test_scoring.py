"""Tests for accuracy scoring and validation reports."""

import pytest

from tdae.ground_truth import GroundTruthTestData
from tdae.scoring import (
    PASS_THRESHOLD,
    TestResultValidator,
    calculate_accuracy_score,
)


# ---------------------------------------------------------------------------
# calculate_accuracy_score (generic per-field scorer)
# ---------------------------------------------------------------------------

def test_accuracy_score_exact_match():
    expected = {"origin": "Shanghai", "total": 5230.0}
    assert calculate_accuracy_score(expected, dict(expected)) == 1.0


def test_accuracy_score_numeric_tolerance():
    expected = {"total": 100.0}
    assert calculate_accuracy_score(expected, {"total": 105.0}) == 1.0  # within 10%
    assert calculate_accuracy_score(expected, {"total": 150.0}) == 0.0  # outside


def test_accuracy_score_fuzzy_string_partial_credit():
    expected = {"carrier": "Pacific Star"}
    score = calculate_accuracy_score(expected, {"carrier": "Pacific Star Line"})
    assert score == pytest.approx(0.8)


def test_accuracy_score_empty_expected():
    assert calculate_accuracy_score({}, {"anything": 1}) == 0.0


# ---------------------------------------------------------------------------
# TestResultValidator components
# ---------------------------------------------------------------------------

def test_field_accuracy_counts_required_fields():
    expected = {"origin": "Shanghai", "destination": "LA", "cargo_type": "FCL"}
    actual = {"origin": "shanghai ", "destination": "Rotterdam", "cargo_type": "FCL"}
    score = TestResultValidator.calculate_field_accuracy(
        expected, actual, ["origin", "destination", "cargo_type"]
    )
    assert score == pytest.approx(2 / 3)  # origin case-insensitive, destination wrong


def test_price_accuracy_scales_within_tolerance():
    assert TestResultValidator.calculate_price_accuracy(100.0, 100.0, 0.05) == 1.0
    # 2.5% off with 5% tolerance -> half credit
    assert TestResultValidator.calculate_price_accuracy(100.0, 102.5, 0.05) == pytest.approx(0.5)
    assert TestResultValidator.calculate_price_accuracy(100.0, 120.0, 0.05) == 0.0
    assert TestResultValidator.calculate_price_accuracy(100.0, 0.0, 0.05) == 0.0


def test_completeness_uses_nested_keys():
    expected = {"a": {"b": 1, "c": 2}, "d": 3}
    actual = {"a": {"b": 99}, "d": 0}
    # keys: a, a.b, a.c, d -> matched: a, a.b, d
    score = TestResultValidator.calculate_completeness_score(expected, actual)
    assert score == pytest.approx(3 / 4)


def test_critical_fields_skip_when_not_in_expected():
    result = TestResultValidator.validate_critical_fields(
        expected={"origin": "Shanghai"},
        actual={"origin": "Shanghai"},
        critical_fields=["origin", "not_a_real_field"],
    )
    assert result == {"origin": True, "not_a_real_field": True}


# ---------------------------------------------------------------------------
# Full validation reports against real scenarios
# ---------------------------------------------------------------------------

def test_perfect_output_passes():
    scenario = GroundTruthTestData.get_scenario_by_id("FCL_SHANGHAI_LA_001")
    report = TestResultValidator.generate_validation_report(
        scenario, scenario["expected_output"]
    )
    assert report["passed"] is True
    assert report["overall_score"] == pytest.approx(1.0)
    assert report["field_accuracy"] == 1.0
    assert report["price_accuracy"] == 1.0


def test_empty_output_fails():
    scenario = GroundTruthTestData.get_scenario_by_id("FCL_SHANGHAI_LA_001")
    report = TestResultValidator.generate_validation_report(scenario, {})
    assert report["passed"] is False
    assert report["overall_score"] < PASS_THRESHOLD
    assert report["price_accuracy"] == 0.0  # expected a quote, got none


def test_price_outside_tolerance_reduces_score():
    scenario = GroundTruthTestData.get_scenario_by_id("FCL_SHANGHAI_LA_001")
    import copy

    output = copy.deepcopy(scenario["expected_output"])
    output["calculated_quote"]["total_cost"] *= 2  # way outside 5% tolerance
    report = TestResultValidator.generate_validation_report(scenario, output)
    assert report["price_accuracy"] == 0.0
    assert report["passed"] is False


def test_no_rates_found_agreement_scores_full_price_accuracy():
    scenario = GroundTruthTestData.get_scenario_by_id("FCL_SHANGHAI_LA_001")
    import copy

    scenario = copy.deepcopy(scenario)
    scenario["expected_output"]["calculated_quote"] = {"no_rates_found": True}
    output = copy.deepcopy(scenario["expected_output"])
    report = TestResultValidator.generate_validation_report(scenario, output)
    assert report["price_accuracy"] == 1.0


def test_booking_scenario_perfect_output_passes():
    scenario = GroundTruthTestData.get_scenario_by_id("BOOKING_CONF_001")
    report = TestResultValidator.generate_validation_report(
        scenario, scenario["expected_output"]
    )
    assert report["passed"] is True
