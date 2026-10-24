"""Tests for the domain-neutral scoring helpers (within_tolerance, fact_check, precision_recall)."""

import pytest

from tdae.scoring import fact_check, precision_recall, within_tolerance


# ---------------------------------------------------------------------------
# within_tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expected,actual,kwargs,ok", [
    (100.0, 100.0, {}, True),                          # exact, no tolerance given
    (100.0, 100.1, {}, False),
    (100.0, 104.9, {"rel_tol": 0.05}, True),
    (100.0, 105.1, {"rel_tol": 0.05}, False),
    (0.0, 0.004, {"abs_tol": 0.005}, True),            # abs_tol is what saves zero
    (0.0, 0.004, {"rel_tol": 0.5}, False),
    (100.0, 102.0, {"rel_tol": 0.01, "abs_tol": 3.0}, True),   # the larger allowance wins
    (-50.0, -52.0, {"rel_tol": 0.05}, True),           # negative expected values
    (100.0, "100", {"rel_tol": 0.5}, False),           # non-numeric never matches
    (100.0, None, {"rel_tol": 0.5}, False),
    (1.0, True, {"abs_tol": 1.0}, False),              # bools are not numbers here
])
def test_within_tolerance(expected, actual, kwargs, ok):
    assert within_tolerance(expected, actual, **kwargs) is ok


# ---------------------------------------------------------------------------
# fact_check
# ---------------------------------------------------------------------------

ANSWER = {
    "answer": "Rabi frequency dropped after the AOM power change in run 140.",
    "cited_runs": [112, 140, 141],
    "language": "EN",
    "facts": {"rabi_freq_khz": 71.2, "parameter": "aom_power"},
}


def test_fact_check_all_facts_pass():
    verdicts = fact_check(
        {
            "cited_runs": [112, 140],                  # subset: an extra cited run is fine
            "language": "en",                          # case-insensitive
            "facts.rabi_freq_khz": 70.7,
            "facts.parameter": " AOM_power ",          # whitespace-insensitive
        },
        ANSWER,
        tolerances={"facts.rabi_freq_khz": {"rel_tol": 0.02}},
    )
    assert verdicts == {
        "cited_runs": True,
        "language": True,
        "facts.rabi_freq_khz": True,
        "facts.parameter": True,
    }


def test_fact_check_reports_each_failure():
    verdicts = fact_check(
        {
            "cited_runs": [112, 150],                  # 150 not cited
            "facts.rabi_freq_khz": 85.0,               # no tolerance given -> exact
            "facts.detuning_mhz": 0.2,                 # missing path
            "language": "ja",
        },
        ANSWER,
    )
    assert verdicts == {
        "cited_runs": False,
        "facts.rabi_freq_khz": False,
        "facts.detuning_mhz": False,
        "language": False,
    }


def test_fact_check_type_mismatches_fail():
    output = {"cited_runs": "112, 140", "count": "3", "flag": True}
    verdicts = fact_check({"cited_runs": [112], "count": 3, "flag": True}, output)
    assert verdicts == {"cited_runs": False, "count": False, "flag": True}


# ---------------------------------------------------------------------------
# precision_recall
# ---------------------------------------------------------------------------

def test_precision_recall_exact_items():
    result = precision_recall(["lock_loss", "drift", "column"], ["drift", "lock_loss", "exposure"])
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["matches"] == [(0, 1), (1, 0)]


def test_precision_recall_one_to_one():
    # Two alerts for one event: only one can count, the other is a false positive.
    result = precision_recall(["lock_loss"], ["lock_loss", "lock_loss"])
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_precision_recall_custom_match_finds_best_pairing():
    # Events at t=10 and t=20; alerts at t=15 and t=26; match = within 6.
    # Greedy (alert 15 -> event 10 or 20 first) could strand an alert;
    # the maximum matching pairs 15->10 and 26->20.
    events = [10, 20]
    alerts = [15, 26]
    result = precision_recall(events, alerts, match=lambda e, a: 0 <= a - e <= 6)
    assert result["true_positives"] == 2
    assert result["matches"] == [(0, 0), (1, 1)]

    # Order-sensitive case: alert 15 fits both events, alert 12 fits only event 10.
    # Taking alert 15 for event 10 first would strand event 20; the matching
    # re-pairs so both events are caught.
    result = precision_recall(events, [15, 12], match=lambda e, a: abs(a - e) <= 5)
    assert result["true_positives"] == 2
    assert result["matches"] == [(0, 1), (1, 0)]


@pytest.mark.parametrize("expected,found,precision,recall,f1", [
    ([], [], 1.0, 1.0, 1.0),          # quiet window, no alerts: perfect
    (["e"], [], 1.0, 0.0, 0.0),       # missed event
    ([], ["a"], 0.0, 1.0, 0.0),       # false alarm
])
def test_precision_recall_empty_conventions(expected, found, precision, recall, f1):
    result = precision_recall(expected, found)
    assert result["precision"] == precision
    assert result["recall"] == recall
    assert result["f1"] == f1
