"""Tests for the ground-truth scenario suite."""

import json

import pytest

from tdae.ground_truth import CUSTOMER_FIXTURES, GroundTruthTestData

REQUIRED_SCENARIO_KEYS = {
    "scenario_id",
    "scenario_type",
    "difficulty",
    "agent_capabilities",
    "db_query",
    "validation_metrics",
    "email_content",
    "expected_output",
}

EXPECTED_TYPES = {
    "FCL_QUOTE",
    "LCL_QUOTE",
    "AIR_FREIGHT",
    "MULTIMODAL",
    "CUSTOMER_MATCHING",
    "BOOKING_CONFIRMATION",
}


@pytest.fixture(autouse=True)
def clean_registry():
    GroundTruthTestData.clear_generated_scenarios()
    yield
    GroundTruthTestData.clear_generated_scenarios()


def test_suite_contains_36_scenarios():
    scenarios = GroundTruthTestData.get_all_scenarios()
    assert len(scenarios) == 36


def test_scenario_ids_are_unique():
    ids = GroundTruthTestData.list_scenario_ids()
    assert len(ids) == len(set(ids))


def test_six_variants_per_type():
    scenarios = GroundTruthTestData.get_all_scenarios()
    by_type = {}
    for scenario in scenarios:
        by_type.setdefault(scenario["scenario_type"], []).append(scenario)
    assert set(by_type) == EXPECTED_TYPES
    for scenario_type, group in by_type.items():
        assert len(group) == 6, f"{scenario_type} should have 6 variants"


def test_scenarios_have_required_structure():
    for scenario in GroundTruthTestData.get_all_scenarios():
        missing = REQUIRED_SCENARIO_KEYS - set(scenario)
        assert not missing, f"{scenario.get('scenario_id')} missing {missing}"
        assert scenario["difficulty"] in {"simple", "moderate", "complex", "expert"}
        assert scenario["email_content"].strip()
        assert scenario["validation_metrics"]["required_fields"]


def test_quote_totals_are_internally_consistent():
    """total_cost must be reachable from the itemized charges."""
    non_charge_keys = {
        "total_cost", "cost_per_container", "cost_per_cbm", "cost_per_kg",
        "currency", "transit_time_days", "validity_days", "carrier_options",
        "next_consolidation", "freight_rate_per_cbm", "air_freight_rate",
        "chargeable_weight",
    }
    for scenario in GroundTruthTestData.get_all_scenarios():
        quote = scenario["expected_output"].get("calculated_quote")
        if not quote:
            continue
        charges = [
            v for k, v in quote.items()
            if k not in non_charge_keys and isinstance(v, (int, float))
        ]
        assert quote["total_cost"] == pytest.approx(sum(charges), abs=0.05), (
            f"{scenario['scenario_id']}: total_cost inconsistent with charges"
        )


def test_get_scenarios_by_type_and_capability():
    fcl = GroundTruthTestData.get_scenarios_by_type("FCL_QUOTE")
    assert len(fcl) == 6
    dg = GroundTruthTestData.get_scenarios_by_capability("dangerous_goods")
    assert len(dg) == 6  # all air-freight scenarios


def test_get_scenario_by_id():
    scenario = GroundTruthTestData.get_scenario_by_id("FCL_SHANGHAI_LA_001")
    assert scenario is not None
    assert scenario["scenario_type"] == "FCL_QUOTE"
    assert GroundTruthTestData.get_scenario_by_id("DOES_NOT_EXIST") is None


def test_register_and_unregister_generated_scenario():
    scenario = {
        "scenario_id": "GEN_TEST_001",
        "scenario_type": "FCL_QUOTE",
        "difficulty": "simple",
        "email_content": "From: someone@example-freight.test\nTest",
        "expected_output": {"parsed_data": {}},
        "validation_metrics": {"required_fields": ["origin"]},
    }
    registered_id = GroundTruthTestData.register_scenario(scenario)
    assert registered_id == "GEN_TEST_001"
    assert len(GroundTruthTestData.get_all_scenarios()) == 37
    assert len(GroundTruthTestData.get_all_scenarios(include_generated=False)) == 36
    assert GroundTruthTestData.unregister_scenario(registered_id) is True
    assert len(GroundTruthTestData.get_all_scenarios()) == 36


def test_customer_scenarios_align_with_fixtures():
    """Every customer scenario's expected customer must exist in the fixtures."""
    fixture_ids = {f["customer_id"] for f in CUSTOMER_FIXTURES}
    for scenario in GroundTruthTestData.get_scenarios_by_type("CUSTOMER_MATCHING"):
        expected = scenario["expected_output"]["customer_match"]
        assert expected["customer_id"] in fixture_ids


def test_no_production_identifiers_in_scenario_data():
    """Sanitization guard: scenario data must stay synthetic and neutral.

    The needles are assembled via concatenation so that a plain-text grep of
    this repository for the identifiers themselves comes back empty.
    """
    blob = json.dumps(GroundTruthTestData.get_all_scenarios()).lower()
    forbidden = [
        "dd" + "wl",                       # the source platform's name
        "supa" + "base",
        "postgres" + "://",
        "postgresql" + "://",
        "redis" + "://",
        "sk" + "-ant" + "-",               # API key prefixes
        "ak" + "ia",
        "gocs" + "px",
        "ey" + "j",                        # JWT header prefix
    ]
    for needle in forbidden:
        assert needle not in blob, f"forbidden identifier in scenario data: {needle}"
    # All scenario lookup addresses use the reserved test domain.
    for scenario in GroundTruthTestData.get_all_scenarios():
        assert scenario["db_query"]["sender_email"].endswith("@example-freight.test")
