"""TDAE ground-truth scenario suite.

36 synthetic freight-forwarding scenarios (6 scenario types x 6 variants each)
with expected outputs and per-scenario validation metrics.

Every company, contact, email address, rate, and booking reference in this
module is synthetic. Sender addresses for scenario lookups use the reserved
``@example-freight.test`` domain; customer domains use the reserved
``.example`` TLD. None of this data corresponds to a real business.

Scenario shape
--------------
Each scenario is a plain dict:

    {
        "scenario_id":        "FCL_SHANGHAI_LA_001",
        "scenario_type":      "FCL_QUOTE",
        "difficulty":         "simple" | "moderate" | "complex" | "expert",
        "agent_capabilities": [...],
        "db_query":           {"sender_email": "tdae.fcl.001@example-freight.test"},
        "validation_metrics": {...},          # see ValidationMetrics
        "email_content":      "...",          # the raw input handed to the agent
        "expected_output":    {...},          # the ground truth to score against
    }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Optional

# Reserved, non-routable domain used for scenario lookup addresses.
SCENARIO_SENDER_DOMAIN = "example-freight.test"

# Neutral placeholder name for the freight forwarder the emails address.
FORWARDER_NAME = "Meridian Freight"


@dataclass
class ValidationMetrics:
    """Metrics contract for validating agent outputs against ground truth."""

    required_fields: List[str]
    price_tolerance: float = 0.05  # fractional tolerance on total_cost
    transit_time_tolerance: int = 2  # days
    critical_validations: List[str] = field(default_factory=list)
    completeness_threshold: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_fields": list(self.required_fields),
            "price_tolerance": self.price_tolerance,
            "transit_time_tolerance": self.transit_time_tolerance,
            "critical_validations": list(self.critical_validations),
            "completeness_threshold": self.completeness_threshold,
        }


def _round2(value: float) -> float:
    return round(value + 1e-9, 2)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _sender(kind: str, seq: int) -> str:
    return f"tdae.{kind}.{seq:03d}@{SCENARIO_SENDER_DOMAIN}"


# ---------------------------------------------------------------------------
# Synthetic customer fixtures (shared with the demo customer directory)
# ---------------------------------------------------------------------------

CUSTOMER_FIXTURES: List[Dict[str, Any]] = [
    {
        "customer_id": "CUST-2024-0042",
        "company_name": "TechCorp International Ltd",
        "email_domain": "techcorp-intl.example",
        "contact_name": "Priya Raman",
        "contact_email": "accounts@techcorp-intl.example",
        "phone": "+1-555-0100",
    },
    {
        "customer_id": "CUST-2024-0107",
        "company_name": "Northwind Traders Pty Ltd",
        "email_domain": "northwind-traders.example",
        "contact_name": "Marcus Bell",
        "contact_email": "ops@northwind-traders.example",
        "phone": "+61-2-5550-0101",
    },
    {
        "customer_id": "CUST-2023-0311",
        "company_name": "Osaka Textile Works KK",
        "email_domain": "osaka-textiles.example",
        "contact_name": "Yuki Tanaka",
        "contact_email": "export@osaka-textiles.example",
        "phone": "+81-6-5550-0102",
    },
    {
        "customer_id": "CUST-2024-0015",
        "company_name": "Andes Mining Supply SA",
        "email_domain": "andes-mining.example",
        "contact_name": "Carla Mendoza",
        "contact_email": "logistics@andes-mining.example",
        "phone": "+56-2-5550-0103",
    },
    {
        "customer_id": "CUST-2022-0288",
        "company_name": "Baltic Furniture OU",
        "email_domain": "baltic-furniture.example",
        "contact_name": "Anders Kask",
        "contact_email": "shipping@baltic-furniture.example",
        "phone": "+372-5550-0104",
    },
    {
        "customer_id": "CUST-2024-0199",
        "company_name": "Sahara AgriExports Ltd",
        "email_domain": "sahara-agri.example",
        "contact_name": "Fatima El Amrani",
        "contact_email": "trade@sahara-agri.example",
        "phone": "+212-5550-0105",
    },
]


# ---------------------------------------------------------------------------
# FCL quote scenarios
# ---------------------------------------------------------------------------

def build_fcl_scenario(
    seq: int = 1,
    origin: str = "Shanghai, China (Port)",
    origin_short: str = "Shanghai",
    destination: str = "Los Angeles, CA, USA",
    dest_short: str = "LA",
    containers: int = 2,
    container_type: str = "40HC",
    weight_per_container: int = 18000,
    volume_per_container: int = 65,
    transit_days: int = 16,
    scale: float = 1.0,
    difficulty: str = "moderate",
) -> Dict[str, Any]:
    """FCL (Full Container Load) quote scenario with comprehensive validation."""
    charge_scale = scale * (containers / 2)
    quote = {
        "ocean_freight": _round2(3200.0 * charge_scale),
        "fuel_surcharge": _round2(480.0 * charge_scale),
        "documentation_fee": _round2(150.0 * scale),
        "terminal_handling": _round2(350.0 * charge_scale),
        "destination_charges": _round2(450.0 * charge_scale),
        "temperature_control_surcharge": _round2(600.0 * charge_scale),
    }
    total = _round2(sum(quote.values()))
    quote.update(
        {
            "total_cost": total,
            "cost_per_container": _round2(total / containers),
            "currency": "USD",
            "transit_time_days": transit_days,
            "validity_days": 7,
            "carrier_options": [
                "Pacific Star Line",
                "Blue Anchor Shipping",
                "Northline Carriers",
            ],
        }
    )

    email = dedent(
        f"""
        From: purchasing@techcorp-intl.example
        Subject: Urgent - Quote needed for {origin_short} to {dest_short} shipment

        Hi {FORWARDER_NAME} Team,

        We need an urgent quote for the following shipment:
        - Origin: {origin}
        - Destination: {destination}
        - Cargo: Electronics (HS Code: 8471)
        - Container Type: {containers}x {container_type} containers
        - Weight: {weight_per_container:,} kg per container
        - Volume: {volume_per_container} CBM per container
        - Incoterm: FOB {origin_short}
        - Required Delivery: {transit_days - 1}-{transit_days + 2} days
        - Special Requirements: Temperature controlled 20-25C

        Please include:
        - Ocean freight charges
        - Documentation fees
        - Destination charges
        - Insurance options
        - Transit time details

        Need the quote by EOD today.

        Thanks,
        John Smith
        TechCorp Procurement
        """
    ).strip()

    return {
        "scenario_id": f"FCL_{_slug(origin_short)}_{_slug(dest_short)}_{seq:03d}",
        "scenario_type": "FCL_QUOTE",
        "difficulty": difficulty,
        "agent_capabilities": ["email_parsing", "quote_calculation", "fcl_handling"],
        "db_query": {"sender_email": _sender("fcl", seq)},
        "validation_metrics": ValidationMetrics(
            required_fields=[
                "origin",
                "destination",
                "cargo_type",
                "commodity",
                "special_requirements",
            ],
            price_tolerance=0.05,
            transit_time_tolerance=2,
            critical_validations=["origin", "destination", "special_requirements"],
        ).to_dict(),
        "email_content": email,
        "expected_output": {
            "parsed_data": {
                "origin": origin,
                "destination": destination,
                "cargo_type": "FCL",
                "commodity": "Electronics",
                "container_details": {
                    "type": container_type,
                    "weight_per_container": weight_per_container,
                    "volume_per_container": volume_per_container,
                },
                "special_requirements": ["Temperature controlled 20-25C"],
                "required_transit_time": f"{transit_days - 1}-{transit_days + 2} days",
            },
            "calculated_quote": quote,
            "validation_points": {
                "email_classification": "rfq",
                "completeness_score": 0.95,
                "urgency_detected": True,
                "special_handling_identified": True,
            },
        },
    }


_FCL_VARIANTS: List[Dict[str, Any]] = [
    dict(seq=1),
    dict(seq=2, origin="Busan, South Korea (Port)", origin_short="Busan",
         destination="Seattle, WA, USA", dest_short="Seattle",
         scale=0.95, transit_days=14),
    dict(seq=3, origin="Ningbo, China (Port)", origin_short="Ningbo",
         destination="Long Beach, CA, USA", dest_short="Long Beach",
         scale=1.05, transit_days=17, containers=3),
    dict(seq=4, origin="Shenzhen, China (Port)", origin_short="Shenzhen",
         destination="Oakland, CA, USA", dest_short="Oakland",
         scale=1.10, transit_days=18, difficulty="complex"),
    dict(seq=5, origin="Qingdao, China (Port)", origin_short="Qingdao",
         destination="Vancouver, BC, Canada", dest_short="Vancouver",
         scale=0.90, transit_days=15),
    dict(seq=6, origin="Kaohsiung, Taiwan (Port)", origin_short="Kaohsiung",
         destination="Los Angeles, CA, USA", dest_short="LA",
         scale=1.02, transit_days=16),
]


# ---------------------------------------------------------------------------
# LCL quote scenarios
# ---------------------------------------------------------------------------

def build_lcl_scenario(
    seq: int = 1,
    origin: str = "Melbourne, Australia",
    destination: str = "Singapore",
    weight_kg: float = 2500.0,
    volume_cbm: float = 8.0,
    pieces: int = 150,
    transit_days: int = 7,
    scale: float = 1.0,
    difficulty: str = "simple",
) -> Dict[str, Any]:
    """LCL (Less than Container Load) quote scenario with validation."""
    rate_per_cbm = _round2(125.0 * scale)
    quote = {
        "freight_rate_per_cbm": rate_per_cbm,
        "ocean_freight": _round2(rate_per_cbm * volume_cbm),
        "pickup_charges": _round2(250.0 * scale),
        "documentation_fee": _round2(100.0 * scale),
        "terminal_handling": _round2(180.0 * scale),
        "consolidation_fee": _round2(150.0 * scale),
    }
    total = _round2(
        quote["ocean_freight"]
        + quote["pickup_charges"]
        + quote["documentation_fee"]
        + quote["terminal_handling"]
        + quote["consolidation_fee"]
    )
    quote.update(
        {
            "total_cost": total,
            "cost_per_cbm": _round2(total / volume_cbm),
            "cost_per_kg": _round2(total / weight_kg),
            "currency": "USD",
            "transit_time_days": transit_days,
            "validity_days": 14,
            "next_consolidation": "2024-01-15",
        }
    )

    origin_city = origin.split(",")[0]
    email = dedent(
        f"""
        From: logistics@fashionhouse.example
        Subject: LCL Quote Request - {origin_city} to {destination.split(',')[0]}

        Hello,

        Please provide a quote for:
        - From: {origin}
        - To: {destination}
        - Cargo: Fashion accessories
        - Weight: {weight_kg:,.0f} kg
        - Volume: {volume_cbm} CBM
        - Pieces: {pieces} cartons
        - Service: Door to Port
        - Pickup Address: 45 Example Street, {origin_city}

        Standard service is fine, no rush.

        Regards,
        Sarah Lee
        Fashion House Logistics
        """
    ).strip()

    return {
        "scenario_id": f"LCL_{_slug(origin_city)}_{_slug(destination.split(',')[0])}_{seq:03d}",
        "scenario_type": "LCL_QUOTE",
        "difficulty": difficulty,
        "agent_capabilities": ["email_parsing", "quote_calculation", "lcl_consolidation"],
        "db_query": {"sender_email": _sender("lcl", seq)},
        "validation_metrics": ValidationMetrics(
            required_fields=["origin", "destination", "weight_kg", "volume_cbm", "cargo_type"],
            price_tolerance=0.08,
            transit_time_tolerance=3,
            critical_validations=["weight_volume_ratio", "consolidation_feasibility"],
        ).to_dict(),
        "email_content": email,
        "expected_output": {
            "parsed_data": {
                "origin": origin,
                "destination": destination,
                "cargo_type": "LCL",
                "commodity": "Fashion accessories",
                "weight_kg": weight_kg,
                "volume_cbm": volume_cbm,
                "special_requirements": [],
                "container_size": None,
                "required_transit_time": None,
            },
            "calculated_quote": quote,
            "validation_points": {
                "email_classification": "rfq",
                "completeness_score": 0.90,
                "urgency_detected": False,
                "consolidation_recommended": True,
            },
        },
    }


_LCL_VARIANTS: List[Dict[str, Any]] = [
    dict(seq=1),
    dict(seq=2, origin="Sydney, Australia", destination="Hong Kong",
         weight_kg=1800.0, volume_cbm=6.5, pieces=110, scale=1.10, transit_days=9),
    dict(seq=3, origin="Brisbane, Australia", destination="Port Klang, Malaysia",
         weight_kg=3200.0, volume_cbm=10.0, pieces=200, scale=0.95, transit_days=11),
    dict(seq=4, origin="Auckland, New Zealand", destination="Singapore",
         weight_kg=900.0, volume_cbm=3.2, pieces=40, scale=1.05, transit_days=12),
    dict(seq=5, origin="Melbourne, Australia", destination="Jakarta, Indonesia",
         weight_kg=2100.0, volume_cbm=7.4, pieces=130, scale=1.0, transit_days=10),
    dict(seq=6, origin="Perth, Australia", destination="Colombo, Sri Lanka",
         weight_kg=2750.0, volume_cbm=9.1, pieces=160, scale=1.15, transit_days=14,
         difficulty="moderate"),
]


# ---------------------------------------------------------------------------
# Air freight scenarios
# ---------------------------------------------------------------------------

def build_air_scenario(
    seq: int = 1,
    origin: str = "JFK Airport, New York",
    destination: str = "Frankfurt Airport, Germany",
    origin_code: str = "JFK",
    dest_code: str = "FRA",
    weight_kg: float = 45.0,
    transit_days: int = 2,
    scale: float = 1.0,
    difficulty: str = "complex",
) -> Dict[str, Any]:
    """Air freight quote scenario with dangerous-goods validation."""
    rate = _round2(12.5 * scale)
    air_freight = _round2(rate * weight_kg)
    quote = {
        "air_freight_rate": rate,
        "air_freight": air_freight,
        "chargeable_weight": weight_kg,
        "fuel_surcharge": _round2(air_freight * 0.15),
        "security_fee": _round2(75.0 * scale),
        "dangerous_goods_fee": _round2(350.0 * scale),
        "temperature_control_fee": _round2(450.0 * scale),
        "pickup_charges": _round2(150.0 * scale),
        "delivery_charges": _round2(180.0 * scale),
        "customs_clearance": _round2(250.0 * scale),
        "documentation_fee": _round2(150.0 * scale),
        "insurance_premium": _round2(250.0 * scale),
    }
    charge_keys = [k for k in quote if k not in ("air_freight_rate", "chargeable_weight")]
    total = _round2(sum(quote[k] for k in charge_keys))
    quote.update(
        {
            "total_cost": total,
            "currency": "USD",
            "transit_time_days": transit_days,
            "validity_days": 3,
        }
    )

    email = dedent(
        f"""
        From: shipping@pharmatech.example
        Subject: URGENT - Air Freight Quote Required - Pharma Samples

        Dear {FORWARDER_NAME},

        We need an immediate air freight quote for pharmaceutical samples:

        Shipment Details:
        - Origin: {origin}
        - Destination: {destination}
        - Product: Pharmaceutical samples (UN3373 - Biological Substance Cat B)
        - Weight: {weight_kg:g} kg gross
        - Dimensions: 2 boxes, each 60x40x40 cm
        - Temperature: Must maintain 2-8C
        - Service Required: Express (24-48 hours door-to-door)
        - Collection: From our facility near {origin_code}
        - Delivery: To the research center near {dest_code}

        Requirements:
        - GDP compliant handling
        - Real-time temperature monitoring
        - Insurance for $50,000 value
        - All customs documentation

        This is time-critical for clinical trials.

        Best regards,
        Dr. Michael Chen
        PharmaTech Logistics Manager
        +1-555-0123
        """
    ).strip()

    return {
        "scenario_id": f"AIR_{origin_code}_{dest_code}_{seq:03d}",
        "scenario_type": "AIR_FREIGHT",
        "difficulty": difficulty,
        "agent_capabilities": ["email_parsing", "air_freight_calculation", "dangerous_goods"],
        "db_query": {"sender_email": _sender("air", seq)},
        "validation_metrics": ValidationMetrics(
            required_fields=["origin", "destination", "weight_kg", "cargo_type", "commodity"],
            price_tolerance=0.30,
            transit_time_tolerance=1,
            critical_validations=["dangerous_goods_compliance", "iata_regulations"],
        ).to_dict(),
        "email_content": email,
        "expected_output": {
            "parsed_data": {
                "origin": origin,
                "destination": destination,
                "cargo_type": "Air",
                "commodity": "Pharmaceutical samples (UN3373 - Biological Substance Cat B)",
                "weight_kg": weight_kg,
                "volume_cbm": None,
                "special_requirements": [
                    "GDP compliant handling",
                    "Real-time temperature monitoring",
                    "Insurance for $50,000 value",
                    "All customs documentation",
                ],
            },
            "calculated_quote": quote,
            "validation_points": {
                "dangerous_goods_identified": True,
                "temperature_control_required": True,
            },
        },
    }


_AIR_VARIANTS: List[Dict[str, Any]] = [
    dict(seq=1),
    dict(seq=2, origin="O'Hare Airport, Chicago", destination="Narita Airport, Tokyo",
         origin_code="ORD", dest_code="NRT", weight_kg=60.0, scale=1.20, transit_days=3),
    dict(seq=3, origin="LAX Airport, Los Angeles", destination="Incheon Airport, Seoul",
         origin_code="LAX", dest_code="ICN", weight_kg=38.0, scale=0.90, transit_days=2),
    dict(seq=4, origin="Changi Airport, Singapore", destination="Schiphol Airport, Amsterdam",
         origin_code="SIN", dest_code="AMS", weight_kg=52.0, scale=1.05, transit_days=2),
    dict(seq=5, origin="Sydney Airport, Australia", destination="LAX Airport, Los Angeles",
         origin_code="SYD", dest_code="LAX", weight_kg=47.0, scale=1.10, transit_days=2),
    dict(seq=6, origin="Pearson Airport, Toronto", destination="Heathrow Airport, London",
         origin_code="YYZ", dest_code="LHR", weight_kg=41.0, scale=0.95, transit_days=1),
]


# ---------------------------------------------------------------------------
# Multimodal scenarios
# ---------------------------------------------------------------------------

def build_multimodal_scenario(
    seq: int = 1,
    origin: str = "Dubai, UAE",
    origin_detail: str = "our warehouse in the Jebel Ali Free Zone",
    destination: str = "Montreal, Quebec, Canada",
    weight_kg: float = 12500.0,
    cargo_value_usd: int = 850000,
    transit_days: int = 35,
    scale: float = 1.0,
    difficulty: str = "expert",
) -> Dict[str, Any]:
    """Complex multimodal shipment scenario with multiple validation points."""
    quote = {
        "ocean_freight": _round2(4500.0 * scale),
        "heavy_lift_handling": _round2(2800.0 * scale),
        "export_customs": _round2(200.0 * scale),
        "import_customs_clearance": _round2(350.0 * scale),
        "inland_trucking": _round2(1200.0 * scale),
        "insurance_premium": _round2(1550.0 * scale),
    }
    total = _round2(sum(quote.values()))
    quote.update(
        {
            "total_cost": total,
            "currency": "USD",
            "transit_time_days": transit_days,
        }
    )

    origin_city = origin.split(",")[0]
    dest_city = destination.split(",")[0]
    email = dedent(
        f"""
        From: procurement@globalmanufacturing.example
        Subject: Multimodal Quote - {origin_city} to {dest_city} - Heavy Machinery

        Hello {FORWARDER_NAME} Team,

        We need a comprehensive quote for shipping heavy machinery from our
        {origin_city} facility to {destination}.

        Shipment Information:
        - Origin: {origin} ({origin_detail})
        - Final Destination: {destination}
        - Cargo Description: Industrial printing machinery
        - Total Weight: {weight_kg:,.0f} kg
        - Dimensions:
          * Main unit: 4.5m x 2.8m x 2.2m
          * Auxiliary unit: 2.5m x 2.0m x 1.8m
        - Value: USD {cargo_value_usd:,}
        - HS Code: 8443.11

        Requirements:
        - Suggest best routing (sea+truck or sea+rail)
        - Need delivery to customer site in the {dest_city} industrial area
        - Machinery needs to stay upright, cannot be tilted
        - Requires crane for loading/unloading
        - Full insurance coverage
        - Import customs clearance at destination
        - Delivery window: {transit_days - 5}-{transit_days + 5} days acceptable

        Open to alternative suggestions for cost optimization.
        Please include a breakdown of all costs including customs duties estimate.

        Thanks,
        Ahmed Hassan
        Global Manufacturing FZE
        procurement@globalmanufacturing.example
        """
    ).strip()

    return {
        "scenario_id": f"MULTI_{_slug(origin_city)}_{_slug(dest_city)}_{seq:03d}",
        "scenario_type": "MULTIMODAL",
        "difficulty": difficulty,
        "agent_capabilities": [
            "email_parsing",
            "multimodal_routing",
            "customs_expertise",
            "rate_optimization",
        ],
        "db_query": {"sender_email": _sender("multimodal", seq)},
        "validation_metrics": ValidationMetrics(
            required_fields=["origin", "destination", "weight_kg", "cargo_type", "commodity"],
            price_tolerance=0.60,
            transit_time_tolerance=5,
            critical_validations=["routing_optimization", "customs_compliance", "mode_selection"],
        ).to_dict(),
        "email_content": email,
        "expected_output": {
            "parsed_data": {
                "origin": origin,
                "destination": destination,
                "cargo_type": "MULTIMODAL",
                "commodity": "Industrial printing machinery",
                "weight_kg": weight_kg,
                "volume_cbm": None,
                "special_requirements": [
                    "Keep upright - no tilting",
                    "Crane required for handling",
                    "Full insurance coverage",
                    "Import customs clearance",
                ],
            },
            "calculated_quote": quote,
            "validation_points": {
                "routing_optimization_performed": True,
                "customs_requirements_identified": True,
            },
        },
    }


_MULTIMODAL_VARIANTS: List[Dict[str, Any]] = [
    dict(seq=1),
    dict(seq=2, origin="Istanbul, Turkiye", origin_detail="our plant near the port",
         destination="Chicago, IL, USA", weight_kg=9800.0, cargo_value_usd=620000,
         scale=0.90, transit_days=32),
    dict(seq=3, origin="Mumbai, India", origin_detail="our Nhava Sheva staging yard",
         destination="Toronto, Ontario, Canada", weight_kg=14200.0, cargo_value_usd=910000,
         scale=1.10, transit_days=38),
    dict(seq=4, origin="Rotterdam, Netherlands", origin_detail="our bonded warehouse",
         destination="Kansas City, MO, USA", weight_kg=8600.0, cargo_value_usd=540000,
         scale=0.85, transit_days=28, difficulty="complex"),
    dict(seq=5, origin="Jeddah, Saudi Arabia", origin_detail="our export terminal",
         destination="Sao Paulo, Brazil", weight_kg=11000.0, cargo_value_usd=760000,
         scale=1.05, transit_days=40),
    dict(seq=6, origin="Ho Chi Minh City, Vietnam", origin_detail="our assembly plant",
         destination="Calgary, Alberta, Canada", weight_kg=10400.0, cargo_value_usd=690000,
         scale=1.0, transit_days=36),
]


# ---------------------------------------------------------------------------
# Customer matching scenarios
# ---------------------------------------------------------------------------

def build_customer_matching_scenario(
    seq: int = 1,
    fixture_index: int = 0,
    sender_email: str = "jdoe@techcorp-intl.example",
    sender_name: str = "John Doe",
    signature_company: str = "Tech Corp International",
    expected_method: str = "domain_match",
    expected_confidence: float = 0.9,
    difficulty: str = "moderate",
) -> Dict[str, Any]:
    """Customer matching and identification scenario.

    The expected customer comes from :data:`CUSTOMER_FIXTURES`, which the demo
    also uses to seed the in-memory customer directory - so a correct matching
    agent scores 100% on these scenarios.
    """
    fixture = CUSTOMER_FIXTURES[fixture_index]
    email = dedent(
        f"""
        From: {sender_email}
        Subject: Quote Request - Urgent Shipment

        Hi there,

        {sender_name.split()[0]} from {signature_company} here. We've worked
        together before on several shipments.

        Need a quote for:
        - Shanghai to Los Angeles
        - 1x 20GP container
        - Electronics
        - Need it ASAP

        Please use our standard payment terms.

        {sender_name}
        {fixture["company_name"]}
        """
    ).strip()

    return {
        "scenario_id": f"CUST_MATCH_{seq:03d}",
        "scenario_type": "CUSTOMER_MATCHING",
        "difficulty": difficulty,
        "agent_capabilities": ["customer_identification", "database_search", "fuzzy_matching"],
        "db_query": {"sender_email": _sender("customer", seq)},
        "validation_metrics": {
            "required_fields": ["customer_id", "identified_customer"],
            "confidence_threshold": 0.85,
            "critical_validations": ["customer_verification", "duplicate_check"],
        },
        "email_content": email,
        "expected_output": {
            "customer_match": {
                "identified_customer": fixture["company_name"],
                "customer_id": fixture["customer_id"],
                "confidence_score": expected_confidence,
                "match_method": expected_method,
            },
            "parsed_data": {
                "identified_customer": fixture["company_name"],
                "customer_id": fixture["customer_id"],
                "customer_confidence": expected_confidence,
                "match_method": expected_method,
            },
            "validation_points": {
                "customer_verified": True,
                "payment_terms_found": True,
                "contact_verified": True,
            },
        },
    }


_CUSTOMER_VARIANTS: List[Dict[str, Any]] = [
    dict(seq=1),
    # Sender address equals the directory contact email -> exact match.
    dict(seq=2, fixture_index=1, sender_email="ops@northwind-traders.example",
         sender_name="Marcus Bell", signature_company="Northwind Traders",
         expected_method="exact_match", expected_confidence=1.0, difficulty="simple"),
    dict(seq=3, fixture_index=2, sender_email="ytanaka@osaka-textiles.example",
         sender_name="Yuki Tanaka", signature_company="Osaka Textile Works"),
    dict(seq=4, fixture_index=3, sender_email="cmendoza@andes-mining.example",
         sender_name="Carla Mendoza", signature_company="Andes Mining Supply"),
    dict(seq=5, fixture_index=4, sender_email="akask@baltic-furniture.example",
         sender_name="Anders Kask", signature_company="Baltic Furniture"),
    dict(seq=6, fixture_index=5, sender_email="felamrani@sahara-agri.example",
         sender_name="Fatima El Amrani", signature_company="Sahara AgriExports"),
]


# ---------------------------------------------------------------------------
# Booking confirmation scenarios
# ---------------------------------------------------------------------------

def build_booking_scenario(
    seq: int = 1,
    booking_reference: str = "BKG2024010234",
    carrier: str = "Pacific Star Line",
    vessel_name: str = "MV PACIFIC STAR",
    voyage_number: str = "2401W",
    etd: str = "2024-01-25",
    eta: str = "2024-02-10",
    origin_port: str = "Shanghai",
    destination_port: str = "Los Angeles",
    container_type: str = "40HC",
    container_quantity: int = 1,
    cutoff: str = "2024-01-23 14:00",
    difficulty: str = "simple",
) -> Dict[str, Any]:
    """Booking confirmation processing scenario."""
    email = dedent(
        f"""
        From: bookings@carrier-line.example
        Subject: Booking Confirmation - {booking_reference}

        Dear Customer,

        Your booking has been confirmed:

        Booking Reference: {booking_reference}
        Carrier: {carrier}
        Vessel: {vessel_name}
        Voyage: {voyage_number}
        ETD {origin_port}: {etd}
        ETA {destination_port}: {eta}
        Container: {container_quantity}x{container_type}
        Cut-off: {cutoff}

        Please ensure cargo delivery before cut-off time.

        Best regards,
        Carrier Line Bookings
        """
    ).strip()

    return {
        "scenario_id": f"BOOKING_CONF_{seq:03d}",
        "scenario_type": "BOOKING_CONFIRMATION",
        "difficulty": difficulty,
        "agent_capabilities": ["booking_extraction", "reference_parsing", "date_extraction"],
        "db_query": {"sender_email": _sender("booking", seq)},
        "validation_metrics": {
            "required_fields": ["booking_reference", "vessel_name", "etd", "eta"],
            "date_format_check": True,
            "critical_validations": ["booking_number_format", "vessel_validation"],
        },
        "email_content": email,
        "expected_output": {
            "booking_data": {
                "booking_reference": booking_reference,
                "carrier": carrier,
                "vessel_name": vessel_name,
                "voyage_number": voyage_number,
                "etd": etd,
                "eta": eta,
                "origin_port": origin_port,
                "destination_port": destination_port,
                "container_type": container_type,
                "container_quantity": container_quantity,
                "container_number": None,
            },
            "validation_points": {
                "booking_format_valid": True,
                "dates_parsed": True,
                "vessel_verified": True,
            },
        },
    }


_BOOKING_VARIANTS: List[Dict[str, Any]] = [
    dict(seq=1),
    dict(seq=2, booking_reference="BKG2024020871", carrier="Blue Anchor Shipping",
         vessel_name="MV BLUE ANCHOR", voyage_number="2407E", etd="2024-02-14",
         eta="2024-03-02", origin_port="Busan", destination_port="Long Beach",
         container_type="20GP", container_quantity=2, cutoff="2024-02-12 12:00"),
    dict(seq=3, booking_reference="BKG2024031145", carrier="Northline Carriers",
         vessel_name="MV NORTH DAWN", voyage_number="2412N", etd="2024-03-05",
         eta="2024-03-19", origin_port="Ningbo", destination_port="Seattle",
         container_type="40GP", container_quantity=1, cutoff="2024-03-03 16:00"),
    dict(seq=4, booking_reference="BKG2024040992", carrier="Pacific Star Line",
         vessel_name="MV CORAL WIND", voyage_number="2418S", etd="2024-04-11",
         eta="2024-05-01", origin_port="Singapore", destination_port="Rotterdam",
         container_type="40HC", container_quantity=3, cutoff="2024-04-09 10:00",
         difficulty="moderate"),
    dict(seq=5, booking_reference="BKG2024051267", carrier="Blue Anchor Shipping",
         vessel_name="MV HARBOUR LIGHT", voyage_number="2423W", etd="2024-05-20",
         eta="2024-06-06", origin_port="Melbourne", destination_port="Singapore",
         container_type="20GP", container_quantity=1, cutoff="2024-05-18 14:00"),
    dict(seq=6, booking_reference="BKG2024060455", carrier="Northline Carriers",
         vessel_name="MV SILVER GULL", voyage_number="2430E", etd="2024-06-15",
         eta="2024-07-08", origin_port="Mumbai", destination_port="Felixstowe",
         container_type="40HC", container_quantity=2, cutoff="2024-06-13 09:00"),
]


# ---------------------------------------------------------------------------
# Suite assembly and registry
# ---------------------------------------------------------------------------

# In-memory registry for dynamically generated scenarios. The production
# system persisted these to a JSON registry file; the showcase keeps them
# in-process for a dependency-free, side-effect-free test run.
_GENERATED_SCENARIOS: List[Dict[str, Any]] = []


class GroundTruthTestData:
    """Repository of ground-truth test scenarios with validation metrics."""

    @staticmethod
    def get_all_scenarios(include_generated: bool = True) -> List[Dict[str, Any]]:
        """Return all 36 built-in scenarios (plus any registered ones)."""
        scenarios: List[Dict[str, Any]] = []
        for params in _FCL_VARIANTS:
            scenarios.append(build_fcl_scenario(**params))
        for params in _LCL_VARIANTS:
            scenarios.append(build_lcl_scenario(**params))
        for params in _AIR_VARIANTS:
            scenarios.append(build_air_scenario(**params))
        for params in _MULTIMODAL_VARIANTS:
            scenarios.append(build_multimodal_scenario(**params))
        for params in _CUSTOMER_VARIANTS:
            scenarios.append(build_customer_matching_scenario(**params))
        for params in _BOOKING_VARIANTS:
            scenarios.append(build_booking_scenario(**params))
        if include_generated:
            scenarios.extend(_GENERATED_SCENARIOS)
        return scenarios

    @staticmethod
    def get_scenarios_by_type(scenario_type: str) -> List[Dict[str, Any]]:
        """All scenarios of a given type (e.g. ``FCL_QUOTE``, ``CUSTOMER_MATCHING``)."""
        wanted = scenario_type.upper()
        return [
            s
            for s in GroundTruthTestData.get_all_scenarios()
            if wanted in s["scenario_type"].upper()
        ]

    @staticmethod
    def get_scenario_by_id(scenario_id: str) -> Optional[Dict[str, Any]]:
        for scenario in GroundTruthTestData.get_all_scenarios():
            if scenario.get("scenario_id") == scenario_id:
                return scenario
        return None

    @staticmethod
    def list_scenario_ids() -> List[str]:
        return [s["scenario_id"] for s in GroundTruthTestData.get_all_scenarios()]

    @staticmethod
    def get_scenarios_by_capability(capability: str) -> List[Dict[str, Any]]:
        return [
            s
            for s in GroundTruthTestData.get_all_scenarios()
            if capability.lower() in [c.lower() for c in s.get("agent_capabilities", [])]
        ]

    @staticmethod
    def register_scenario(scenario: Dict[str, Any]) -> str:
        """Register a dynamically generated scenario (in-memory).

        Duplicate IDs get a numeric suffix, mirroring the production registry.
        """
        existing_ids = set(GroundTruthTestData.list_scenario_ids())
        scenario_id = scenario.get("scenario_id", "GENERATED_001")
        if scenario_id in existing_ids:
            base_id = scenario_id.rsplit("_", 1)[0]
            counter = len([s for s in existing_ids if s.startswith(base_id)]) + 1
            scenario_id = f"{base_id}_{counter:03d}"
            scenario["scenario_id"] = scenario_id
        scenario.setdefault(
            "_generation_metadata",
            {"registered_at": datetime.now().isoformat(), "source": "manual_registration"},
        )
        _GENERATED_SCENARIOS.append(scenario)
        return scenario_id

    @staticmethod
    def unregister_scenario(scenario_id: str) -> bool:
        for i, scenario in enumerate(_GENERATED_SCENARIOS):
            if scenario.get("scenario_id") == scenario_id:
                del _GENERATED_SCENARIOS[i]
                return True
        return False

    @staticmethod
    def clear_generated_scenarios() -> None:
        _GENERATED_SCENARIOS.clear()
