"""Customer matching agent - the example agent under TDAE evaluation.

Given an inbound email, the agent:

1. extracts the sender's company name via a three-stage pipeline
   (LLM -> regex fallback -> sender-domain fallback),
2. matches it against a customer directory using layered strategies
   (exact contact email -> email domain -> fuzzy company name), and
3. emits a structured match result that TDAE scores against ground truth.

The production version queried a customers table through an ORM session and
called a vendor LLM SDK directly. The showcase swaps those for an in-memory
:class:`CustomerDirectory` and the :class:`~tdae.providers.LLMProvider`
protocol so the whole pipeline runs offline and deterministically.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tdae.providers import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Customer directory (replaces the production customers table)
# ---------------------------------------------------------------------------

@dataclass
class CustomerRecord:
    """One known customer in the directory."""

    customer_id: str
    company_name: str
    email_domain: str = ""
    contact_name: str = ""
    contact_email: str = ""
    phone: str = ""
    is_active: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CustomerDirectory:
    """In-memory customer store with the query surface the agent needs."""

    def __init__(self, records: Iterable[CustomerRecord] = ()):
        self._records: List[CustomerRecord] = list(records)

    @classmethod
    def from_fixtures(cls, fixtures: Iterable[Dict[str, Any]]) -> "CustomerDirectory":
        """Build a directory from plain dicts (e.g. ``tdae.ground_truth.CUSTOMER_FIXTURES``)."""
        return cls(
            CustomerRecord(
                customer_id=f["customer_id"],
                company_name=f["company_name"],
                email_domain=f.get("email_domain", ""),
                contact_name=f.get("contact_name", ""),
                contact_email=f.get("contact_email", ""),
                phone=f.get("phone", ""),
            )
            for f in fixtures
        )

    def add(self, record: CustomerRecord) -> None:
        self._records.append(record)

    def active_customers(self) -> List[CustomerRecord]:
        return [r for r in self._records if r.is_active]

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_TEMPLATE = """Extract the company name from this email. Return ONLY the company name, nothing else.

RULES:
- Return just the company entity name (e.g., "Global Shipping Solutions Ltd")
- Do NOT include descriptive text or sentences
- Do NOT include email addresses or contact info
- Include legal suffixes (Ltd, Inc, Corp, LLC, GmbH, etc.)
- Capitalize properly (Title Case for most, preserve acronyms)

EXAMPLES:

Input: "We are writing from ABC Freight Corp to request a quote..."
Output: ABC Freight Corp

Input: "John Smith\\nOperations Manager\\nGlobal Logistics Ltd\\njohn@globallogistics.example"
Output: Global Logistics Ltd

Input: "Thanks,\\nDavid Lee\\nSenior Freight Coordinator"
Output: NOT_FOUND

EMAIL TO ANALYZE:
{email_content}

Sender Email: {sender_email}

Return only the company name or "NOT_FOUND" if no company is mentioned."""


class CustomerMatchingAgent:
    """LLM-assisted customer identification agent."""

    name = "customer_matching"

    def __init__(self, provider: LLMProvider, directory: CustomerDirectory):
        self.provider = provider
        self.directory = directory

    # -- stage 1: company extraction -------------------------------------------

    def extract_company_info(self, email_content: str, sender_email: str = "") -> Dict[str, Any]:
        """Extract company information via LLM with regex and domain fallbacks."""
        domain = None
        if sender_email and "@" in sender_email:
            domain = sender_email.split("@")[1].lower()

        company_name: Optional[str] = None
        confidence = 0.0
        extraction_method = "none"

        # Stage 1: LLM extraction (highest accuracy).
        company_name, confidence = self._extract_company_with_llm(email_content, sender_email)
        if company_name and confidence >= 0.6:
            extraction_method = "llm"
        else:
            # Stage 2: regex fallback.
            regex_name, regex_confidence = self._extract_company_with_regex(email_content)
            if regex_confidence > confidence:
                company_name = regex_name
                confidence = regex_confidence
                extraction_method = "regex"

        # Stage 3: last resort - derive a name from the sender domain.
        if not company_name and domain:
            domain_name = re.sub(r"\.(example|test|com|org|net|co|au|uk)$", "", domain)
            domain_name = domain_name.split(".")[-1]
            company_name = domain_name.replace("-", " ").replace("_", " ").title()
            confidence = 0.35
            extraction_method = "domain"

        return {
            "success": True,
            "company_name": company_name or "",
            "domain": domain or "",
            "confidence": round(confidence, 2),
            "extraction_method": extraction_method,
            "email_addresses": [sender_email] if sender_email else [],
        }

    def _extract_company_with_llm(
        self, email_content: str, sender_email: str
    ) -> Tuple[Optional[str], float]:
        """LLM-based company name extraction."""
        content_preview = email_content[:1500]
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            email_content=content_preview, sender_email=sender_email
        )

        try:
            extracted = self.provider.complete(prompt, max_tokens=100, temperature=0.3).strip()
        except Exception as exc:
            logger.error("LLM extraction failed: %s", exc)
            return None, 0.0

        if not extracted or extracted == "NOT_FOUND" or len(extracted) < 2:
            return None, 0.0

        cleaned = self._clean_company_name(extracted)
        confidence = self._calculate_extraction_confidence(cleaned, email_content)
        return cleaned, confidence

    def _extract_company_with_regex(self, email_content: str) -> Tuple[Optional[str], float]:
        """Regex-based fallback when LLM extraction fails or is low-confidence."""
        # Pattern 1: line containing a legal suffix (most reliable).
        pattern1 = (
            r"([A-Z][A-Za-z0-9\s&\-\.\']+"
            r"(?:Ltd|Inc|Corp|LLC|Limited|Corporation|GmbH|PLC|KK|OU|Co\.?)"
            r"(?:\s*\([A-Za-z\s]+\))?\.?)"
        )
        matches = re.findall(pattern1, email_content, re.MULTILINE)
        if matches:
            company_name = min(matches, key=len).strip()  # shortest = most precise
            cleaned = self._clean_company_name(company_name)
            if len(cleaned) >= 5:
                return cleaned, 0.65

        # Pattern 2: capitalized multi-word line near the signature.
        signature_area = email_content[-300:]
        matches = re.findall(r"\n([A-Z][A-Za-z0-9\s&\-\.]{8,50})\n", signature_area)
        candidates = [m.strip() for m in matches if " " in m.strip() and len(m.strip()) >= 8]
        if candidates:
            cleaned = self._clean_company_name(candidates[0])
            if len(cleaned.split()) >= 2:
                return cleaned, 0.55

        # Pattern 3: "From: Company Name" style mention in the body.
        matches = re.findall(
            r"(?:from|From|FROM)[\s:]+([A-Z][A-Za-z0-9\s&\-\.]{5,50})(?:\n|,|\.)",
            email_content,
        )
        for match in matches:
            cleaned = self._clean_company_name(match)
            if len(cleaned.split()) > 2 or any(
                kw in cleaned.lower()
                for kw in ["shipping", "logistics", "freight", "corp", "company", "ltd", "inc"]
            ):
                return cleaned, 0.5

        return None, 0.0

    @staticmethod
    def _clean_company_name(name: str) -> str:
        name = name.strip("\"'").strip()
        name = re.sub(r"\s+", " ", name)
        name = re.sub(r"[,;:!?]+$", "", name)
        if name.isupper() and len(name) > 10:
            name = CustomerMatchingAgent._smart_title_case(name)
        return name

    @staticmethod
    def _smart_title_case(name: str) -> str:
        """Title-case a shouting company name while preserving legal acronyms."""
        acronyms = {
            "LLC", "INC", "LTD", "USA", "UK", "UAE", "GMBH", "PLC",
            "CORP", "CO", "SA", "AG", "NV", "BV", "SPA", "SARL",
            "PTY", "PVT", "SDN", "BHD", "AB", "AS", "OY", "KFT", "KK", "OU",
        }
        words = []
        for word in name.split():
            if word.upper() in acronyms:
                words.append(word.upper())
            else:
                words.append(word.title())
        return " ".join(words)

    @staticmethod
    def _calculate_extraction_confidence(company_name: str, email_content: str) -> float:
        """Heuristic confidence score for an extracted company name."""
        confidence = 0.5  # base confidence for LLM extraction

        occurrences = email_content.lower().count(company_name.lower())
        if occurrences >= 3:
            confidence += 0.25
        elif occurrences >= 2:
            confidence += 0.15
        elif occurrences >= 1:
            confidence += 0.05

        legal_suffixes = [
            "Ltd", "Inc", "Corp", "LLC", "Limited", "Corporation",
            "GmbH", "PLC", "Co", "SA", "AG", "NV", "BV", "Spa",
            "Pty", "Pvt", "Sdn", "Bhd", "KK", "OU",
        ]
        if any(suffix in company_name for suffix in legal_suffixes):
            confidence += 0.15

        if 3 <= len(company_name) <= 60:
            confidence += 0.1
        if len(company_name) < 3:
            confidence -= 0.3
        if len(company_name) > 70:
            confidence -= 0.2

        business_keywords = [
            "shipping", "logistics", "freight", "transport",
            "cargo", "trading", "international", "global",
        ]
        if any(keyword in company_name.lower() for keyword in business_keywords):
            confidence += 0.05

        return min(max(confidence, 0.0), 1.0)

    # -- stage 2: customer matching ----------------------------------------------

    def match_customer_by_context(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """Match the extracted company against the customer directory.

        Strategies, in priority order:
        1. exact contact-email match (confidence 1.0)
        2. email-domain match (confidence 0.9)
        3. fuzzy company-name match (confidence = similarity ratio, >= 0.6)
        """
        company_name = company_info.get("company_name", "")
        domain = company_info.get("domain", "")
        email_addresses = company_info.get("email_addresses", [])
        sender_email = email_addresses[0] if email_addresses else ""

        if not domain and sender_email and "@" in sender_email:
            domain = sender_email.split("@")[1].lower()

        primary_match: Optional[Dict[str, Any]] = None
        alternative_matches: List[Dict[str, Any]] = []
        confidence_score = 0.0
        matching_strategy = "none"
        matching_reasoning = ""

        customers = self.directory.active_customers()

        # Priority 1: exact contact-email match.
        if sender_email:
            for customer in customers:
                if customer.contact_email and customer.contact_email.lower() == sender_email.lower():
                    primary_match = self._build_customer_match(customer, 1.0, "exact_match")
                    confidence_score = 1.0
                    matching_strategy = "exact_match"
                    matching_reasoning = f"Exact email match: {sender_email}"
                    break

        # Priority 2: email-domain match.
        if not primary_match and domain:
            domain_matches = []
            for customer in customers:
                customer_domain = customer.email_domain
                if not customer_domain and customer.contact_email and "@" in customer.contact_email:
                    customer_domain = customer.contact_email.split("@")[1].lower()
                if customer_domain and customer_domain.lower() == domain.lower():
                    domain_matches.append(customer)

            if domain_matches:
                primary_match = self._build_customer_match(domain_matches[0], 0.9, "domain_match")
                confidence_score = 0.9
                matching_strategy = "domain_match"
                matching_reasoning = f"Domain match: {domain}"
                for customer in domain_matches[1:]:
                    alternative_matches.append(
                        self._build_customer_match(customer, 0.85, "domain_match")
                    )

        # Priority 3: fuzzy company-name match.
        if not primary_match and company_name:
            name_matches: List[Tuple[CustomerRecord, float]] = []
            for customer in customers:
                similarity = SequenceMatcher(
                    None,
                    company_name.lower().strip(),
                    (customer.company_name or "").lower().strip(),
                ).ratio()
                if similarity >= 0.6:
                    name_matches.append((customer, similarity))

            name_matches.sort(key=lambda pair: pair[1], reverse=True)
            if name_matches:
                best_customer, best_score = name_matches[0]
                primary_match = self._build_customer_match(best_customer, best_score, "fuzzy_match")
                confidence_score = best_score
                matching_strategy = "fuzzy_match"
                matching_reasoning = (
                    f"Name similarity match: {best_score:.0%} similar to '{company_name}'"
                )
                for customer, score in name_matches[1:5]:
                    alternative_matches.append(
                        self._build_customer_match(customer, score, "fuzzy_match")
                    )

        if confidence_score >= 0.95:
            confidence_level = "exact"
        elif confidence_score >= 0.80:
            confidence_level = "high"
        elif confidence_score >= 0.60:
            confidence_level = "medium"
        elif confidence_score >= 0.40:
            confidence_level = "low"
        else:
            confidence_level = "none"

        return {
            "success": primary_match is not None,
            "primary_match": primary_match,
            "alternative_matches": alternative_matches,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "matching_strategy": matching_strategy,
            "matching_reasoning": matching_reasoning or "No matching customer found in directory",
            "requires_human_review": confidence_score < 0.7,
        }

    @staticmethod
    def _build_customer_match(
        customer: CustomerRecord, score: float, strategy: str
    ) -> Dict[str, Any]:
        return {
            "customer_id": customer.customer_id,
            "customer_name": customer.company_name or customer.contact_email or "Unknown",
            "match_score": score,
            "matching_strategies": [strategy],
            "customer_metadata": {
                "contact_name": customer.contact_name,
                "contact_email": customer.contact_email,
                "email_domain": customer.email_domain,
                "phone": customer.phone,
                "is_active": customer.is_active,
            },
        }

    # -- validation ------------------------------------------------------------

    def validate_customer_match(self, match_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a match result against the confidence threshold."""
        confidence_score = match_result.get("confidence_score", 0.0)
        primary = match_result.get("primary_match") or {}
        customer_id = primary.get("customer_id")

        threshold = 0.5
        is_valid = confidence_score > threshold and customer_id is not None

        recommendations: List[str] = []
        if confidence_score < 0.7:
            recommendations.append("Consider manual review due to low confidence")
        if not customer_id:
            recommendations.append("No customer match found - consider creating a new customer")

        if is_valid:
            reasoning = (
                f"Match validation passed: confidence score {confidence_score:.2f} "
                f"exceeds threshold {threshold:.2f} and customer ID is present."
            )
        else:
            reasons = []
            if confidence_score <= threshold:
                reasons.append(
                    f"confidence score {confidence_score:.2f} is below threshold {threshold:.2f}"
                )
            if not customer_id:
                reasons.append("customer ID is missing")
            reasoning = f"Match validation failed: {', '.join(reasons)}."

        return {
            "is_valid": is_valid,
            "validation_score": confidence_score,
            "confidence_meets_threshold": confidence_score > threshold,
            "validation_reasoning": reasoning,
            "recommendations": recommendations,
        }

    # -- TDAE adapter -------------------------------------------------------------

    def run(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """ScenarioAgent entry point: process one ground-truth scenario."""
        email_content = scenario["email_content"]

        sender_match = re.search(r"From:\s*([^\s<>]+@[^\s<>]+)", email_content)
        sender_email = sender_match.group(1) if sender_match else ""

        company_info = self.extract_company_info(email_content, sender_email)
        match_result = self.match_customer_by_context(company_info)
        validation = self.validate_customer_match(match_result)

        primary = match_result.get("primary_match") or {}
        identified_customer = primary.get("customer_name")
        customer_id = primary.get("customer_id")
        confidence = match_result.get("confidence_score", 0.0)
        method = match_result.get("matching_strategy", "none")

        return {
            "customer_match": {
                "identified_customer": identified_customer,
                "customer_id": customer_id,
                "confidence_score": confidence,
                "match_method": method,
            },
            "parsed_data": {
                "identified_customer": identified_customer,
                "customer_id": customer_id,
                "customer_confidence": confidence,
                "match_method": method,
            },
            "validation_points": {
                "customer_verified": validation["is_valid"],
                "payment_terms_found": "payment terms" in email_content.lower(),
                "contact_verified": bool(sender_email) and match_result["success"],
            },
        }
