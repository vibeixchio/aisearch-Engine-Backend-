"""
PII redaction layer with India-specific patterns.
Detects and redacts Aadhaar, PAN, Indian mobile, IFSC, GSTIN, SSN, credit card, email.
"""
import re
from fastapi import Request

# ---------------------------------------------------------------------------
# PII Patterns — India-focused + Global
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    # Indian patterns
    "AADHAAR": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",                          # 1234 5678 9012
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",                                        # ABCDE1234F
    "INDIAN_MOBILE": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",                       # +91 9876543210
    "IFSC": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",                                     # SBIN0001234
    "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b",                    # 27AADCB2230M1Z3
    "INDIAN_PASSPORT": r"\b[A-Z]\d{7}\b",                                    # A1234567
    "VOTER_ID": r"\b[A-Z]{3}\d{7}\b",                                        # ABC1234567

    # Global patterns
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",                                         # 123-45-6789
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",                               # 4111-1111-1111-1111
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",        # user@example.com
    "PHONE_INTL": r"\b\+\d{1,3}[\s-]?\d{7,14}\b",                            # +1 1234567890
}


def redact_pii(text: str) -> str:
    """Scan text and replace any detected PII with redaction markers."""
    redacted = text
    for pii_type, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{pii_type}]", redacted)
    return redacted


def detect_pii(text: str) -> list[dict]:
    """Return a list of detected PII types and their positions (for auditing)."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            findings.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "redacted": True,
            })
    return findings


class PrivacyLayer:
    @staticmethod
    def sanitize_query(query: str, user_cloud_enhanced: bool) -> str:
        """
        If the user has toggled cloud off, or if PII is detected, redact it before hitting external LLMs.
        """
        if not user_cloud_enhanced:
            # In complete local mode, we would route this to WebGPU/transformers.js or local Ollama.
            # Returning redacted string for safety if hitting external API.
            pass

        return redact_pii(query)

    @staticmethod
    def sanitize_response(response: str) -> str:
        """Redact any PII that may appear in LLM responses."""
        return redact_pii(response)
