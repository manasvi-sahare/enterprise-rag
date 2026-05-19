from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize Presidio engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Define user roles and their allowed clearance levels
ROLE_PERMISSIONS = {
    "admin":     ["public", "internal", "confidential", "top_secret"],
    "manager":   ["public", "internal", "confidential"],
    "developer": ["public", "internal"],
    "guest":     ["public"]
}

def mask_pii(text: str) -> str:
    """Mask PII like names, emails, phone numbers, SSNs from text."""
    results = analyzer.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"],
        language="en"
    )
    if not results:
        return text
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text

def get_allowed_clearances(role: str) -> list:
    """Return list of clearance levels a role can access."""
    return ROLE_PERMISSIONS.get(role, ["public"])

def filter_results_by_role(results: list, role: str) -> list:
    """Filter search results based on user role."""
    allowed = get_allowed_clearances(role)
    return [r for r in results if r.get("clearance_level", "public") in allowed]
