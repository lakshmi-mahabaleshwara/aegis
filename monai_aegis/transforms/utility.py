"""
MONAI Aegis Utility — AegisIdentityManager

Deterministic identity tokenization for de-identification.
Generates reproducible SHA-256 tokens for PII values, enabling
re-identification when the salt is preserved.
"""
import hashlib
import logging

logger = logging.getLogger(__name__)


class AegisIdentityManager:
    """
    Manages identity tokenization for de-identification.
    
    Generates deterministic, consistent tokens for PII values using
    SHA-256 hashing with a configurable salt. The same input always
    produces the same token, enabling re-identification if the salt
    is preserved.
    """
    def __init__(self, salt: str = "monai_aegis") -> None:
        """Initialize the identity manager.

        Args:
            salt: Secret salt for deterministic hashing. The same salt
                must be used for consistent tokenization across runs.
                Preserve this value if re-identification is needed.
        """
        self.salt = salt

    def get_token(self, value: str) -> str:
        """
        Generates a consistent token for a given value using SHA-256.

        Args:
            value: The input string to tokenize (e.g., PatientID).

        Returns:
            A shortened hash token string prefixed with 'TOKEN_'.
        """
        if not value:
            return ""

        data = f"{value}{self.salt}".encode('utf-8')
        token = hashlib.sha256(data).hexdigest()[:16]
        return f"TOKEN_{token}"
