"""Reject known secrets before persistence; no credential lookup occurs here."""

import base64
import json
import re
from urllib.parse import quote


class SecretPolicy:
    def __init__(self, forbidden_values: tuple[str, ...] = ()):
        self.forbidden_values = tuple(value for value in forbidden_values if value)
        self.pattern = re.compile(
            r"gh[pousr]_[A-Za-z0-9]{30,}|AIza[A-Za-z0-9_-]{30,}|"
            r"sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        )

    def register_secret(self, value: str) -> None:
        """Register a credential discovered later without exposing it in errors."""
        if not value:
            return
        variants = {
            value,
            quote(value, safe=""),
            base64.b64encode(value.encode()).decode("ascii"),
        }
        self.forbidden_values = tuple(dict.fromkeys((*self.forbidden_values, *variants)))

    def validate_text(self, text: str) -> None:
        if any(value in text for value in self.forbidden_values) or self.pattern.search(text):
            raise ValueError("SECRET_PAYLOAD_BLOCKED")

    def validate_json(self, text: str) -> None:
        self.validate_text(text)

        def inspect(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if str(key).casefold() in {
                        "api_key",
                        "authorization",
                        "password",
                        "access_token",
                        "refresh_token",
                        "cookie",
                        "secret",
                    }:
                        raise ValueError("SECRET_FIELD_BLOCKED")
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(json.loads(text))
