"""Password hashing and demo doctor accounts. Stdlib only (no ML)."""

from __future__ import annotations

import hashlib
import hmac
import secrets


PBKDF2_ROUNDS = 120_000


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    )
    return hmac.compare_digest(candidate.hex(), digest)


DEMO_DOCTORS = [
    {
        "username": "doctor",
        "full_name": "Dr. A. Sharma",
        "specialty": "General Medicine",
        "password": "clinic123",
    },
    {
        "username": "dr.mehta",
        "full_name": "Dr. R. Mehta",
        "specialty": "Internal Medicine",
        "password": "clinic123",
    },
]
