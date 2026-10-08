from __future__ import annotations

import json
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {
    "application_id",
    "applicationid",
    "submission_id",
    "venture_name",
    "venture",
    "startup_name",
    "reviewer_name",
    "reviewer_email",
    "reviewer_email_address",
    "contact_id",
    "problem_text",
    "application_text",
    "top_examples",
    "moderate_examples",
    "representative_problem_excerpts",
    "representative_startups",
    "member_number",
    "embeddings",
    "activations",
    "model_checkpoint",
    "checkpoint_path",
}


def assert_public_privacy(payload: dict[str, Any]) -> None:
    """Fail closed if a row-level or confidential field enters the public tree."""
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert str(key).lower() not in FORBIDDEN_PUBLIC_KEYS, f"Forbidden public key: {key}"
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).lower()
    assert "@" not in serialized, "Possible email address in public payload"
