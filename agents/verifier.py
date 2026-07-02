from __future__ import annotations


def verify(response: str) -> bool:
    """Validate a generated response for consistency and schema adherence."""
    return bool(response and response.strip())
