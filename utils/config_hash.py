"""Hashes of non-secret runtime configuration for orchestration provenance."""

import hashlib
import json
from typing import Any


def config_hash(configuration: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest of a JSON-serializable configuration."""
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()