import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _dataframe_fingerprint(dataframe: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    ordered = dataframe.sort_index().copy()
    row_hashes = pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
    content_hash = hashlib.sha256(row_hashes).hexdigest()

    index = ordered.index
    summary: dict[str, Any] = {
        "rows": int(len(ordered)),
        "columns": [str(column) for column in ordered.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in ordered.dtypes.items()},
        "null_counts": {str(column): int(count) for column, count in ordered.isna().sum().items()},
        "content_sha256": content_hash,
    }
    if len(index):
        summary["index_start"] = _json_default(index.min())
        summary["index_end"] = _json_default(index.max())

    return content_hash, summary


def build_data_manifest(
    datasets: dict[str, pd.DataFrame],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest for the exact prepared training datasets."""
    dataset_summaries: dict[str, dict[str, Any]] = {}
    for name, dataframe in datasets.items():
        _, summary = _dataframe_fingerprint(dataframe)
        dataset_summaries[name] = summary

    identity = {
        "datasets": dataset_summaries,
        "metadata": metadata or {},
    }
    canonical_identity = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    data_version_id = hashlib.sha256(canonical_identity).hexdigest()

    return {
        "manifest_version": "1.0",
        "data_version_id": data_version_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "metadata": metadata or {},
        "datasets": dataset_summaries,
    }


def save_local_manifest(manifest: dict[str, Any], directory: str = "ml/artifacts") -> str:
    """Save both an immutable local manifest and a convenience latest copy."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = output_dir / f"data_version_manifest_{manifest['data_version_id']}.json"
    latest_path = output_dir / "data_version_manifest.json"
    payload = json.dumps(manifest, indent=2, default=_json_default)
    if not versioned_path.exists():
        versioned_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return str(versioned_path)
