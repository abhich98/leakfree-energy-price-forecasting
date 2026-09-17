"""Run dbt for a pipeline run and archive its generated artifacts."""

import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

import boto3

DBT_DEFINITION_PATHS = (
    "dbt/models",
    "dbt/macros",
    "dbt/seeds",
    "dbt/tests",
    "dbt/dbt_project.yml",
    "dbt/packages.yml",
    "dbt/package-lock.yml",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DBT_PROJECT_DIR = PROJECT_ROOT / "dbt"


def _require_clean_dbt_definition() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *DBT_DEFINITION_PATHS],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        raise RuntimeError(
            "dbt definition has uncommitted changes. Commit changes to dbt models, "
            "macros, seeds, tests, or project configuration before running dbt."
        )


def _upload_artifact(s3: Any, bucket: str, source: Path, key: str) -> str:
    s3.upload_file(str(source), bucket, key)
    return f"s3://{bucket}/{key}"


def run_dbt(run_id: UUID) -> dict[str, str]:
    """Run and test dbt from committed sources, then archive its run artifacts."""
    _require_clean_dbt_definition()
    bucket = os.environ.get("ZEPHYRWERK_AWS_BUCKET_NAME")
    if not bucket:
        raise ValueError("ZEPHYRWERK_AWS_BUCKET_NAME environment variable is not set.")

    target_path = Path("target") / "pipeline-runs" / str(run_id)
    artifact_dir = DBT_PROJECT_DIR / target_path
    command_base = [
        "dbt",
        "--project-dir",
        str(DBT_PROJECT_DIR),
        "--profiles-dir",
        str(DBT_PROJECT_DIR),
        "--target-path",
        str(target_path),
    ]
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    prefix = f"artifacts/pipeline-runs/{run_id}/dbt"

    subprocess.run([*command_base, "run"], cwd=PROJECT_ROOT, check=True)
    run_results_uri = _upload_artifact(
        s3, bucket, artifact_dir / "run_results.json", f"{prefix}/run_results_run.json"
    )

    subprocess.run([*command_base, "test"], cwd=PROJECT_ROOT, check=True)
    test_results_uri = _upload_artifact(
        s3, bucket, artifact_dir / "run_results.json", f"{prefix}/run_results_test.json"
    )
    manifest_uri = _upload_artifact(
        s3, bucket, artifact_dir / "manifest.json", f"{prefix}/manifest.json"
    )
    return {
        "manifest_uri": manifest_uri,
        "run_results_run_uri": run_results_uri,
        "run_results_test_uri": test_results_uri,
    }