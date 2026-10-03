"""PostgreSQL audit records for reproducible orchestration runs."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from db.database import engine
from utils.git import get_git_sha


def start_pipeline_run(
    pipeline_name: str,
    requested_start: datetime,
    requested_end: datetime,
    config_hash: str | None = None,
    source_elt_run_id: UUID | None = None,
) -> UUID:
    """Create and return a running pipeline audit record."""
    run_id = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text(
            """
            INSERT INTO operations.pipeline_runs (
                run_id, pipeline_name, source_elt_run_id, status, requested_start, requested_end, git_sha, config_hash
            ) VALUES (
                :run_id, :pipeline_name, :source_elt_run_id, 'running',
                :requested_start, :requested_end, :git_sha, :config_hash
            )
            """,
            ),
            {
                "run_id": run_id,
                "pipeline_name": pipeline_name,
                "source_elt_run_id": source_elt_run_id,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "git_sha": get_git_sha(),
                "config_hash": config_hash,
            },
        )
    return run_id


def finish_pipeline_run(
    run_id: UUID,
    status: str,
    raw_inventory_uri: str | None = None,
    error_message: str | None = None,
) -> None:
    """Mark a pipeline run as succeeded or failed."""
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE operations.pipeline_runs
                SET completed_at = NOW(), status = :status,
                    raw_inventory_uri = :raw_inventory_uri, error_message = :error_message
                WHERE run_id = :run_id
                """
            ),
            {
                "status": status,
                "raw_inventory_uri": raw_inventory_uri,
                "error_message": error_message,
                "run_id": run_id,
            },
        )


def get_pipeline_run(run_id: UUID) -> dict[str, Any]:
    """Return an existing pipeline run or raise when the ID is unknown."""
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
            SELECT pipeline_name, status, requested_start, requested_end, raw_inventory_uri, source_elt_run_id
            FROM operations.pipeline_runs
            WHERE run_id = :run_id
            """
            ),
            {"run_id": run_id},
        ).mappings().first()
    if row is None:
        raise ValueError(f"Pipeline run {run_id} does not exist.")
    return dict(row)


def get_latest_stage_status(run_id: UUID, stage_name: str) -> str | None:
    """Return the latest recorded status for a stage in a pipeline run."""
    with engine.connect() as connection:
        return connection.execute(
            text(
                """
            SELECT status
            FROM operations.pipeline_stage_runs
            WHERE run_id = :run_id AND stage_name = :stage_name
            ORDER BY attempt_number DESC
            LIMIT 1
            """
            ),
            {"run_id": run_id, "stage_name": stage_name},
        ).scalar_one_or_none()


def get_latest_stage_details(run_id: UUID, stage_name: str) -> dict[str, Any] | None:
    """Return the details stored for the latest attempt of a stage."""
    with engine.connect() as connection:
        return connection.execute(
            text(
                """
            SELECT details
            FROM operations.pipeline_stage_runs
            WHERE run_id = :run_id AND stage_name = :stage_name
            ORDER BY attempt_number DESC
            LIMIT 1
            """
            ),
            {"run_id": run_id, "stage_name": stage_name},
        ).scalar_one_or_none()


def reopen_pipeline_run(run_id: UUID) -> None:
    """Mark an existing run as active while an individual stage is retried."""
    with engine.begin() as connection:
        connection.execute(
            text(
                """
            UPDATE operations.pipeline_runs
            SET completed_at = NULL, status = 'running', error_message = NULL
            WHERE run_id = :run_id
            """
            ),
            {"run_id": run_id},
        )


def start_stage_run(run_id: UUID, stage_name: str, config_hash: str | None = None) -> UUID:
    """Create a running audit record for one retryable pipeline stage."""
    stage_run_id = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
            INSERT INTO operations.pipeline_stage_runs (
                stage_run_id, run_id, stage_name, attempt_number, status, git_sha, config_hash
            )
            SELECT :stage_run_id, :run_id, :stage_name,
                   COALESCE(MAX(attempt_number), 0) + 1, 'running', :git_sha, :config_hash
            FROM operations.pipeline_stage_runs
            WHERE run_id = :run_id AND stage_name = :stage_name
            """
            ),
            {
                "stage_run_id": stage_run_id,
                "run_id": run_id,
                "stage_name": stage_name,
                "git_sha": get_git_sha(),
                "config_hash": config_hash,
            },
        )
    return stage_run_id


def finish_stage_run(
    stage_run_id: UUID,
    status: str,
    details: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """Mark a stage as succeeded or failed and persist JSON-serializable details."""
    statement = text(
        """
            UPDATE operations.pipeline_stage_runs
            SET completed_at = NOW(), status = :status, details = :details, error_message = :error_message
            WHERE stage_run_id = :stage_run_id
        """
    ).bindparams(bindparam("details", type_=JSONB))
    normalized_details = json.loads(json.dumps(details or {}, default=str))
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "status": status,
                "details": normalized_details,
                "error_message": error_message,
                "stage_run_id": stage_run_id,
            },
        )