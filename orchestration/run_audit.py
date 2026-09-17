"""PostgreSQL audit records for reproducible orchestration runs."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import psycopg2
from psycopg2.extras import Json

from db.settings import get_settings
from utils.git_utils import get_git_sha


def _get_connection():
    settings = get_settings()
    return psycopg2.connect(
        host=settings.ZEPHYRWERK_RDS_HOST,
        port=settings.ZEPHYRWERK_RDS_PORT,
        user=settings.ZEPHYRWERK_RDS_USER,
        password=settings.ZEPHYRWERK_RDS_PASSWORD,
        dbname=settings.ZEPHYRWERK_RDS_DB,
    )


def start_pipeline_run(
    pipeline_name: str,
    requested_start: datetime,
    requested_end: datetime,
    config_hash: str | None = None,
) -> UUID:
    """Create and return a running pipeline audit record."""
    run_id = uuid4()
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO operations.pipeline_runs (
                run_id, pipeline_name, status, requested_start, requested_end, git_sha, config_hash
            ) VALUES (%s, %s, 'running', %s, %s, %s, %s)
            """,
            (run_id, pipeline_name, requested_start, requested_end, get_git_sha(), config_hash),
        )
    return run_id


def finish_pipeline_run(
    run_id: UUID,
    status: str,
    raw_inventory_uri: str | None = None,
    error_message: str | None = None,
) -> None:
    """Mark a pipeline run as succeeded or failed."""
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE operations.pipeline_runs
            SET completed_at = NOW(), status = %s, raw_inventory_uri = %s, error_message = %s
            WHERE run_id = %s
            """,
            (status, raw_inventory_uri, error_message, run_id),
        )


def get_pipeline_run(run_id: UUID) -> dict[str, Any]:
    """Return an existing pipeline run or raise when the ID is unknown."""
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT pipeline_name, requested_start, requested_end, raw_inventory_uri
            FROM operations.pipeline_runs
            WHERE run_id = %s
            """,
            (run_id,),
        )
        row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Pipeline run {run_id} does not exist.")
    return {
        "pipeline_name": row[0],
        "requested_start": row[1],
        "requested_end": row[2],
        "raw_inventory_uri": row[3],
    }


def get_latest_stage_status(run_id: UUID, stage_name: str) -> str | None:
    """Return the latest recorded status for a stage in a pipeline run."""
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status
            FROM operations.pipeline_stage_runs
            WHERE run_id = %s AND stage_name = %s
            ORDER BY attempt_number DESC
            LIMIT 1
            """,
            (run_id, stage_name),
        )
        row = cursor.fetchone()
    return row[0] if row else None


def get_latest_stage_details(run_id: UUID, stage_name: str) -> dict[str, Any] | None:
    """Return the details stored for the latest attempt of a stage."""
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT details
            FROM operations.pipeline_stage_runs
            WHERE run_id = %s AND stage_name = %s
            ORDER BY attempt_number DESC
            LIMIT 1
            """,
            (run_id, stage_name),
        )
        row = cursor.fetchone()
    return row[0] if row else None


def reopen_pipeline_run(run_id: UUID) -> None:
    """Mark an existing run as active while an individual stage is retried."""
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE operations.pipeline_runs
            SET completed_at = NULL, status = 'running', error_message = NULL
            WHERE run_id = %s
            """,
            (run_id,),
        )


def start_stage_run(run_id: UUID, stage_name: str, config_hash: str | None = None) -> UUID:
    """Create a running audit record for one retryable pipeline stage."""
    stage_run_id = uuid4()
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO operations.pipeline_stage_runs (
                stage_run_id, run_id, stage_name, attempt_number, status, git_sha, config_hash
            )
            SELECT %s, %s, %s, COALESCE(MAX(attempt_number), 0) + 1, 'running', %s, %s
            FROM operations.pipeline_stage_runs
            WHERE run_id = %s AND stage_name = %s
            """,
            (stage_run_id, run_id, stage_name, get_git_sha(), config_hash, run_id, stage_name),
        )
    return stage_run_id


def finish_stage_run(
    stage_run_id: UUID,
    status: str,
    details: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """Mark a stage as succeeded or failed and persist JSON-serializable details."""
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE operations.pipeline_stage_runs
            SET completed_at = NOW(), status = %s, details = %s, error_message = %s
            WHERE stage_run_id = %s
            """,
            (status, Json(json.loads(json.dumps(details or {}, default=str))), error_message, stage_run_id),
        )