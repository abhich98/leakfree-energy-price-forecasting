"""PostgreSQL audit records for reproducible orchestration runs."""

import json
import subprocess
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


def start_stage_run(run_id: UUID, stage_name: str) -> UUID:
    """Create a running audit record for one idempotent pipeline stage."""
    stage_run_id = uuid4()
    with _get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO operations.pipeline_stage_runs (stage_run_id, run_id, stage_name, status)
            VALUES (%s, %s, %s, 'running')
            """,
            (stage_run_id, run_id, stage_name),
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