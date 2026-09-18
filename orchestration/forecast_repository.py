"""Transactional persistence for validated forecasts."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from psycopg2.extras import execute_values

from utils.rds import get_connection


def persist_forecasts(run_id: UUID, model_bundle: dict[str, Any], forecasts: list[dict[str, Any]]) -> int:
    """Upsert validated point or quantile forecasts in one PostgreSQL transaction."""
    generated_at = datetime.now(timezone.utc)
    values = [
        (
            uuid4(),
            run_id,
            "two_stage_price",
            model_bundle["pointer_uri"],
            model_bundle["data_version_id"],
            generated_at,
            forecast["target_timestamp"],
            forecast["resolution"],
            forecast.get("quantile"),
            forecast["predicted_value"],
        )
        for forecast in forecasts
    ]
    with get_connection() as connection, connection.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO analytics.forecast_results (
                forecast_id, run_id, model_name, model_version_uri, data_version_id,
                generated_at, target_timestamp, resolution, quantile, predicted_value, status, published_at
            ) VALUES %s
            ON CONFLICT (run_id, model_name, target_timestamp, resolution, COALESCE(quantile, -1))
            DO UPDATE SET predicted_value = EXCLUDED.predicted_value,
                          generated_at = EXCLUDED.generated_at,
                          model_version_uri = EXCLUDED.model_version_uri,
                          data_version_id = EXCLUDED.data_version_id,
                          status = EXCLUDED.status,
                          published_at = EXCLUDED.published_at
            """,
            [(*value, "published", generated_at) for value in values],
        )
    return len(values)