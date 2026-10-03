"""Transactional persistence for validated forecasts."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

from db.database import engine


def persist_forecasts(
    run_id: UUID, model_bundle: dict[str, Any], forecasts: list[dict[str, Any]]
) -> int:
    """Upsert validated point or quantile forecasts in one PostgreSQL transaction."""
    generated_at = datetime.now(timezone.utc)
    parameters = [
        {
            "forecast_id": uuid4(),
            "run_id": run_id,
            "model_name": "two_stage_price",
            "model_version_uri": model_bundle["pointer_uri"],
            "data_version_id": model_bundle["data_version_id"],
            "generated_at": generated_at,
            "target_timestamp": forecast["target_timestamp"],
            "resolution": forecast["resolution"],
            "quantile": forecast.get("quantile"),
            "predicted_value": forecast["predicted_value"],
            "status": "published",
            "published_at": generated_at,
        }
        for forecast in forecasts
    ]
    if not parameters:
        return 0

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO analytics.forecast_results (
                    forecast_id, run_id, model_name, model_version_uri, data_version_id,
                    generated_at, target_timestamp, resolution, quantile, predicted_value,
                    status, published_at
                ) VALUES (
                    :forecast_id, :run_id, :model_name, :model_version_uri, :data_version_id,
                    :generated_at, :target_timestamp, :resolution, :quantile, :predicted_value,
                    :status, :published_at
                )
                ON CONFLICT (run_id, model_name, target_timestamp, resolution, COALESCE(quantile, -1))
                DO UPDATE SET predicted_value = EXCLUDED.predicted_value,
                              generated_at = EXCLUDED.generated_at,
                              model_version_uri = EXCLUDED.model_version_uri,
                              data_version_id = EXCLUDED.data_version_id,
                              status = EXCLUDED.status,
                              published_at = EXCLUDED.published_at
                """
            ),
            parameters,
        )
    return len(parameters)