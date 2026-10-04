"""Reusable storage and metric helpers for published electricity price forecasts."""

from datetime import datetime
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import bindparam, text

from db.database import engine
from ml.evaluate import regression_metrics

PRICE_SIGNAL = "DE_LU"


def update_actual_values(start: datetime, end: datetime, model_names: Iterable[str]) -> int:
    """Backfill realized SMARD prices for published point forecasts."""
    names = tuple(model_names)
    if not names:
        return 0
    statement = text(
        """
        UPDATE analytics.forecast_results AS forecast
        SET actual_value = price.value
        FROM raw.smard_prices AS price
        WHERE forecast.model_name IN :model_names
          AND forecast.status = 'published'
          AND forecast.quantile IS NULL
          AND forecast.target_timestamp >= :start
          AND forecast.target_timestamp < :end
          AND price.timestamp = forecast.target_timestamp
          AND price.resolution = forecast.resolution
          AND price.signal = :price_signal
          AND price.value IS NOT NULL
          AND forecast.actual_value IS DISTINCT FROM price.value
        """
    ).bindparams(bindparam("model_names", expanding=True))
    with engine.begin() as connection:
        result = connection.execute(
            statement,
            {
                "model_names": names,
                "price_signal": PRICE_SIGNAL,
                "start": start,
                "end": end,
            },
        )
    return result.rowcount


def load_latest_forecasts(start: datetime, end: datetime, model_names: Iterable[str]) -> pd.DataFrame:
    """Load the latest published point forecast per model, timestamp, and resolution."""
    names = tuple(model_names)
    if not names:
        return pd.DataFrame()
    query = text(
        """
        SELECT DISTINCT ON (model_name, target_timestamp, resolution)
            model_name,
            target_timestamp,
            resolution,
            predicted_value,
            actual_value,
            generated_at,
            run_id
        FROM analytics.forecast_results
        WHERE model_name IN :model_names
          AND status = 'published'
          AND quantile IS NULL
          AND target_timestamp >= :start
          AND target_timestamp < :end
        ORDER BY model_name, target_timestamp, resolution, generated_at DESC
        """
    ).bindparams(bindparam("model_names", expanding=True))
    with engine.connect() as connection:
        return pd.read_sql_query(
            query,
            connection,
            params={"model_names": names, "start": start, "end": end},
        )


def metrics_by_resolution(forecasts: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Return error metrics for rows with realized prices, grouped by resolution."""
    matched = forecasts.dropna(subset=["actual_value"]).copy()
    metrics_by_group: dict[str, dict[str, float | int | None]] = {}
    for resolution, rows in matched.groupby("resolution", sort=True):
        errors = rows["predicted_value"] - rows["actual_value"]
        if len(rows) == 1:
            metrics: dict[str, float | int | None] = {
                "mae": float(errors.abs().iloc[0]),
                "rmse": float(errors.abs().iloc[0]),
                "r2": None,
            }
        else:
            metrics = regression_metrics(rows["actual_value"], rows["predicted_value"])
        metrics["bias"] = float(errors.mean())
        metrics["row_count"] = len(rows)
        metrics_by_group[str(resolution)] = metrics
    return metrics_by_group


def forecast_error_summary(forecasts: pd.DataFrame) -> dict[str, Any]:
    """Summarize coverage and realized-price errors for one forecast source."""
    matched = forecasts.dropna(subset=["actual_value"])
    return {
        "latest_forecast_rows": len(forecasts),
        "evaluated_rows": len(matched),
        "unavailable_actual_rows": len(forecasts) - len(matched),
        "resolutions": metrics_by_resolution(forecasts),
    }