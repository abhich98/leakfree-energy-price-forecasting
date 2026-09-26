"""
ml/features/feature_engineering.py

Leak-safe feature engineering for the Zephyrwerk price and generation models.

Two sklearn-compatible transformers:
  - PriceModelFeatureEngineer: for the day-ahead price target (price_eur_mwh)
  - GenerationModelFeatureEngineer: for wind_total_mw or solar_mw targets

Both are shape-preserving (do not drop rows), so they compose cleanly inside
sklearn Pipelines. Row-dropping for lag NaN rows happens in the training
script, outside the pipeline.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.training_utils import ModelType

# ---- Column constants ----

TARGET_COLUMNS = {
    ModelType.PRICE_HOURLY: "price_eur_mwh",
    ModelType.PRICE_QUARTER_HOURLY: "price_eur_mwh",
    ModelType.PRICE: "price_eur_mwh",
    ModelType.WIND: "wind_total_mw",
    ModelType.SOLAR: "solar_mw",
}

BASELINE_PRED_COLUMNS = {
    ModelType.PRICE_HOURLY: "price_lag_24h",
    ModelType.PRICE_QUARTER_HOURLY: "price_lag_96qh",
}

# Nuclear is ~75% NaN in the post-phase-out window; holiday_name is free text.
DROP_ALWAYS = ["nuclear_mw", "holiday_name", "date_day", "season"]

HOURLY_PRICE_METADATA = ["resolution", "delivery_date", "timestamp", "local_timestamp"]
HOURLY_PRICE_RAW_CALENDAR = ["year", "quarter", "month", "week_of_year", "day_of_week"]
QUARTER_HOUR_PRICE_METADATA = HOURLY_PRICE_METADATA.copy()
QUARTER_HOUR_PRICE_RAW_CALENDAR = HOURLY_PRICE_RAW_CALENDAR.copy()


logger = logging.getLogger(__name__)


# ---- Pure helper functions ----


def drop_incomplete_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every delivery day containing at least one null.

    A naive DatetimeIndex is used as-is.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("complete-day filtering requires a DatetimeIndex")

    index = pd.DatetimeIndex(df.index)
    local_dates = pd.Series(index.date, index=index)

    complete_dates = df.notna().groupby(local_dates).all().all(axis=1)
    keep = local_dates.map(complete_dates).to_numpy()

    to_drop = local_dates[~keep].unique()
    logger.info(
        "Dropping %d incomplete days: %s",
        len(to_drop),
        ", ".join(str(d) for d in sorted(to_drop)),
    )

    return df.loc[keep].copy()


def drop_incomplete_rows(
    df: pd.DataFrame, required_columns: list[str] | None = None
) -> pd.DataFrame:
    """Remove rows with unresolved nulls in the columns required by a model."""
    columns = required_columns or list(df.columns)
    unknown_columns = set(columns) - set(df.columns)
    if unknown_columns:
        raise KeyError(f"required columns are missing from DataFrame: {sorted(unknown_columns)}")

    complete_rows = df.loc[:, columns].notna().all(axis=1)
    dropped_rows = int((~complete_rows).sum())
    if dropped_rows:
        logger.info("Dropping %d rows with unresolved required values", dropped_rows)
    return df.loc[complete_rows].copy()


# ---- sklearn transformers ----


class HourlyPriceModelFeatureEngineer(BaseEstimator, TransformerMixin):
    """Prepare rows from ``fct_ml_hourly_price_model_features`` for training."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("hourly price features require a DatetimeIndex")

        df = X.copy()
        local_index = pd.DatetimeIndex(df.index)
        hour = local_index.hour
        day_of_week = local_index.dayofweek
        month = local_index.month

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

        to_drop = HOURLY_PRICE_METADATA + HOURLY_PRICE_RAW_CALENDAR
        return df.drop(columns=[column for column in to_drop if column in df.columns])


class QuarterHourPriceModelFeatureEngineer(BaseEstimator, TransformerMixin):
    """Prepare rows from ``fct_ml_quarter_hour_price_model_features`` for training."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("quarter-hour price features require a DatetimeIndex")

        df = X.copy()
        local_index = pd.DatetimeIndex(df.index)
        hour = local_index.hour
        minute = local_index.minute
        day_of_week = local_index.dayofweek
        month = local_index.month

        df["hour_sin"] = np.sin(2 * np.pi * (hour + minute / 60) / 24)
        df["hour_cos"] = np.cos(2 * np.pi * (hour + minute / 60) / 24)
        df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

        to_drop = QUARTER_HOUR_PRICE_METADATA + QUARTER_HOUR_PRICE_RAW_CALENDAR
        return df.drop(columns=[column for column in to_drop if column in df.columns])


# ---- Utilities ----


def split_x_y(
    df: pd.DataFrame, model_type: ModelType
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target column from features."""
    target = TARGET_COLUMNS[model_type]
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found in DataFrame")
    return df.drop(columns=[target]), df[target].copy()


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    holdout_start_date: str | None = None,
    holdout_days: int = 90,
):
    """
    Time-based holdout split. Requires X to have a DatetimeIndex.
    Returns (X_trainval, X_test, y_trainval, y_test).

    If `holdout_start_date` is provided, it overrides `holdout_days` and defines
    the start date of the holdout period.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("temporal_split requires a DatetimeIndex on X")

    if holdout_start_date is not None:
        cutoff = pd.to_datetime(holdout_start_date)
    else:
        cutoff = X.index.max() - pd.Timedelta(days=holdout_days)

    holdout_mask = X.index >= cutoff
    return (
        X.loc[~holdout_mask],
        X.loc[holdout_mask],
        y.loc[~holdout_mask],
        y.loc[holdout_mask],
    )
