import numpy as np
import pandas as pd

from ml.features.feature_engineering import (
    HourlyPriceModelFeatureEngineer,
    drop_incomplete_days,
)


def test_drop_incomplete_days_uses_local_index_dates():
    index = pd.date_range("2024-01-01", periods=72, freq="h")
    frame = pd.DataFrame({"feature": 1.0}, index=index)
    affected_date = pd.Timestamp("2024-01-01").date()
    affected_rows = index.date == affected_date
    frame.loc[affected_rows, "feature"] = np.nan

    filtered = drop_incomplete_days(frame)

    filtered_dates = pd.DatetimeIndex(filtered.index).date
    assert all(filtered_dates != affected_date)
    assert filtered_dates.min() > affected_date


def test_hourly_price_feature_engineer_adds_local_calendar_encodings():
    index = pd.DatetimeIndex(["2024-01-02 00:00:00"])
    frame = pd.DataFrame(
        {
            "resolution": ["hour"],
            "delivery_date": [pd.Timestamp("2024-01-01").date()],
            "local_timestamp": [pd.Timestamp("2024-01-01 00:00:00")],
            "year": [2024],
            "quarter": [1],
            "month": [1],
            "week_of_year": [1],
            "day_of_week": [0],
            "is_weekend": [False],
            "is_german_public_holiday": [True],
            "is_workday": [False],
            "price_lag_24h": [50.0],
            "wind_direction_100m_brandenburg_forecast_sin": [0.0],
            "wind_direction_100m_brandenburg_forecast_cos": [1.0],
        },
        index=index,
    )

    engineered = HourlyPriceModelFeatureEngineer().transform(frame)

    assert "resolution" not in engineered.columns
    assert "delivery_date" not in engineered.columns
    assert "month" not in engineered.columns
    assert engineered.loc[index[0], "hour_sin"] == 0.0
    assert engineered.loc[index[0], "hour_cos"] == 1.0
    assert engineered.loc[index[0], "dow_sin"] == np.sin(2 * np.pi / 7)
    assert engineered.loc[index[0], "dow_cos"] == np.cos(2 * np.pi / 7)