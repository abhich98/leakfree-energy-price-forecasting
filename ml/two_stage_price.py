"""Shared preparation and inference contract for two-stage price models."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.features.feature_engineering import TARGET_COLUMNS, drop_incomplete_rows
from ml.training_utils import ModelType, broadcast_predictions_h2qh, fill_short_feature_gaps

STAGE1_PREDICTION_COLUMNS = ("stage1_hourly_prediction", "hourly_prediction")


def prepare_price_features(
    dataframe: pd.DataFrame,
    model_type: ModelType,
    *,
    require_target: bool,
) -> pd.DataFrame:
    """Fill short feature gaps and remove only rows still unusable by a model."""
    target_column = TARGET_COLUMNS[model_type]
    filled = fill_short_feature_gaps(dataframe, omit_columns=[target_column])
    required_columns = list(filled.columns)
    if not require_target and target_column in required_columns:
        required_columns.remove(target_column)
    return drop_incomplete_rows(filled, required_columns)


def _prediction_features(dataframe: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
    return dataframe.drop(
        columns=[TARGET_COLUMNS[model_type], *STAGE1_PREDICTION_COLUMNS],
        errors="ignore",
    )


@dataclass(frozen=True)
class TwoStagePricePredictions:
    hourly: pd.Series
    quarter_hourly: pd.Series


@dataclass
class TwoStagePriceModel:
    """Fitted hourly model plus quarter-hour residual model."""

    hourly_pipeline: Any
    quarter_hourly_pipeline: Any

    def predict_hourly(self, hourly_features: pd.DataFrame) -> pd.Series:
        predictions = self.hourly_pipeline.predict(
            _prediction_features(hourly_features, ModelType.PRICE_HOURLY)
        )
        return pd.Series(predictions, index=hourly_features.index, name="predicted_value")

    def predict_quarter_hourly(
        self,
        quarter_hourly_features: pd.DataFrame,
        hourly_predictions: pd.Series,
    ) -> pd.Series:
        stage1_predictions = broadcast_predictions_h2qh(
            pd.DatetimeIndex(quarter_hourly_features.index), hourly_predictions
        )
        if stage1_predictions.isna().any():
            raise ValueError("Stage 1 predictions do not cover every quarter-hour prediction row")

        residuals = self.quarter_hourly_pipeline.predict(
            _prediction_features(quarter_hourly_features, ModelType.PRICE_QUARTER_HOURLY)
        )
        return pd.Series(
            stage1_predictions.to_numpy() + residuals,
            index=quarter_hourly_features.index,
            name="predicted_value",
        )

    def predict(
        self,
        hourly_features: pd.DataFrame,
        quarter_hourly_features: pd.DataFrame,
    ) -> TwoStagePricePredictions:
        hourly_predictions = self.predict_hourly(hourly_features)
        return TwoStagePricePredictions(
            hourly=hourly_predictions,
            quarter_hourly=self.predict_quarter_hourly(
                quarter_hourly_features, hourly_predictions
            ),
        )