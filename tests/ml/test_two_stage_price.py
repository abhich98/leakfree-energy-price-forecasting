import numpy as np
import pandas as pd

from ml.training_utils import ModelType
from ml.two_stage_price import TwoStagePriceModel, prepare_price_features


class CapturingPipeline:
    def __init__(self, predictions):
        self.predictions = predictions
        self.features = None

    def predict(self, features):
        self.features = features.copy()
        return np.asarray(self.predictions)


def test_prepare_price_features_fills_short_gaps_and_drops_only_unresolved_rows():
    index = pd.date_range("2026-09-20", periods=5, freq="h")
    raw = pd.DataFrame(
        {"feature": [1.0, None, None, None, None], "price_eur_mwh": [None] * 5},
        index=index,
    )

    prepared = prepare_price_features(
        raw, ModelType.PRICE_HOURLY, require_target=False
    )

    assert prepared.index.equals(index[:4])
    assert prepared["feature"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_two_stage_model_excludes_targets_and_stage1_training_column():
    hourly_index = pd.date_range("2026-09-20", periods=2, freq="h")
    qh_index = pd.date_range("2026-09-20", periods=8, freq="15min")
    hourly = pd.DataFrame(
        {"feature": [10.0, 20.0], "price_eur_mwh": [None, None]}, index=hourly_index
    )
    quarter_hourly = pd.DataFrame(
        {
            "feature": range(8),
            "price_eur_mwh": [None] * 8,
            "stage1_hourly_prediction": [0.0] * 8,
        },
        index=qh_index,
    )
    hourly_pipeline = CapturingPipeline([10.0, 20.0])
    qh_pipeline = CapturingPipeline([1.0] * 8)

    predictions = TwoStagePriceModel(hourly_pipeline, qh_pipeline).predict(
        hourly, quarter_hourly
    )

    assert hourly_pipeline.features.columns.tolist() == ["feature"]
    assert qh_pipeline.features.columns.tolist() == ["feature"]
    assert predictions.quarter_hourly.tolist() == [11.0] * 4 + [21.0] * 4