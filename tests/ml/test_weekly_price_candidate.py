from unittest.mock import MagicMock

import pandas as pd
import pytest

import orchestration.weekly_price_candidate as candidate
from orchestration.weekly_price_candidate import build_weekly_training_windows


def test_build_weekly_training_windows_uses_four_week_evaluation_and_two_year_hourly_training():
    windows = build_weekly_training_windows("2026-09-18")

    assert windows.evaluation_start == pd.Timestamp("2026-08-21")
    assert windows.evaluation_end_exclusive == pd.Timestamp("2026-09-18")
    assert windows.hourly_train_start == pd.Timestamp("2024-08-21")
    assert windows.quarter_hourly_train_start == pd.Timestamp("2025-10-01")


def test_windows_serialize_as_dates_for_run_metadata():
    windows = build_weekly_training_windows("2026-09-18")

    assert windows.as_dict() == {
        "hourly_train_start": "2024-08-21",
        "quarter_hourly_train_start": "2025-10-01",
        "evaluation_start": "2026-08-21",
        "evaluation_end_exclusive": "2026-09-18",
    }


def _evaluation(candidate_value: float, baseline_value: float) -> dict[str, pd.Series]:
    index = pd.date_range("2026-08-21", periods=28, freq="D")
    actuals = pd.Series(100.0, index=index)
    return {
        "hourly_predictions": pd.Series(candidate_value, index=index),
        "hourly_actuals": actuals,
        "hourly_baseline": pd.Series(baseline_value, index=index),
        "qh_predictions": pd.Series(candidate_value, index=index),
        "qh_actuals": actuals,
        "qh_baseline": pd.Series(baseline_value, index=index),
    }


def _configure_workflow(monkeypatch, evaluation: dict[str, pd.Series]):
    raw = pd.DataFrame(index=pd.date_range("2024-08-21", periods=2, freq="D"))
    monkeypatch.setattr(
        candidate,
        "get_latest_elt_run",
        lambda **_: {"run_id": "elt-run-id", "status": "succeeded"},
    )
    monkeypatch.setattr(candidate, "load_hourly_price_model_features", lambda **_: raw)
    monkeypatch.setattr(candidate, "load_quarter_hourly_price_model_features", lambda **_: raw)
    monkeypatch.setattr(candidate, "_prepare_features", lambda dataframe, *_, **__: dataframe)
    monkeypatch.setattr(candidate, "_load_hyperparameters", lambda _: None)
    monkeypatch.setattr(
        candidate,
        "_fit_candidate",
        lambda *_: (MagicMock(), MagicMock(), evaluation),
    )
    monkeypatch.setattr(candidate, "_load_champion", lambda: None)
    monkeypatch.setattr(
        candidate,
        "build_data_manifest",
        lambda *_: {"data_version_id": "data-version-1"},
    )
    monkeypatch.setattr(candidate, "save_local_data_manifest", lambda _: "local-manifest.json")
    monkeypatch.setattr(candidate, "save_data_manifest", lambda _: "s3://bucket/manifest.json")
    monkeypatch.setattr(
        candidate,
        "save_pipeline",
        lambda _, model_name, metadata: f"s3://bucket/{model_name}.joblib",
    )
    monkeypatch.setattr(candidate, "save_report", lambda *_: None)


def test_approved_initial_candidate_publishes_production_bundle(monkeypatch):
    _configure_workflow(monkeypatch, _evaluation(candidate_value=101.0, baseline_value=105.0))
    publish = MagicMock(return_value="s3://bucket/models/price/production.json")
    monkeypatch.setattr(candidate, "save_production_model_bundle", publish)

    report = candidate.run_weekly_price_candidate("2026-09-18")

    assert report["promotion"]["approved"]
    assert report["production_pointer_uri"] == "s3://bucket/models/price/production.json"
    publish.assert_called_once()


def test_rejected_candidate_does_not_publish_production_bundle(monkeypatch):
    _configure_workflow(monkeypatch, _evaluation(candidate_value=105.0, baseline_value=101.0))
    publish = MagicMock()
    monkeypatch.setattr(candidate, "save_production_model_bundle", publish)

    report = candidate.run_weekly_price_candidate("2026-09-18")

    assert not report["promotion"]["approved"]
    assert "production_pointer_uri" not in report
    publish.assert_not_called()


def test_candidate_requires_a_successful_latest_elt_run(monkeypatch):
    monkeypatch.setattr(
        candidate,
        "get_latest_elt_run",
        lambda **_: (_ for _ in ()).throw(ValueError("Latest ELT run failed")),
    )

    with pytest.raises(ValueError, match="Latest ELT run failed"):
        candidate.run_weekly_price_candidate("2026-09-18")