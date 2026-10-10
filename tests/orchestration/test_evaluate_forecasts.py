from datetime import datetime, timezone

import pandas as pd
import pytest

from orchestration import evaluate_forecasts

START = datetime(2026, 9, 19, tzinfo=timezone.utc)
END = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_timestamp": pd.date_range(START, periods=4, freq="h"),
            "resolution": ["hour", "hour", "quarterhour", "quarterhour"],
            "predicted_value": [50.0, 60.0, 55.0, 65.0],
            "actual_value": [45.0, 65.0, 50.0, None],
        }
    )


def test_build_evaluation_report_groups_metrics_by_resolution():
    report = evaluate_forecasts.build_evaluation_report(_forecasts(), START, END, updated_actuals=3)

    assert report["latest_forecast_rows"] == 4
    assert report["evaluated_rows"] == 3
    assert report["unavailable_actual_rows"] == 1
    assert report["actual_values_updated"] == 3
    assert report["resolutions"]["hour"]["mae"] == pytest.approx(5.0)
    assert report["resolutions"]["quarterhour"]["row_count"] == 1
    assert report["resolutions"]["quarterhour"]["r2"] is None


def test_build_evaluation_report_requires_at_least_one_actual():
    forecasts = _forecasts().assign(actual_value=None)

    with pytest.raises(ValueError, match="No realized prices"):
        evaluate_forecasts.build_evaluation_report(forecasts, START, END, updated_actuals=0)


def test_save_comparison_plot_writes_png(tmp_path):
    output_path = tmp_path / "comparison.png"

    evaluate_forecasts.save_comparison_plot(_forecasts(), output_path)

    assert output_path.is_file()


def test_save_comparison_plot_uses_berlin_timestamps(monkeypatch, tmp_path):
    plotted_timestamps: list[pd.Series] = []
    original_subplots = evaluate_forecasts.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        for axis in axes[:, 0]:
            original_plot = axis.plot

            def capture_plot(timestamps, *plot_args, original_plot=original_plot, **plot_kwargs):
                plotted_timestamps.append(timestamps)
                return original_plot(timestamps, *plot_args, **plot_kwargs)

            monkeypatch.setattr(axis, "plot", capture_plot)
        return figure, axes

    monkeypatch.setattr(evaluate_forecasts.plt, "subplots", capture_subplots)
    evaluate_forecasts.save_comparison_plot(_forecasts(), tmp_path / "comparison.png")

    assert plotted_timestamps[0].dt.tz == evaluate_forecasts.MARKET_TIMEZONE


def test_build_forecast_comparison_report_includes_forecast_and_actual_metrics():
    forecasts = pd.DataFrame(
        [
            {
                "model_name": evaluate_forecasts.MODEL_NAME,
                "target_timestamp": START,
                "resolution": "quarterhour",
                "predicted_value": 101.0,
                "actual_value": 100.0,
            },
            {
                "model_name": evaluate_forecasts.EXTERNAL_MODEL_NAME,
                "target_timestamp": START,
                "resolution": "quarterhour",
                "predicted_value": 102.0,
                "actual_value": 100.0,
            },
        ]
    )

    report = evaluate_forecasts.build_forecast_comparison_report(forecasts, START, END, updated_actuals=2)

    assert report["forecast_difference"] == {
        "row_count": 1,
        "mae": 1.0,
        "rmse": 1.0,
        "bias_external_minus_internal": 1.0,
    }
    assert report["models"][evaluate_forecasts.MODEL_NAME]["resolutions"]["quarterhour"]["mae"] == 1.0
    assert report["models"][evaluate_forecasts.EXTERNAL_MODEL_NAME]["resolutions"]["quarterhour"]["mae"] == 2.0