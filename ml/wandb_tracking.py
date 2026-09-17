import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import wandb

from ml.training_utils import ModelType


@dataclass(frozen=True)
class WandbConfig:
    project: str
    run_name_prefix: str
    run_name_suffix: str
    log_model_artifact: bool
    log_prediction_plot: bool


def load_wandb_config(path: Path | None = None) -> WandbConfig:
    """Load W&B tracking settings from the repository TOML configuration."""
    config_path = path or Path(__file__).with_name("wandb_config.toml")
    with config_path.open("rb") as config_file:
        values = tomllib.load(config_file)["wandb"]

    return WandbConfig(
        project=values["project"],
        run_name_prefix=values.get("run_name_prefix", ""),
        run_name_suffix=values.get("run_name_suffix", ""),
        log_model_artifact=values.get("log_model_artifact", False),
        log_prediction_plot=values.get("log_prediction_plot", False),
    )


def start_wandb_run(run_name: str, group: str | None = None) -> wandb.sdk.wandb_run.Run:
    """Start an online W&B run without collecting machine statistics."""
    return wandb.init(
        mode="online",
        project="zephyrwerk-platform-forecasting",
        name=run_name,
        group=group,
        settings=wandb.Settings(
            mode="online",
            console="wrap",
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )


def log_training_report(run, report: dict[str, object]) -> None:
    """Log training configuration and scalar evaluation metrics to W&B."""
    run.config.update(
        {
            "n_train": report["n_train"],
            "n_test": report["n_test"],
            "train_window": report["train_window"],
            "test_window": report["test_window"],
            "n_features": report["n_features"],
            "hyperparameters": report["hyperparameters"],
        },
        allow_val_change=True,
    )

    metrics: dict[str, float] = {
        "cv/mae_mean": report["cv_mae_mean"],
        "cv/mae_std": report["cv_mae_std"],
    }
    metrics.update({
        f"cv/fold_{fold_number}_mae": value
        for fold_number, value in enumerate(report["cv_mae_per_fold"], start=1)
    })

    for section_name in ("holdout", "baseline_persistence"):
        section = report[section_name]
        metrics.update({f"{section_name}/{key}": value for key, value in section.items()})

    run.log(metrics)


def log_optional_artifacts(run, 
                           pipeline, 
                           mode: ModelType, 
                           log_model_artifact: bool = False, 
                           log_prediction_plot: bool = False) -> None:
    """Upload the optional pipeline and holdout plot artifacts to W&B."""
    artifacts_dir = Path("ml/artifacts")

    if log_model_artifact:
        model_path = artifacts_dir / f"{mode.value}_forecast_wandb.joblib"
        joblib.dump(pipeline, model_path)
        artifact = wandb.Artifact(f"{mode.value}-forecast-pipeline", type="model")
        artifact.add_file(str(model_path))
        run.log_artifact(artifact)

    if log_prediction_plot:
        plot_path = artifacts_dir / f"{mode.value}_holdout.png"
        run.log({"holdout/prediction_plot": wandb.Image(str(plot_path))})
