import subprocess
from typing import Any

import wandb


def _git_output(*args: str, cwd: str | None = None) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def get_git_metadata() -> dict[str, str | bool | None]:
    """Return Git provenance for the current checkout without raising outside Git."""
    try:
        repository_root = _git_output("rev-parse", "--show-toplevel")
        commit_sha = _git_output("rev-parse", "HEAD", cwd=repository_root)
        short_commit_sha = _git_output(
            "rev-parse", "--short", "HEAD", cwd=repository_root
        )
        is_dirty = bool(_git_output("status", "--porcelain", cwd=repository_root))
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}

    try:
        branch = _git_output(
            "symbolic-ref", "--short", "-q", "HEAD", cwd=repository_root
        )
    except subprocess.CalledProcessError:
        branch = None

    return {
        "available": True,
        "commit_sha": commit_sha,
        "short_commit_sha": short_commit_sha,
        "branch": branch or None,
        "is_dirty": is_dirty,
        "repository_root": repository_root,
    }


def start_wandb_run(run_name: str, group: str | None = None) -> Any:
    """Start an online W&B run with application-owned Git provenance."""
    return wandb.init(
        mode="online",
        project="zephyrwerk-platform-forecasting",
        name=run_name,
        group=group,
        config={"git": get_git_metadata()},
        settings=wandb.Settings(
            mode="online",
            console="wrap",
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )


def update_wandb_config(run: Any, values: dict[str, Any]) -> None:
    """Update lightweight, reproducibility-relevant W&B configuration."""
    run.config.update(values, allow_val_change=True)


def log_wandb_model_results(
    run,
    model_name: str,
    model_report: dict[str, Any],
) -> None:
    """Log selected scalar outcomes and durable references without uploading files."""
    config_fields = {
        field: model_report[field]
        for field in (
            "n_train",
            "n_holdout",
            "n_features",
            "hyperparameters",
            "hyperparameters_source",
            "artifacts",
        )
        if field in model_report
    }
    update_wandb_config(run, {f"models/{model_name}": config_fields})

    metrics: dict[str, float] = {}
    if "cv_mae" in model_report:
        metrics[f"{model_name}/cv_mae"] = float(model_report["cv_mae"])
    for metric_group, metric_prefix in (
        ("holdout", "holdout"),
        ("baseline_persistence", "baseline"),
    ):
        for metric_name, metric_value in (
            model_report.get("metrics", {}).get(metric_group, {}).items()
        ):
            if isinstance(metric_value, (int, float)):
                metrics[f"{model_name}/{metric_prefix}_{metric_name}"] = float(
                    metric_value
                )

    if metrics:
        run.log(metrics)
        for metric_name, metric_value in metrics.items():
            run.summary[metric_name] = metric_value
