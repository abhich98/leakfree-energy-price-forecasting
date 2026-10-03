import ml.wandb_tracking as wandb_tracking


def test_start_wandb_run_logs_git_metadata_and_disables_machine_stats(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(wandb_tracking.wandb, "init", fake_init)
    monkeypatch.setattr(
        wandb_tracking,
        "get_git_metadata",
        lambda: {"available": True, "commit_sha": "abc123"},
    )

    run = wandb_tracking.start_wandb_run("test-run", group="test-group")

    assert run is not None
    assert captured["mode"] == "online"
    assert captured["project"] == "zephyrwerk-platform-forecasting"
    assert captured["name"] == "test-run"
    assert captured["group"] == "test-group"
    assert captured["config"] == {"git": {"available": True, "commit_sha": "abc123"}}
    assert captured["settings"].mode == "online"
    assert captured["settings"].console == "wrap"
    assert captured["settings"].x_disable_stats is True
    assert captured["settings"].x_disable_machine_info is True


def test_get_git_metadata_handles_non_git_environment(monkeypatch):
    def raise_git_error(*args, **kwargs):
        raise wandb_tracking.subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(wandb_tracking, "_git_output", raise_git_error)

    assert wandb_tracking.get_git_metadata() == {"available": False}


def test_log_wandb_model_results_logs_scalars_and_references_only():
    class FakeConfig:
        def __init__(self):
            self.values = None

        def update(self, values, allow_val_change):
            self.values = values

    class FakeRun:
        def __init__(self):
            self.config = FakeConfig()
            self.logged = None
            self.summary = {}

        def log(self, metrics):
            self.logged = metrics

    run = FakeRun()
    model_report = {
        "n_train": 100,
        "n_holdout": 20,
        "n_features": 12,
        "hyperparameters": {"max_depth": 6},
        "cv_mae": 4.0,
        "metrics": {
            "holdout": {"mae": 5.0, "rmse": 6.0},
            "baseline_persistence": {"mae": 7.0},
        },
        "artifacts": {"pipeline": "s3://models/hourly.joblib"},
    }

    wandb_tracking.log_wandb_model_results(run, "price_hourly", model_report)

    assert run.config.values == {
        "models/price_hourly": {
            "n_train": 100,
            "n_holdout": 20,
            "n_features": 12,
            "hyperparameters": {"max_depth": 6},
            "artifacts": {"pipeline": "s3://models/hourly.joblib"},
        }
    }
    assert run.logged == {
        "price_hourly/cv_mae": 4.0,
        "price_hourly/holdout_mae": 5.0,
        "price_hourly/holdout_rmse": 6.0,
        "price_hourly/baseline_mae": 7.0,
    }
