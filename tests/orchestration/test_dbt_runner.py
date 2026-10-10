from subprocess import CompletedProcess

import pytest

from orchestration import dbt_runner


def test_require_clean_dbt_definition_rejects_uncommitted_changes(monkeypatch):
    monkeypatch.setattr(
        dbt_runner.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, " M dbt/models/analytics/dim_date.sql\n"),
    )

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        dbt_runner._require_clean_dbt_definition()