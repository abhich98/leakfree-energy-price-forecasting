from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from orchestration import run_ingestion

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 2, tzinfo=timezone.utc)


def test_run_pipeline_records_successful_stages_and_inventory():
    run_id = uuid4()
    stage_run_ids = [uuid4(), uuid4(), uuid4(), uuid4()]
    uploaded_objects = [{"key": "raw/smard/example.parquet"}]
    inventory = {"inventory_id": "inventory-1", "s3_uri": "s3://bucket/inventory-1.json"}
    dbt_artifacts = {"manifest_uri": "s3://bucket/manifest.json"}

    with (
        patch("orchestration.run_ingestion.start_pipeline_run", return_value=run_id),
        patch("orchestration.run_ingestion.start_stage_run", side_effect=stage_run_ids) as start_stage,
        patch("orchestration.run_ingestion.finish_stage_run") as finish_stage,
        patch("orchestration.run_ingestion.finish_pipeline_run") as finish_pipeline,
        patch("orchestration.run_ingestion.ingest_and_upload", return_value=uploaded_objects),
        patch("orchestration.run_ingestion.record_raw_inventory", return_value=inventory),
        patch("orchestration.run_ingestion.load_postgres") as load_postgres,
        patch("orchestration.run_ingestion.run_dbt", return_value=dbt_artifacts) as run_dbt,
    ):
        run_ingestion.run_pipeline(START, END)

    assert [call.args[1] for call in start_stage.call_args_list] == [
        "ingest_and_upload",
        "record_raw_inventory",
        "load_postgres",
        "run_dbt",
    ]
    assert finish_stage.call_count == 4
    load_postgres.assert_called_once_with(START, END)
    run_dbt.assert_called_once_with(run_id, START, END + timedelta(seconds=1))
    finish_pipeline.assert_called_once_with(
        run_id,
        "succeeded",
        raw_inventory_uri=inventory["s3_uri"],
    )


def test_run_pipeline_records_failure_and_stops_following_stages():
    run_id = uuid4()
    stage_run_id = uuid4()

    with (
        patch("orchestration.run_ingestion.start_pipeline_run", return_value=run_id),
        patch("orchestration.run_ingestion.start_stage_run", return_value=stage_run_id),
        patch("orchestration.run_ingestion.finish_stage_run") as finish_stage,
        patch("orchestration.run_ingestion.finish_pipeline_run") as finish_pipeline,
        patch("orchestration.run_ingestion.ingest_and_upload", side_effect=RuntimeError("source unavailable")),
        patch("orchestration.run_ingestion.record_raw_inventory") as record_inventory,
        patch("orchestration.run_ingestion.load_postgres") as load_postgres,
        patch("orchestration.run_ingestion.run_dbt") as run_dbt,
    ):
        with pytest.raises(RuntimeError, match="source unavailable"):
            run_ingestion.run_pipeline(START, END)

    finish_stage.assert_called_once_with(
        stage_run_id,
        "failed",
        error_message="source unavailable",
    )
    finish_pipeline.assert_called_once_with(
        run_id,
        "failed",
        raw_inventory_uri=None,
        error_message="source unavailable",
    )
    record_inventory.assert_not_called()
    load_postgres.assert_not_called()
    run_dbt.assert_not_called()


def test_resume_pipeline_restarts_at_failed_load_and_runs_downstream_stages():
    run_id = uuid4()
    stage_run_ids = [uuid4(), uuid4()]
    pipeline_run = {
        "pipeline_name": "elt",
        "requested_start": START,
        "requested_end": END,
        "raw_inventory_uri": "s3://bucket/inventory-1.json",
    }
    dbt_artifacts = {"manifest_uri": "s3://bucket/manifest.json"}
    statuses = {
        "ingest_and_upload": "succeeded",
        "record_raw_inventory": "succeeded",
        "load_postgres": "failed",
        "run_dbt": None,
    }

    with (
        patch("orchestration.run_ingestion.get_pipeline_run", return_value=pipeline_run),
        patch("orchestration.run_ingestion.get_latest_stage_status", side_effect=lambda _, name: statuses[name]),
        patch("orchestration.run_ingestion.get_latest_stage_details", return_value={"uploaded_objects": []}),
        patch("orchestration.run_ingestion.reopen_pipeline_run") as reopen_pipeline,
        patch("orchestration.run_ingestion.start_stage_run", side_effect=stage_run_ids) as start_stage,
        patch("orchestration.run_ingestion.finish_stage_run") as finish_stage,
        patch("orchestration.run_ingestion.finish_pipeline_run") as finish_pipeline,
        patch("orchestration.run_ingestion.ingest_and_upload") as ingest_and_upload,
        patch("orchestration.run_ingestion.record_raw_inventory") as record_raw_inventory,
        patch("orchestration.run_ingestion.load_postgres") as load_postgres,
        patch("orchestration.run_ingestion.run_dbt", return_value=dbt_artifacts) as run_dbt,
    ):
        run_ingestion.resume_pipeline(run_id)

    reopen_pipeline.assert_called_once_with(run_id)
    assert [call.args[1] for call in start_stage.call_args_list] == ["load_postgres", "run_dbt"]
    ingest_and_upload.assert_not_called()
    record_raw_inventory.assert_not_called()
    load_postgres.assert_called_once_with(START, END)
    run_dbt.assert_called_once_with(run_id, START, END + timedelta(seconds=1))
    assert finish_stage.call_count == 2
    finish_pipeline.assert_called_once_with(
        run_id,
        "succeeded",
        raw_inventory_uri=pipeline_run["raw_inventory_uri"],
    )


def test_resume_pipeline_retries_only_dbt_when_it_is_the_first_failed_stage():
    run_id = uuid4()
    pipeline_run = {
        "pipeline_name": "elt",
        "requested_start": START,
        "requested_end": END,
        "raw_inventory_uri": "s3://bucket/inventory-1.json",
    }
    statuses = {
        "ingest_and_upload": "succeeded",
        "record_raw_inventory": "succeeded",
        "load_postgres": "succeeded",
        "run_dbt": "failed",
    }
    with (
        patch("orchestration.run_ingestion.get_pipeline_run", return_value=pipeline_run),
        patch("orchestration.run_ingestion.get_latest_stage_status", side_effect=lambda _, name: statuses[name]),
        patch("orchestration.run_ingestion.get_latest_stage_details", return_value={"uploaded_objects": []}),
        patch("orchestration.run_ingestion.reopen_pipeline_run"),
        patch("orchestration.run_ingestion.start_stage_run", return_value=uuid4()) as start_stage,
        patch("orchestration.run_ingestion.finish_stage_run"),
        patch("orchestration.run_ingestion.finish_pipeline_run"),
        patch("orchestration.run_ingestion.load_postgres") as load_postgres,
        patch(
            "orchestration.run_ingestion.run_dbt",
            return_value={"manifest_uri": "s3://bucket/manifest.json"},
        ) as run_dbt,
    ):
        run_ingestion.resume_pipeline(run_id)

    assert [call.args[1] for call in start_stage.call_args_list] == ["run_dbt"]
    load_postgres.assert_not_called()
    run_dbt.assert_called_once_with(run_id, START, END + timedelta(seconds=1))


def test_resume_pipeline_rejects_non_elt_runs():
    run_id = uuid4()
    with (
        patch("orchestration.run_ingestion.get_pipeline_run", return_value={"pipeline_name": "forecast"}),
        patch("orchestration.run_ingestion.reopen_pipeline_run") as reopen_pipeline,
    ):
        with pytest.raises(ValueError, match="not an ELT run"):
            run_ingestion.resume_pipeline(run_id)

    reopen_pipeline.assert_not_called()