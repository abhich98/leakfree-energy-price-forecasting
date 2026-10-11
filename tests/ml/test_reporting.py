from ml.reporting import StandardReport


def test_standard_report_attaches_manifest_metadata_under_data():
    report = StandardReport(
        report_version="1.0",
        run={"name": "test", "started_at": "2026-09-26T00:00:00+00:00"},
        data={"hourly": {"start_date": "2026-01-01"}},
        sections={"models": {}},
    )

    manifest = {
        "data_version_id": "data-version-1",
        "raw_inventory": {"inventory_id": "inventory-1"},
    }
    report.attach_data_manifest(manifest, "ml/artifacts/manifest.json", "s3://bucket/manifest.json")

    payload = report.to_dict()
    assert payload["report_version"] == "1.0"
    assert payload["data"]["data_version_id"] == "data-version-1"
    assert payload["data"]["manifest_local_path"] == "ml/artifacts/manifest.json"
    assert payload["data"]["manifest_s3_uri"] == "s3://bucket/manifest.json"
    assert payload["data"]["raw_inventory"] == {"inventory_id": "inventory-1"}
    assert payload["models"] == {}