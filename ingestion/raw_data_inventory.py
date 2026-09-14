import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

RAW_INVENTORY_LATEST_KEY = "data/raw_inventory/latest.json"
RAW_INVENTORY_ARCHIVE_PREFIX = "data/raw_inventory/archive"


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None,
    )


def _get_bucket_name() -> str:
    bucket = os.environ.get("ZEPHYRWERK_AWS_BUCKET_NAME")
    if not bucket:
        raise ValueError("ZEPHYRWERK_AWS_BUCKET_NAME environment variable is not set.")
    return bucket


def _load_latest_inventory(s3, bucket: str) -> dict[str, Any]:
    try:
        response = s3.get_object(Bucket=bucket, Key=RAW_INVENTORY_LATEST_KEY)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            return {"objects": {}}
        raise
    return json.loads(response["Body"].read())


def _inventory_id(objects: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(objects, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inventory_existing_raw_objects(s3, bucket: str) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="raw/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            response = s3.head_object(Bucket=bucket, Key=key)
            objects[key] = {
                "key": key,
                "version_id": response.get("VersionId"),
                "etag": response.get("ETag"),
                "size_bytes": response.get("ContentLength"),
                "last_modified": response.get("LastModified").isoformat(),
            }
    return objects


def update_raw_data_inventory(uploaded_objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge uploaded raw object versions into a new immutable inventory snapshot."""
    bucket = _get_bucket_name()
    s3 = _get_s3_client()
    latest_inventory = _load_latest_inventory(s3, bucket)
    objects = latest_inventory.get("objects", {}).copy()
    if not latest_inventory.get("inventory_id"):
        objects = _inventory_existing_raw_objects(s3, bucket)
    elif not uploaded_objects:
        return load_latest_raw_data_inventory()

    for uploaded_object in uploaded_objects:
        objects[uploaded_object["key"]] = uploaded_object

    inventory_id = _inventory_id(objects)
    archived_key = f"{RAW_INVENTORY_ARCHIVE_PREFIX}/{inventory_id}.json"
    inventory = {
        "inventory_version": "1.0",
        "inventory_id": inventory_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objects": objects,
    }
    body = json.dumps(inventory, indent=2).encode("utf-8")

    try:
        s3.head_object(Bucket=bucket, Key=archived_key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
            raise
        s3.put_object(Bucket=bucket, Key=archived_key, Body=body)

    s3.copy_object(
        Bucket=bucket,
        Key=RAW_INVENTORY_LATEST_KEY,
        CopySource={"Bucket": bucket, "Key": archived_key},
    )
    inventory["s3_uri"] = f"s3://{bucket}/{archived_key}"
    return inventory


def load_latest_raw_data_inventory() -> dict[str, Any]:
    """Load the latest raw object inventory and return its immutable archive URI."""
    bucket = _get_bucket_name()
    inventory = _load_latest_inventory(_get_s3_client(), bucket)
    inventory_id = inventory.get("inventory_id")
    if not inventory_id:
        raise RuntimeError("No raw data inventory exists. Run ingestion before training.")
    inventory["s3_uri"] = f"s3://{bucket}/{RAW_INVENTORY_ARCHIVE_PREFIX}/{inventory_id}.json"
    return inventory
