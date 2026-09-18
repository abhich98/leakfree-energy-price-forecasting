import os

import boto3


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None,
    )


def get_bucket_name() -> str:
    bucket = os.environ.get("ZEPHYRWERK_AWS_BUCKET_NAME")
    if not bucket:
        raise ValueError("ZEPHYRWERK_AWS_BUCKET_NAME environment variable is not set.")
    return bucket