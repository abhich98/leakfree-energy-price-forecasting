import logging

import pandas as pd
from sqlalchemy import text

from db.database import engine

logger = logging.getLogger(__name__)


def load_ml_features(
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    raise NotImplementedError


def load_hourly_price_model_features(
    start_date: str | None = None,
    end_date_exclusive: str | None = None,
    filter_by_local_timestamp: bool = True,
) -> pd.DataFrame:
    """Load the hourly price-model contract from the analytics schema.
    Setting filter_by_local_timestamp to True will use the local_timestamp column, which is the timestamp in the local timezone (Europe/Berlin), for filtering.
    """
    logger.info(
        "Loading hourly price-model features from %s to %s (exclusive) (filter_by_local_timestamp=%s)",
        start_date,
        end_date_exclusive,
        filter_by_local_timestamp,
    )

    timestamp_column = "local_timestamp" if filter_by_local_timestamp else "timestamp"

    try:
        query = "SELECT * FROM analytics.fct_ml_hourly_price_model_features WHERE 1=1"
        params: dict = {}

        if start_date:
            query += f" AND {timestamp_column} >= :start_date"
            params["start_date"] = start_date
        if end_date_exclusive:
            query += f" AND {timestamp_column} < :end_date_exclusive"
            params["end_date_exclusive"] = end_date_exclusive

        query += f" ORDER BY {timestamp_column} ASC"

        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
            logger.info("Loaded %s hourly price-model rows from the database.", len(df))

        if start_date is not None and end_date_exclusive is not None:
            logger.info(
                "Number of days queried: %s vs returned: %s",
                (pd.Timestamp(end_date_exclusive) - pd.Timestamp(start_date)).days,
                df[timestamp_column].dt.floor("d").nunique(),
            )

        return df.set_index(timestamp_column).sort_index()

    except Exception:
        logger.exception("Failed to load hourly price-model features")
        raise


def load_quarter_hourly_price_model_features(
    start_date: str | None = None,
    end_date_exclusive: str | None = None,
    filter_by_local_timestamp: bool = True,
) -> pd.DataFrame:
    """Load the quarter-hourly price-model contract from the analytics schema."""
    logger.info(
        "Loading quarter-hour price-model features from %s to %s (exclusive) "
        "(filter_by_local_timestamp=%s)",
        start_date,
        end_date_exclusive,
        filter_by_local_timestamp,
    )

    timestamp_column = "local_timestamp" if filter_by_local_timestamp else "timestamp"

    try:
        query = "SELECT * FROM analytics.fct_ml_quarter_hourly_price_model_features WHERE 1=1"
        params: dict = {}

        if start_date:
            query += f" AND {timestamp_column} >= :start_date"
            params["start_date"] = start_date
        if end_date_exclusive:
            query += f" AND {timestamp_column} < :end_date_exclusive"
            params["end_date_exclusive"] = end_date_exclusive

        query += f" ORDER BY {timestamp_column} ASC"

        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
            logger.info(
                "Loaded %s quarter-hour price-model rows from the database.", len(df)
            )

        if start_date is not None and end_date_exclusive is not None:
            logger.info(
                "Number of days queried: %s vs returned: %s",
                (pd.Timestamp(end_date_exclusive) - pd.Timestamp(start_date)).days,
                df[timestamp_column].dt.floor("d").nunique(),
            )
        return df.set_index(timestamp_column).sort_index()

    except Exception:
        logger.exception("Failed to load quarter-hour price-model features")
        raise


if __name__ == "__main__":
    # Example usage
    df = load_ml_features(start_date="2023-04-16")
    print(df.head())
    print(df.shape, df.index.dtype, df.index.min(), df.index.max())
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print(df["nuclear_mw"].describe())
    print(df[df["nuclear_mw"].notna()]["nuclear_mw"].value_counts().head())
