SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    issue_timestamp :: TIMESTAMP WITH TIME ZONE,
    region,
    signal_type AS signal_name,
    NULLIF(value, 'NaN') AS value,
    unit,
    model,
    fetched_at :: TIMESTAMP WITH TIME ZONE
FROM
    {{ source('raw', 'weather_forecast') }}
