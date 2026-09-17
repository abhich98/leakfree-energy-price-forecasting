SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    region,
    signal_type AS signal_name,
    NULLIF(value, 'NaN') AS value,
    unit
FROM
    {{ source('raw', 'weather') }}
