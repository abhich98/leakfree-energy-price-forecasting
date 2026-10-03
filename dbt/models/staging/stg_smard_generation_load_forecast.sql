-- SMARD forecasted generation and load signals (filters 122, 123, 125, 3791, 5097, etc.).
-- Some published before the day-ahead auction — leak-safe features for day-ahead
-- price forecasting. Essential for Approach B (residual load -> price).

WITH nan_to_null AS(
    SELECT
        timestamp :: TIMESTAMP WITH TIME ZONE,
        issue_timestamp :: TIMESTAMP WITH TIME ZONE,
        signal AS signal_name,
        NULLIF(value, 'NaN')::numeric AS value,
        unit,
        resolution,
        fetched_at :: TIMESTAMP WITH TIME ZONE
    FROM
        {{ source('raw', 'smard_forecast') }}
)
SELECT
    *
FROM
    nan_to_null
WHERE value IS NOT NULL
  AND (
        signal_name IN (
            'TOTAL_CONSUMPTION_FC',
            'RESIDUAL_LOAD_FC'
        )
        -- generation forecasts must be non-negative
        OR value >= 0
      )