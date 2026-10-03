-- SMARD generation and load signals (actuals)

WITH nan_to_null AS(
    SELECT
        timestamp::TIMESTAMP WITH TIME ZONE,
        signal AS signal_name,
        NULLIF(value, 'NaN')::numeric AS value,
        unit,
        resolution
    FROM
        {{ source('raw', 'smard_generation') }}
)
-- Other generation signal can not be negative
SELECT *
FROM nan_to_null
WHERE value IS NOT NULL
  AND (
        -- Normal generation signals must be non-negative
        (
            signal_name NOT IN (
                'NUCLEAR',
                'PUMPED_STORAGE',
                'TOTAL_CONSUMPTION',
                'RESIDUAL_LOAD'
            )
            AND value >= 0
        )

        -- Nuclear is valid only before retirement
        OR (
            signal_name = 'NUCLEAR'
            AND timestamp <= '{{ var("nuclear_retirement_date") }}'::timestamptz
            AND value >= 0
        )

        -- These signals may be negative
        OR signal_name IN (
            'PUMPED_STORAGE',
            'TOTAL_CONSUMPTION',
            'RESIDUAL_LOAD'
        )
      )