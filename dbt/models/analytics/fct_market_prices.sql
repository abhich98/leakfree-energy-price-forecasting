-- Clean day-ahead prices. Lags and rolling averages have been REMOVED from
-- this model — they assumed hourly spacing (LAG(value, 24) = 24h) which breaks
-- at 15-min resolution (24 steps = 6h). All lag/rolling computations now live
-- in the ML feature engineer (ml/features/feature_engineering.py) where the
-- resolution is known and the correct step count can be applied.
SELECT
    timestamp,
    resolution,
    value AS price_eur_mwh
FROM {{ ref('stg_smard_prices')}}
