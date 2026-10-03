SELECT 'fct_energy_generation_load' AS fact_table, timestamp
FROM {{ ref('fct_energy_generation_load') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_market_prices' AS fact_table, timestamp
FROM {{ ref('fct_market_prices') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_price_spreads' AS fact_table, timestamp
FROM {{ ref('fct_price_spreads') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_weather' AS fact_table, timestamp
FROM {{ ref('fct_weather') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_ml_features' AS fact_table, timestamp
FROM {{ ref('fct_ml_features') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_energy_generation_load_forecast' AS fact_table, timestamp
FROM {{ ref('fct_energy_generation_load_forecast') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_weather_forecast' AS fact_table, timestamp
FROM {{ ref('fct_weather_forecast') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_weather_forecast_features' AS fact_table, timestamp
FROM {{ ref('fct_weather_forecast_features') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_ml_hourly_price_model_features' AS fact_table, timestamp
FROM {{ ref('fct_ml_hourly_price_model_features') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_ml_quarter_hourly_price_model_features' AS fact_table, timestamp
FROM {{ ref('fct_ml_quarter_hourly_price_model_features') }}
WHERE (timestamp AT TIME ZONE 'Europe/Berlin')::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})