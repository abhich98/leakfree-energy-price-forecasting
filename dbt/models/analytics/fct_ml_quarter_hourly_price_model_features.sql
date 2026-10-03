WITH quarter_hourly AS (
    SELECT
        ml.*,
        ml.timestamp AT TIME ZONE 'Europe/Berlin' AS local_timestamp,
        (ml.timestamp AT TIME ZONE 'Europe/Berlin')::date AS delivery_date,
        ROW_NUMBER() OVER (
            PARTITION BY (ml.timestamp AT TIME ZONE 'Europe/Berlin')::date
            ORDER BY ml.timestamp
        )::integer AS horizon,
        (
            EXTRACT(MINUTE FROM ml.timestamp AT TIME ZONE 'Europe/Berlin')::integer / 15
        )::integer AS quarter_of_hour
    FROM {{ ref('fct_ml_features') }} AS ml
    WHERE ml.resolution = 'quarterhour'
),
lagged AS (
    SELECT
        target.*,

        p1.price_eur_mwh AS price_lag_96qh,
        p2.price_eur_mwh AS price_lag_192qh,
        p7.price_eur_mwh AS price_lag_672qh,

        p1.austria_price_eur_mwh AS austria_price_lag_96qh,
        p1.austria_spread_eur_mwh AS austria_spread_lag_96qh,
        p1.france_price_eur_mwh AS france_price_lag_96qh,
        p1.france_spread_eur_mwh AS france_spread_lag_96qh,
        p1.netherlands_price_eur_mwh AS netherlands_price_lag_96qh,
        p1.netherlands_spread_eur_mwh AS netherlands_spread_lag_96qh,
        p1.poland_price_eur_mwh AS poland_price_lag_96qh,
        p1.poland_spread_eur_mwh AS poland_spread_lag_96qh,
        p1.switzerland_price_eur_mwh AS switzerland_price_lag_96qh,
        p1.switzerland_spread_eur_mwh AS switzerland_spread_lag_96qh,
        p1.czechia_price_eur_mwh AS czechia_price_lag_96qh,
        p1.czechia_spread_eur_mwh AS czechia_spread_lag_96qh,
        p1.denmark_1_price_eur_mwh AS denmark_1_price_lag_96qh,
        p1.denmark_1_spread_eur_mwh AS denmark_1_spread_lag_96qh,
        p1.denmark_2_price_eur_mwh AS denmark_2_price_lag_96qh,
        p1.denmark_2_spread_eur_mwh AS denmark_2_spread_lag_96qh,

        p2.residual_load_mw AS residual_load_lag_192qh,
        p2.total_consumption_mw AS total_consumption_lag_192qh,
        p2.wind_onshore_mw AS wind_onshore_lag_192qh,
        p2.wind_offshore_mw AS wind_offshore_lag_192qh,
        p2.solar_mw AS solar_lag_192qh,

        p1.wind_onshore_forecast_mw AS wind_onshore_forecast_mw_lag_96qh,
        p1.wind_offshore_forecast_mw AS wind_offshore_forecast_mw_lag_96qh,
        p1.solar_forecast_mw AS solar_forecast_mw_lag_96qh,
        p1.residual_load_forecast_mw AS residual_load_forecast_mw_lag_96qh,

        p2.wind_speed_100m_brandenburg_forecast - p2.wind_speed_100m_brandenburg AS wind_speed_100m_brandenburg_error_lag_192qh,
        p2.wind_speed_100m_schleswig_forecast - p2.wind_speed_100m_schleswig AS wind_speed_100m_schleswig_error_lag_192qh,
        p2.wind_speed_100m_bavaria_forecast - p2.wind_speed_100m_bavaria AS wind_speed_100m_bavaria_error_lag_192qh,
        p2.wind_speed_100m_bawue_forecast - p2.wind_speed_100m_bawue AS wind_speed_100m_bawue_error_lag_192qh,

        SIN(RADIANS(p2.wind_direction_100m_brandenburg_forecast - p2.wind_direction_100m_brandenburg)) AS wind_direction_100m_brandenburg_error_sin_lag_192qh,
        COS(RADIANS(p2.wind_direction_100m_brandenburg_forecast - p2.wind_direction_100m_brandenburg)) AS wind_direction_100m_brandenburg_error_cos_lag_192qh,
        SIN(RADIANS(p2.wind_direction_100m_schleswig_forecast - p2.wind_direction_100m_schleswig)) AS wind_direction_100m_schleswig_error_sin_lag_192qh,
        COS(RADIANS(p2.wind_direction_100m_schleswig_forecast - p2.wind_direction_100m_schleswig)) AS wind_direction_100m_schleswig_error_cos_lag_192qh,
        SIN(RADIANS(p2.wind_direction_100m_bavaria_forecast - p2.wind_direction_100m_bavaria)) AS wind_direction_100m_bavaria_error_sin_lag_192qh,
        COS(RADIANS(p2.wind_direction_100m_bavaria_forecast - p2.wind_direction_100m_bavaria)) AS wind_direction_100m_bavaria_error_cos_lag_192qh,
        SIN(RADIANS(p2.wind_direction_100m_bawue_forecast - p2.wind_direction_100m_bawue)) AS wind_direction_100m_bawue_error_sin_lag_192qh,
        COS(RADIANS(p2.wind_direction_100m_bawue_forecast - p2.wind_direction_100m_bawue)) AS wind_direction_100m_bawue_error_cos_lag_192qh,

        p2.shortwave_radiation_brandenburg_forecast - p2.shortwave_radiation_brandenburg AS shortwave_radiation_brandenburg_error_lag_192qh,
        p2.shortwave_radiation_schleswig_forecast - p2.shortwave_radiation_schleswig AS shortwave_radiation_schleswig_error_lag_192qh,
        p2.shortwave_radiation_bavaria_forecast - p2.shortwave_radiation_bavaria AS shortwave_radiation_bavaria_error_lag_192qh,
        p2.shortwave_radiation_bawue_forecast - p2.shortwave_radiation_bawue AS shortwave_radiation_bawue_error_lag_192qh,

        p2.cloud_cover_brandenburg_forecast - p2.cloud_cover_brandenburg AS cloud_cover_brandenburg_error_lag_192qh,
        p2.cloud_cover_schleswig_forecast - p2.cloud_cover_schleswig AS cloud_cover_schleswig_error_lag_192qh,
        p2.cloud_cover_bavaria_forecast - p2.cloud_cover_bavaria AS cloud_cover_bavaria_error_lag_192qh,
        p2.cloud_cover_bawue_forecast - p2.cloud_cover_bawue AS cloud_cover_bawue_error_lag_192qh,

        p2.temperature_2m_brandenburg_forecast - p2.temperature_2m_brandenburg AS temperature_2m_brandenburg_error_lag_192qh,
        p2.temperature_2m_schleswig_forecast - p2.temperature_2m_schleswig AS temperature_2m_schleswig_error_lag_192qh,
        p2.temperature_2m_bavaria_forecast - p2.temperature_2m_bavaria AS temperature_2m_bavaria_error_lag_192qh,
        p2.temperature_2m_bawue_forecast - p2.temperature_2m_bawue AS temperature_2m_bawue_error_lag_192qh,

        p2.total_consumption_forecast_mw - p2.total_consumption_mw AS total_consumption_error_mw_lag_192qh
    FROM quarter_hourly AS target
    LEFT JOIN quarter_hourly AS p1
        ON p1.timestamp = target.timestamp - INTERVAL '24 hours'
    LEFT JOIN quarter_hourly AS p2
        ON p2.timestamp = target.timestamp - INTERVAL '48 hours'
    LEFT JOIN quarter_hourly AS p7
        ON p7.timestamp = target.timestamp - INTERVAL '168 hours'
)
SELECT
    timestamp,
    resolution,
    local_timestamp,
    delivery_date,
    horizon,
    quarter_of_hour,
    price_eur_mwh,

    year,
    quarter,
    month,
    week_of_year,
    day_of_week,
    is_weekend,
    is_german_public_holiday,
    is_workday,

    price_lag_96qh,
    price_lag_192qh,
    price_lag_672qh,

    austria_price_lag_96qh,
    france_price_lag_96qh,
    netherlands_price_lag_96qh,
    poland_price_lag_96qh,
    switzerland_price_lag_96qh,
    czechia_price_lag_96qh,
    denmark_1_price_lag_96qh,
    denmark_2_price_lag_96qh,

    wind_speed_100m_brandenburg_forecast,
    wind_speed_100m_schleswig_forecast,
    wind_speed_100m_bavaria_forecast,
    wind_speed_100m_bawue_forecast,
    SIN(RADIANS(wind_direction_100m_brandenburg_forecast)) AS wind_direction_100m_brandenburg_forecast_sin,
    COS(RADIANS(wind_direction_100m_brandenburg_forecast)) AS wind_direction_100m_brandenburg_forecast_cos,
    SIN(RADIANS(wind_direction_100m_schleswig_forecast)) AS wind_direction_100m_schleswig_forecast_sin,
    COS(RADIANS(wind_direction_100m_schleswig_forecast)) AS wind_direction_100m_schleswig_forecast_cos,
    SIN(RADIANS(wind_direction_100m_bavaria_forecast)) AS wind_direction_100m_bavaria_forecast_sin,
    COS(RADIANS(wind_direction_100m_bavaria_forecast)) AS wind_direction_100m_bavaria_forecast_cos,
    SIN(RADIANS(wind_direction_100m_bawue_forecast)) AS wind_direction_100m_bawue_forecast_sin,
    COS(RADIANS(wind_direction_100m_bawue_forecast)) AS wind_direction_100m_bawue_forecast_cos,
    shortwave_radiation_brandenburg_forecast,
    shortwave_radiation_schleswig_forecast,
    shortwave_radiation_bavaria_forecast,
    shortwave_radiation_bawue_forecast,
    cloud_cover_brandenburg_forecast,
    cloud_cover_schleswig_forecast,
    cloud_cover_bavaria_forecast,
    cloud_cover_bawue_forecast,
    temperature_2m_brandenburg_forecast,
    temperature_2m_schleswig_forecast,
    temperature_2m_bavaria_forecast,
    temperature_2m_bawue_forecast,

    residual_load_lag_192qh,
    total_consumption_lag_192qh,
    wind_onshore_lag_192qh,
    wind_offshore_lag_192qh,
    solar_lag_192qh,

    total_consumption_forecast_mw,
    total_consumption_error_mw_lag_192qh,
    wind_onshore_forecast_mw_lag_96qh,
    wind_offshore_forecast_mw_lag_96qh,
    solar_forecast_mw_lag_96qh,
    residual_load_forecast_mw_lag_96qh,

    wind_speed_100m_brandenburg_error_lag_192qh,
    wind_speed_100m_schleswig_error_lag_192qh,
    wind_speed_100m_bavaria_error_lag_192qh,
    wind_speed_100m_bawue_error_lag_192qh,
    wind_direction_100m_brandenburg_error_sin_lag_192qh,
    wind_direction_100m_brandenburg_error_cos_lag_192qh,
    wind_direction_100m_schleswig_error_sin_lag_192qh,
    wind_direction_100m_schleswig_error_cos_lag_192qh,
    wind_direction_100m_bavaria_error_sin_lag_192qh,
    wind_direction_100m_bavaria_error_cos_lag_192qh,
    wind_direction_100m_bawue_error_sin_lag_192qh,
    wind_direction_100m_bawue_error_cos_lag_192qh,
    shortwave_radiation_brandenburg_error_lag_192qh,
    shortwave_radiation_schleswig_error_lag_192qh,
    shortwave_radiation_bavaria_error_lag_192qh,
    shortwave_radiation_bawue_error_lag_192qh,
    cloud_cover_brandenburg_error_lag_192qh,
    cloud_cover_schleswig_error_lag_192qh,
    cloud_cover_bavaria_error_lag_192qh,
    cloud_cover_bawue_error_lag_192qh,
    temperature_2m_brandenburg_error_lag_192qh,
    temperature_2m_schleswig_error_lag_192qh,
    temperature_2m_bavaria_error_lag_192qh,
    temperature_2m_bawue_error_lag_192qh
FROM lagged
