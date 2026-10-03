WITH hourly AS (
    SELECT
        ml.*,
        ml.timestamp AT TIME ZONE 'Europe/Berlin' AS local_timestamp,
        (ml.timestamp AT TIME ZONE 'Europe/Berlin')::date AS delivery_date,
        ROW_NUMBER() OVER (
            PARTITION BY (ml.timestamp AT TIME ZONE 'Europe/Berlin')::date
            ORDER BY ml.timestamp
        )::integer AS horizon
    FROM {{ ref('fct_ml_features') }} AS ml
    WHERE ml.resolution = 'hour'
),
lagged AS (
    SELECT
        target.*,

        p1.price_eur_mwh AS price_lag_24h,
        p2.price_eur_mwh AS price_lag_48h,
        p7.price_eur_mwh AS price_lag_168h,
        (
            p1.price_eur_mwh
            + p2.price_eur_mwh
            + p3.price_eur_mwh
            + p4.price_eur_mwh
            + p5.price_eur_mwh
            + p6.price_eur_mwh
            + p7.price_eur_mwh
        ) / 7.0 AS price_rolling_avg_7d_same_hour,

        p1.austria_price_eur_mwh AS austria_price_lag_24h,
        p1.france_price_eur_mwh AS france_price_lag_24h,
        p1.netherlands_price_eur_mwh AS netherlands_price_lag_24h,
        p1.poland_price_eur_mwh AS poland_price_lag_24h,
        p1.switzerland_price_eur_mwh AS switzerland_price_lag_24h,
        p1.czechia_price_eur_mwh AS czechia_price_lag_24h,
        p1.denmark_1_price_eur_mwh AS denmark_1_price_lag_24h,
        p1.denmark_2_price_eur_mwh AS denmark_2_price_lag_24h,

        p2.residual_load_mw AS residual_load_lag_48h,
        p2.total_consumption_mw AS total_consumption_lag_48h,
        p2.wind_onshore_mw AS wind_onshore_lag_48h,
        p2.wind_offshore_mw AS wind_offshore_lag_48h,
        p2.solar_mw AS solar_lag_48h,

        -- p1.wind_onshore_forecast_mw AS wind_onshore_forecast_mw_lag_24h, A LOT OF DATA MISSING
        p1.wind_offshore_forecast_mw AS wind_offshore_forecast_mw_lag_24h,
        p1.solar_forecast_mw AS solar_forecast_mw_lag_24h,
        -- p1.residual_load_forecast_mw AS residual_load_forecast_mw_lag_24h,

        p2.wind_speed_100m_brandenburg_forecast
            - p2.wind_speed_100m_brandenburg AS wind_speed_100m_brandenburg_error_lag_48h,
        p2.wind_speed_100m_schleswig_forecast
            - p2.wind_speed_100m_schleswig AS wind_speed_100m_schleswig_error_lag_48h,
        p2.wind_speed_100m_bavaria_forecast
            - p2.wind_speed_100m_bavaria AS wind_speed_100m_bavaria_error_lag_48h,
        p2.wind_speed_100m_bawue_forecast
            - p2.wind_speed_100m_bawue AS wind_speed_100m_bawue_error_lag_48h,

        SIN(RADIANS(
            p2.wind_direction_100m_brandenburg_forecast
            - p2.wind_direction_100m_brandenburg
        )) AS wind_direction_100m_brandenburg_error_sin_lag_48h,
        COS(RADIANS(
            p2.wind_direction_100m_brandenburg_forecast
            - p2.wind_direction_100m_brandenburg
        )) AS wind_direction_100m_brandenburg_error_cos_lag_48h,
        SIN(RADIANS(
            p2.wind_direction_100m_schleswig_forecast
            - p2.wind_direction_100m_schleswig
        )) AS wind_direction_100m_schleswig_error_sin_lag_48h,
        COS(RADIANS(
            p2.wind_direction_100m_schleswig_forecast
            - p2.wind_direction_100m_schleswig
        )) AS wind_direction_100m_schleswig_error_cos_lag_48h,
        SIN(RADIANS(
            p2.wind_direction_100m_bavaria_forecast
            - p2.wind_direction_100m_bavaria
        )) AS wind_direction_100m_bavaria_error_sin_lag_48h,
        COS(RADIANS(
            p2.wind_direction_100m_bavaria_forecast
            - p2.wind_direction_100m_bavaria
        )) AS wind_direction_100m_bavaria_error_cos_lag_48h,
        SIN(RADIANS(
            p2.wind_direction_100m_bawue_forecast
            - p2.wind_direction_100m_bawue
        )) AS wind_direction_100m_bawue_error_sin_lag_48h,
        COS(RADIANS(
            p2.wind_direction_100m_bawue_forecast
            - p2.wind_direction_100m_bawue
        )) AS wind_direction_100m_bawue_error_cos_lag_48h,

        p2.shortwave_radiation_brandenburg_forecast
            - p2.shortwave_radiation_brandenburg AS shortwave_radiation_brandenburg_error_lag_48h,
        p2.shortwave_radiation_schleswig_forecast
            - p2.shortwave_radiation_schleswig AS shortwave_radiation_schleswig_error_lag_48h,
        p2.shortwave_radiation_bavaria_forecast
            - p2.shortwave_radiation_bavaria AS shortwave_radiation_bavaria_error_lag_48h,
        p2.shortwave_radiation_bawue_forecast
            - p2.shortwave_radiation_bawue AS shortwave_radiation_bawue_error_lag_48h,

        p2.cloud_cover_brandenburg_forecast
            - p2.cloud_cover_brandenburg AS cloud_cover_brandenburg_error_lag_48h,
        p2.cloud_cover_schleswig_forecast
            - p2.cloud_cover_schleswig AS cloud_cover_schleswig_error_lag_48h,
        p2.cloud_cover_bavaria_forecast
            - p2.cloud_cover_bavaria AS cloud_cover_bavaria_error_lag_48h,
        p2.cloud_cover_bawue_forecast
            - p2.cloud_cover_bawue AS cloud_cover_bawue_error_lag_48h,

        p2.temperature_2m_brandenburg_forecast
            - p2.temperature_2m_brandenburg AS temperature_2m_brandenburg_error_lag_48h,
        p2.temperature_2m_schleswig_forecast
            - p2.temperature_2m_schleswig AS temperature_2m_schleswig_error_lag_48h,
        p2.temperature_2m_bavaria_forecast
            - p2.temperature_2m_bavaria AS temperature_2m_bavaria_error_lag_48h,
        p2.temperature_2m_bawue_forecast
            - p2.temperature_2m_bawue AS temperature_2m_bawue_error_lag_48h,

        p2.total_consumption_forecast_mw
            - p2.total_consumption_mw AS total_consumption_error_mw_lag_48h
    FROM hourly AS target
    LEFT JOIN hourly AS p1
        ON p1.timestamp = target.timestamp - INTERVAL '24 hours'
    LEFT JOIN hourly AS p2
        ON p2.timestamp = target.timestamp - INTERVAL '48 hours'
    LEFT JOIN hourly AS p3
        ON p3.timestamp = target.timestamp - INTERVAL '72 hours'
    LEFT JOIN hourly AS p4
        ON p4.timestamp = target.timestamp - INTERVAL '96 hours'
    LEFT JOIN hourly AS p5
        ON p5.timestamp = target.timestamp - INTERVAL '120 hours'
    LEFT JOIN hourly AS p6
        ON p6.timestamp = target.timestamp - INTERVAL '144 hours'
    LEFT JOIN hourly AS p7
        ON p7.timestamp = target.timestamp - INTERVAL '168 hours'
)
SELECT
    timestamp,
    resolution,
    delivery_date,
    local_timestamp,
    horizon,
    price_eur_mwh,

    year,
    quarter,
    month,
    week_of_year,
    day_of_week,
    is_weekend,
    is_german_public_holiday,
    is_workday,

    price_lag_24h,
    price_lag_48h,
    price_lag_168h,
    price_rolling_avg_7d_same_hour,

    austria_price_lag_24h,
    france_price_lag_24h,
    netherlands_price_lag_24h,
    poland_price_lag_24h,
    switzerland_price_lag_24h,
    czechia_price_lag_24h,
    denmark_1_price_lag_24h,
    denmark_2_price_lag_24h,

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

    residual_load_lag_48h,
    total_consumption_lag_48h,
    wind_onshore_lag_48h,
    wind_offshore_lag_48h,
    solar_lag_48h,

    total_consumption_forecast_mw,
    total_consumption_error_mw_lag_48h,
    -- wind_onshore_forecast_mw_lag_24h,
    wind_offshore_forecast_mw_lag_24h,
    solar_forecast_mw_lag_24h,
    -- residual_load_forecast_mw_lag_24h,

    wind_speed_100m_brandenburg_error_lag_48h,
    wind_speed_100m_schleswig_error_lag_48h,
    wind_speed_100m_bavaria_error_lag_48h,
    wind_speed_100m_bawue_error_lag_48h,
    wind_direction_100m_brandenburg_error_sin_lag_48h,
    wind_direction_100m_brandenburg_error_cos_lag_48h,
    wind_direction_100m_schleswig_error_sin_lag_48h,
    wind_direction_100m_schleswig_error_cos_lag_48h,
    wind_direction_100m_bavaria_error_sin_lag_48h,
    wind_direction_100m_bavaria_error_cos_lag_48h,
    wind_direction_100m_bawue_error_sin_lag_48h,
    wind_direction_100m_bawue_error_cos_lag_48h,
    shortwave_radiation_brandenburg_error_lag_48h,
    shortwave_radiation_schleswig_error_lag_48h,
    shortwave_radiation_bavaria_error_lag_48h,
    shortwave_radiation_bawue_error_lag_48h,
    cloud_cover_brandenburg_error_lag_48h,
    cloud_cover_schleswig_error_lag_48h,
    cloud_cover_bavaria_error_lag_48h,
    cloud_cover_bawue_error_lag_48h,
    temperature_2m_brandenburg_error_lag_48h,
    temperature_2m_schleswig_error_lag_48h,
    temperature_2m_bavaria_error_lag_48h,
    temperature_2m_bawue_error_lag_48h
FROM lagged