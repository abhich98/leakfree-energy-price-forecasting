WITH requested_window AS (
    SELECT
        '{{ var("feature_start") }}'::timestamptz AS start_timestamp,
        '{{ var("feature_end_exclusive") }}'::timestamptz AS end_timestamp
),
requested_timestamp_spine AS (
    SELECT
        generate_series(
            start_timestamp,
            end_timestamp - INTERVAL '1 hour',
            INTERVAL '1 hour'
        ) AS timestamp,
        'hour'::text AS resolution
    FROM requested_window

    UNION ALL

    SELECT
        generate_series(
            start_timestamp,
            end_timestamp - INTERVAL '15 minutes',
            INTERVAL '15 minutes'
        ) AS timestamp,
        'quarterhour'::text AS resolution
    FROM requested_window
),
timestamp_spine AS (
    SELECT timestamp, resolution
    FROM {{ ref('fct_market_prices') }}

    UNION

    SELECT timestamp, resolution
    FROM requested_timestamp_spine
)
SELECT
    spine.timestamp,
    spine.resolution,

    -- Energy generation features
    g.wind_onshore_mw,
    g.wind_offshore_mw,
    g.solar_mw,
    g.biomass_mw,
    g.hydropower_mw,
    g.pumped_storage_mw,
    g.natural_gas_mw,
    g.hard_coal_mw,
    g.brown_coal_mw,
    g.other_conventional_mw,
    g.other_renewable_mw,

    g.residual_load_mw,
    g.total_consumption_mw,
    g.total_generation_mw,

    -- Energy generation forecast features
    gf.wind_onshore_forecast_mw,
    gf.wind_offshore_forecast_mw,
    gf.solar_forecast_mw,
    gf.wind_pv_forecast_mw,
    gf.total_generation_forecast_mw,

    gf.residual_load_forecast_mw,
    gf.total_consumption_forecast_mw,

    -- Neighboring market prices and spreads
    ps.de_lu_price_eur_mwh,
    ps.austria_price_eur_mwh,
    ps.austria_spread_eur_mwh,
    ps.france_price_eur_mwh,
    ps.france_spread_eur_mwh,
    ps.netherlands_price_eur_mwh,
    ps.netherlands_spread_eur_mwh,
    ps.poland_price_eur_mwh,
    ps.poland_spread_eur_mwh,
    ps.switzerland_price_eur_mwh,
    ps.switzerland_spread_eur_mwh,
    ps.czechia_price_eur_mwh,
    ps.czechia_spread_eur_mwh,
    ps.denmark_1_price_eur_mwh,
    ps.denmark_1_spread_eur_mwh,
    ps.denmark_2_price_eur_mwh,
    ps.denmark_2_spread_eur_mwh,

    -- Market energy price
    mp.price_eur_mwh,

    -- Weather features
    w.wind_speed_100m_brandenburg,
    w.wind_direction_100m_brandenburg,
    w.shortwave_radiation_brandenburg,
    w.cloud_cover_brandenburg,
    w.temperature_2m_brandenburg,
    w.wind_speed_100m_schleswig,
    w.wind_direction_100m_schleswig,
    w.shortwave_radiation_schleswig,
    w.cloud_cover_schleswig,
    w.temperature_2m_schleswig,
    w.wind_speed_100m_bavaria,
    w.wind_direction_100m_bavaria,
    w.shortwave_radiation_bavaria,
    w.cloud_cover_bavaria,
    w.temperature_2m_bavaria,
    w.wind_speed_100m_bawue,
    w.wind_direction_100m_bawue,
    w.shortwave_radiation_bawue,
    w.cloud_cover_bawue,
    w.temperature_2m_bawue,

    -- Weather forecast features
    wf.wind_speed_100m_brandenburg_forecast,
    wf.wind_direction_100m_brandenburg_forecast,
    wf.shortwave_radiation_brandenburg_forecast,
    wf.cloud_cover_brandenburg_forecast,
    wf.temperature_2m_brandenburg_forecast,
    wf.wind_speed_100m_schleswig_forecast,
    wf.wind_direction_100m_schleswig_forecast,
    wf.shortwave_radiation_schleswig_forecast,
    wf.cloud_cover_schleswig_forecast,
    wf.temperature_2m_schleswig_forecast,
    wf.wind_speed_100m_bavaria_forecast,
    wf.wind_direction_100m_bavaria_forecast,
    wf.shortwave_radiation_bavaria_forecast,
    wf.cloud_cover_bavaria_forecast,
    wf.temperature_2m_bavaria_forecast,
    wf.wind_speed_100m_bawue_forecast,
    wf.wind_direction_100m_bawue_forecast,
    wf.shortwave_radiation_bawue_forecast,
    wf.cloud_cover_bawue_forecast,
    wf.temperature_2m_bawue_forecast,

    -- Date features
    d.date_day,
    d.year,
    d.quarter,
    d.month,
    d.week_of_year,
    d.day_of_week,
    d.is_weekend,
    d.season,
    d.is_german_public_holiday,
    d.holiday_name,
    d.is_workday

FROM timestamp_spine AS spine
    LEFT JOIN {{ ref('fct_market_prices') }} AS mp
        ON spine.timestamp = mp.timestamp
        AND spine.resolution = mp.resolution
    LEFT JOIN
    {{ ref('fct_energy_generation_load') }} AS g
        ON spine.timestamp = g.timestamp
        AND spine.resolution = g.resolution
    LEFT JOIN
    {{ ref('fct_energy_generation_load_forecast') }} AS gf
        ON spine.timestamp = gf.timestamp
        AND spine.resolution = gf.resolution
    LEFT JOIN
    {{ ref('fct_price_spreads') }} AS ps
        ON spine.timestamp = ps.timestamp
        AND spine.resolution = ps.resolution
    LEFT JOIN
    {{ ref('fct_weather') }} AS w
        ON spine.timestamp = w.timestamp
    LEFT JOIN
    {{ ref('fct_weather_forecast') }} AS wf
        ON spine.timestamp = wf.timestamp
    LEFT JOIN
    {{ ref('dim_date') }} AS d
        ON spine.timestamp::date = d.date_day
