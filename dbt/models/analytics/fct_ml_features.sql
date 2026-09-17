SELECT
    mp.timestamp,
    mp.resolution,

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
    g.nuclear_mw,
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

FROM 
    {{ ref('fct_market_prices') }} AS mp
        LEFT JOIN
    {{ ref('fct_energy_generation') }} AS g
        ON mp.timestamp = g.timestamp
        AND mp.resolution = g.resolution
        LEFT JOIN
    {{ ref('fct_energy_generation_forecast') }} AS gf
        ON mp.timestamp = gf.timestamp
        AND mp.resolution = gf.resolution
        LEFT JOIN
    {{ ref('fct_price_spreads') }} AS ps
        ON mp.timestamp = ps.timestamp
        AND mp.resolution = ps.resolution
        LEFT JOIN
    {{ ref('fct_weather') }} AS w
        ON mp.timestamp = w.timestamp
        LEFT JOIN
    {{ ref('fct_weather_forecast') }} AS wf
        ON mp.timestamp = wf.timestamp
        LEFT JOIN
    {{ ref('dim_date') }} AS d
        ON mp.timestamp :: date = d.date_day
