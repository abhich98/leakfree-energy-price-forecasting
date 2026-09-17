WITH weather_forecast_pivot AS (
    SELECT
        timestamp,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_brandenburg_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_brandenburg_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_brandenburg_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'cloud_cover')         AS cloud_cover_brandenburg_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'temperature_2m')      AS temperature_2m_brandenburg_forecast,

        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_schleswig_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_schleswig_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_schleswig_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'cloud_cover')         AS cloud_cover_schleswig_forecast,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'temperature_2m')      AS temperature_2m_schleswig_forecast,

        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_bavaria_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_bavaria_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_bavaria_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'cloud_cover')         AS cloud_cover_bavaria_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'temperature_2m')      AS temperature_2m_bavaria_forecast,

        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_bawue_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_bawue_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_bawue_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'cloud_cover')         AS cloud_cover_bawue_forecast,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'temperature_2m')      AS temperature_2m_bawue_forecast

    FROM {{ ref('stg_weather_forecast') }}
    GROUP BY timestamp
)
SELECT *
FROM weather_forecast_pivot
ORDER BY timestamp