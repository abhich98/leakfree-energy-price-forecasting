WITH weather_pivot AS (
    SELECT
        timestamp,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_brandenburg,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_brandenburg,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_brandenburg,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'cloud_cover')         AS cloud_cover_brandenburg,
        MAX(value) FILTER (WHERE region = 'wind_region_brandenburg' AND signal_name = 'temperature_2m')      AS temperature_2m_brandenburg,

        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_schleswig,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_schleswig,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_schleswig,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'cloud_cover')         AS cloud_cover_schleswig,
        MAX(value) FILTER (WHERE region = 'wind_region_schleswig' AND signal_name = 'temperature_2m')      AS temperature_2m_schleswig,

        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_bavaria,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_bavaria,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_bavaria,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'cloud_cover')         AS cloud_cover_bavaria,
        MAX(value) FILTER (WHERE region = 'solar_region_bavaria' AND signal_name = 'temperature_2m')      AS temperature_2m_bavaria,

        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'wind_speed_100m')    AS wind_speed_100m_bawue,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'wind_direction_100m') AS wind_direction_100m_bawue,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'shortwave_radiation') AS shortwave_radiation_bawue,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'cloud_cover')         AS cloud_cover_bawue,
        MAX(value) FILTER (WHERE region = 'solar_region_bawue' AND signal_name = 'temperature_2m')      AS temperature_2m_bawue

    FROM {{ ref('stg_weather') }}
    GROUP BY timestamp
)
SELECT *
FROM weather_pivot
ORDER BY timestamp