with generation_forecast_pivot as (
  select
    timestamp,
    resolution,
    MAX(value) FILTER(WHERE signal_name = 'WIND_ONSHORE_FC') as wind_onshore_forecast_mw,
    MAX(value) FILTER(WHERE signal_name = 'WIND_OFFSHORE_FC') as wind_offshore_forecast_mw,
    MAX(value) FILTER(WHERE signal_name = 'SOLAR_FC') as solar_forecast_mw,
    MAX(value) FILTER(WHERE signal_name = 'WIND_PV_FC') as wind_pv_forecast_mw,
    MAX(value) FILTER(WHERE signal_name = 'TOTAL_GENERATION_FC') as total_generation_forecast_mw,

    MAX(value) FILTER(WHERE signal_name = 'RESIDUAL_LOAD_FC') as residual_load_forecast_mw,
    MAX(value) FILTER(WHERE signal_name = 'TOTAL_CONSUMPTION_FC') as total_consumption_forecast_mw
  from {{ ref('stg_smard_generation_load_forecast') }}
  group by timestamp, resolution
)

select *
from generation_forecast_pivot
order by timestamp
