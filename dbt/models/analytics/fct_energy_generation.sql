with generation_pivot as (
  select
    timestamp,
    resolution,
    MAX(value) FILTER(WHERE signal_name = 'WIND_ONSHORE') as wind_onshore_mw,
    MAX(value) FILTER(WHERE signal_name = 'WIND_OFFSHORE') as wind_offshore_mw,
    MAX(value) FILTER(WHERE signal_name = 'SOLAR') as solar_mw,
    MAX(value) FILTER(WHERE signal_name = 'BIOMASS') as biomass_mw,
    MAX(value) FILTER(WHERE signal_name = 'HYDROPOWER') as hydropower_mw,
    MAX(value) FILTER(WHERE signal_name = 'PUMPED_STORAGE') as pumped_storage_mw,
    MAX(value) FILTER(WHERE signal_name = 'NATURAL_GAS') as natural_gas_mw,
    MAX(value) FILTER(WHERE signal_name = 'HARD_COAL') as hard_coal_mw,
    MAX(value) FILTER(WHERE signal_name = 'BROWN_COAL') as brown_coal_mw,
    MAX(value) FILTER(WHERE signal_name = 'NUCLEAR') as nuclear_mw,
    MAX(value) FILTER(WHERE signal_name = 'OTHER_CONVENTIONAL') as other_conventional_mw,
    MAX(value) FILTER(WHERE signal_name = 'OTHER_RENEWABLE') as other_renewable_mw,
    MAX(value) FILTER(WHERE signal_name = 'TOTAL_CONSUMPTION') as total_consumption_mw,
    MAX(value) FILTER(WHERE signal_name = 'RESIDUAL_LOAD') as residual_load_mw,
    SUM(value) FILTER(WHERE signal_name IN (
      'WIND_ONSHORE',
      'WIND_OFFSHORE',
      'SOLAR',
      'BIOMASS',
      'HYDROPOWER',
      'PUMPED_STORAGE',
      'NATURAL_GAS',
      'HARD_COAL',
      'BROWN_COAL',
      'NUCLEAR',
      'OTHER_CONVENTIONAL',
      'OTHER_RENEWABLE'
    )) as total_generation_mw
  from {{ ref('stg_smard_generation') }}
  group by timestamp, resolution
)

select *
from generation_pivot
order by timestamp
