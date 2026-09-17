WITH base AS (
    SELECT
        np.timestamp,
        np.resolution,
        np.signal_name             AS neighbour,
        np.value                   AS neighbour_price_eur_mwh,
        p.value                    AS de_lu_price_eur_mwh,
        p.value - np.value         AS spread_eur_mwh
    FROM
        {{ ref('stg_smard_neighbour_prices') }} AS np
            JOIN
        {{ ref('stg_smard_prices') }} AS p
            ON np.timestamp = p.timestamp
            AND np.resolution = p.resolution
)

SELECT
    timestamp,
    resolution,
    MAX(de_lu_price_eur_mwh)                                               AS de_lu_price_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'AUSTRIA')      AS austria_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'AUSTRIA')      AS austria_spread_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'FRANCE')       AS france_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'FRANCE')       AS france_spread_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'NETHERLANDS')  AS netherlands_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'NETHERLANDS')  AS netherlands_spread_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'POLAND')       AS poland_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'POLAND')       AS poland_spread_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'SWITZERLAND')  AS switzerland_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'SWITZERLAND')  AS switzerland_spread_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'CZECHIA')      AS czechia_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'CZECHIA')      AS czechia_spread_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'DENMARK_1')    AS denmark_1_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'DENMARK_1')    AS denmark_1_spread_eur_mwh,
    MAX(neighbour_price_eur_mwh) FILTER (WHERE neighbour = 'DENMARK_2')    AS denmark_2_price_eur_mwh,
    MAX(spread_eur_mwh)          FILTER (WHERE neighbour = 'DENMARK_2')    AS denmark_2_spread_eur_mwh
FROM base
GROUP BY timestamp, resolution
ORDER BY timestamp, resolution
