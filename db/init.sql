CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

-- SMARD raw tables: resolution column added so hourly and quarter-hourly rows
-- for the same timestamp coexist (hourly ts aligns with every 4th 15-min ts).
-- The unique constraint includes resolution to prevent clobbering on re-ingest.
CREATE TABLE IF NOT EXISTS raw.smard_generation(
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    resolution TEXT NOT NULL DEFAULT 'hour',
    UNIQUE(timestamp, signal, resolution)
);

CREATE TABLE IF NOT EXISTS raw.smard_prices(
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    resolution TEXT NOT NULL DEFAULT 'hour',
    UNIQUE(timestamp, signal, resolution)
);

CREATE TABLE IF NOT EXISTS raw.smard_neighbour_prices(
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    resolution TEXT NOT NULL DEFAULT 'hour',
    UNIQUE(timestamp, signal, resolution)
);

-- SMARD forecasted signals (filters 122, 123, 125, 3791, ...).
CREATE TABLE IF NOT EXISTS raw.smard_forecast(
    issue_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    resolution TEXT NOT NULL DEFAULT 'hour',
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE(timestamp, signal, resolution)
);

CREATE TABLE IF NOT EXISTS raw.weather(
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    region TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    UNIQUE(timestamp, region, signal_type)
);

-- Historical and current weather forecasts for ML training (leak-safe: uses the forecast
-- that was actually available at auction time, not ERA5 actuals).
-- issue_timestamp = the UTC timestamp when the forecast was issued.
-- issue_time = legacy alias for issue timestamp retained for compatibility.
-- timestamp = the hour the forecast predicts.
-- model = the weather model (icon_seamless for stitched, ecmwf_ifs for single runs).
CREATE TABLE IF NOT EXISTS raw.weather_forecast(
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    issue_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    region TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    model TEXT NOT NULL,
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE(timestamp, region, signal_type)
);