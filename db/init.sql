CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS operations;

CREATE TABLE IF NOT EXISTS operations.pipeline_runs(
    run_id UUID PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    requested_start TIMESTAMP WITH TIME ZONE NOT NULL,
    requested_end TIMESTAMP WITH TIME ZONE NOT NULL,
    git_sha TEXT,
    config_hash TEXT,
    raw_inventory_uri TEXT,
    error_message TEXT,
    CHECK (completed_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE IF NOT EXISTS operations.pipeline_stage_runs(
    stage_run_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES operations.pipeline_runs(run_id),
    stage_name TEXT NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    UNIQUE(run_id, stage_name),
    CHECK (completed_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE IF NOT EXISTS analytics.forecast_results(
    forecast_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES operations.pipeline_runs(run_id),
    model_name TEXT NOT NULL,
    model_version_uri TEXT NOT NULL,
    data_version_id TEXT NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    target_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    quantile NUMERIC,
    predicted_value DOUBLE PRECISION NOT NULL,
    actual_value DOUBLE PRECISION,
    published_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL CHECK (status IN ('pending', 'published', 'superseded', 'invalid')),
    UNIQUE(run_id, model_name, target_timestamp, quantile)
);

CREATE INDEX IF NOT EXISTS forecast_results_published_target_idx
    ON analytics.forecast_results (target_timestamp, model_name)
    WHERE status = 'published';

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