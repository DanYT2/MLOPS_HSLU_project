-- Schema for the churn model monitoring pipeline.
-- Loaded automatically by Postgres on first boot because docker-compose mounts
-- this file at /docker-entrypoint-initdb.d/init.sql.
--
-- One row per simulated batch. Columns without a ground-truth label (i.e. the
-- test.csv simulation pool) leave the classification metrics NULL.

CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id                         SERIAL PRIMARY KEY,
    ts                         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    data_source                TEXT             NOT NULL,
    batch_id                   INT              NOT NULL,
    batch_size                 INT,
    num_drifted_columns        INT,
    share_drifted_columns      DOUBLE PRECISION,
    prediction_drift           DOUBLE PRECISION,
    share_missing_values       DOUBLE PRECISION,
    mean_predicted_churn_prob  DOUBLE PRECISION,
    churn_rate                 DOUBLE PRECISION,
    accuracy                   DOUBLE PRECISION,
    roc_auc                    DOUBLE PRECISION,
    log_loss                   DOUBLE PRECISION,
    -- Records the magnitude (0.0 = none, 1.0 = full strength) of the drift the
    -- monitor artificially injected into this batch. Useful in Grafana to line
    -- up the *cause* (synthetic drift) with the *effect* (Evidently scores).
    drift_intensity            DOUBLE PRECISION NOT NULL DEFAULT 0
);

-- Idempotent migration for pre-existing pgdata volumes that were created before
-- ``drift_intensity`` existed. The CREATE TABLE above already includes the
-- column on first init, so this is a no-op there.
ALTER TABLE monitoring_metrics
    ADD COLUMN IF NOT EXISTS drift_intensity DOUBLE PRECISION NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS monitoring_metrics_ts_idx
    ON monitoring_metrics (ts);

CREATE INDEX IF NOT EXISTS monitoring_metrics_source_idx
    ON monitoring_metrics (data_source, ts);
