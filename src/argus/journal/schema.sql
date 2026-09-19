-- ARGUS Stage 6: trade journal
--
-- Built early and deliberately, so real paper-trade data starts accumulating before
-- Stages 2-5 exist. That ordering matters more now than when it was planned: retrospective
-- evaluation keeps running into data-quality limits, so forward-recorded outcomes are the
-- most trustworthy evidence this project will have.
--
-- Portable across PostgreSQL and SQLite (types chosen to work in both).

CREATE TABLE IF NOT EXISTS trades (
    id                    INTEGER PRIMARY KEY,

    -- identity
    ticker                TEXT    NOT NULL,
    sub_segment           TEXT    NOT NULL,

    -- entry
    entry_date            DATE    NOT NULL,
    entry_price           REAL    NOT NULL CHECK (entry_price > 0),
    direction             TEXT    NOT NULL DEFAULT 'long',

    -- what the model saw and said at entry
    quant_score           REAL,
    quant_percentile      REAL,
    thesis                TEXT    NOT NULL,
    scenario_chosen       TEXT    NOT NULL,
    scenario_probability  REAL,
    conviction            REAL,

    -- plan
    take_profit           REAL    CHECK (take_profit IS NULL OR take_profit > 0),
    invalidation          REAL    CHECK (invalidation IS NULL OR invalidation > 0),
    target_holding_days   INTEGER CHECK (target_holding_days IS NULL OR target_holding_days > 0),

    -- outcome (NULL while open)
    exit_date             DATE,
    exit_price            REAL    CHECK (exit_price IS NULL OR exit_price > 0),
    exit_reason           TEXT,   -- target | invalidation | time | discretionary
    realized_return       REAL,
    benchmark_return      REAL,   -- equal-weight sector over the same window
    realized_rel_return   REAL,   -- the number that actually matters

    -- PROVENANCE
    -- Without these columns a winning trade cannot be attributed to the training run
    -- rather than the prompt -- which is the entire point of the Phase 3 gate. Cheap to
    -- add now, impossible to backfill.
    arm                   TEXT    NOT NULL,
    base_model            TEXT,
    adapter_sha256        TEXT,
    corpus_manifest_sha256 TEXT,
    corpus_cutoff         DATE,
    config_sha256         TEXT,
    code_git_sha          TEXT,
    quant_model_version   TEXT,

    -- bookkeeping
    is_paper              INTEGER NOT NULL DEFAULT 1,
    notes                 TEXT,
    created_at            TIMESTAMP NOT NULL,
    updated_at            TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker      ON trades (ticker);
CREATE INDEX IF NOT EXISTS idx_trades_entry_date  ON trades (entry_date);
CREATE INDEX IF NOT EXISTS idx_trades_arm         ON trades (arm);
CREATE INDEX IF NOT EXISTS idx_trades_open        ON trades (exit_date);
CREATE INDEX IF NOT EXISTS idx_trades_sub_segment ON trades (sub_segment);


-- Full recommendation payloads, including the ones never traded.
--
-- Recording non-trades is not bookkeeping pedantry: a model that says "no trade" on every
-- bad setup and is only measured on the trades it took looks far better than it is.
-- Calibration requires the denominator.
CREATE TABLE IF NOT EXISTS recommendations (
    id                    INTEGER PRIMARY KEY,
    ticker                TEXT    NOT NULL,
    sub_segment           TEXT    NOT NULL,
    as_of                 DATE    NOT NULL,
    direction             TEXT    NOT NULL,
    conviction            REAL,
    payload_json          TEXT    NOT NULL,   -- serialised Recommendation
    briefing_json         TEXT,               -- serialised Briefing, when tools ran
    sourced_fraction      REAL,               -- Phase 4 audit metric
    arm                   TEXT    NOT NULL,
    provenance_key        TEXT    NOT NULL,
    trade_id              INTEGER REFERENCES trades (id),
    created_at            TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recs_as_of ON recommendations (as_of);
CREATE INDEX IF NOT EXISTS idx_recs_arm   ON recommendations (arm);
CREATE INDEX IF NOT EXISTS idx_recs_prov  ON recommendations (provenance_key);
