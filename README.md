# mntrading — Local + Databricks (Serverless + UC Volumes)

> End-to-end pipeline to ingest market data, screen cointegrated pairs, build features and datasets, train and publish a model, run batch inference, and produce a portfolio report. 

---

## Quick start

1. Set up local environment (Python or Docker).
2. Fill `.env` with Databricks access and job IDs.
3. (First time) Create/Update Jobs A and B via REST helper (or manually in UI).
4. Run the orchestrator locally (single run or loop). It will:
   - collect 1h OHLCV for the universe, upload to UC Volumes;
   - trigger **Job A** (1h ingest + pair screening);
   - download the screened pairs JSON;
   - collect **5m** OHLCV only for the screened symbols, upload to Volumes;
   - trigger **Job B** (5m ingest → features → dataset → train → publish → inference → report).

---

## Architecture (high level)

```
[Local / Docker]
  fetchers.py (CCXT) ──1h OHLCV──► stage/ohlcv_1h.parquet 
                                           │               
                                           ▼               
                          UC Volumes: dbfs:/Volumes/.../raw/ohlcv_1h.parquet
                                           │
                                           ▼
                          [Databricks Job A: 01 → 02]
  01_ingest_1h.py:   raw 1h → bronze_ohlcv_1h (Delta)
  02_screen_pairs.py: screen pairs → silver_pairs_screened (Delta)
                      + raw/pairs_screened.json (Volumes)
                                           │
                                           ▼
[Local]
  stage/pairs_screened.json ◄── download from Volumes
  fetchers.py (CCXT) ──5m OHLCV for screened symbols──► stage/ohlcv_5m.parquet
                                           │
                                           ▼
                          UC Volumes: dbfs:/Volumes/.../raw/ohlcv_5m.parquet
                                           │
                                           ▼
                         [Databricks Job B: 03 → 09]
  03_ingest_5m.py:   raw 5m → bronze_ohlcv_5m (Delta)
  04_build_features.py: gold_features_5m (Delta) + VIEW silver_features_5m (compat)
  05_build_dataset.py: gold_dataset (Delta)
  06_train_models.py: candidates (Volumes)
  07_evaluate_register.py: production model (Volumes)
  08_batch_inference.py: gold_signals (Delta)
  09_portfolio_report.py: portfolio PnL report (pairs next-bar spread diff)
```

---

## What runs where

### Local machine (or Docker)
- Fetch 1h OHLCV for the **full universe**; save to `stage/ohlcv_1h.parquet`.
- Upload `stage/ohlcv_1h.parquet` to **UC Volumes**: `dbfs:/Volumes/workspace/mntrading/raw/ohlcv_1h.parquet`.
- Trigger **Job A** and wait for completion (Databricks Jobs API).
- Download `raw/pairs_screened.json` to `stage/`, parse symbols.
- Fetch **5m** OHLCV **only** for the screened symbols; save to `stage/ohlcv_5m.parquet`.
- Upload to `dbfs:/Volumes/workspace/mntrading/raw/ohlcv_5m.parquet`.
- Trigger **Job B** and wait for completion.

### Databricks (Serverless + Unity Catalog)
- Spark transformations & Delta tables across **Bronze/Silver/Gold** layers.
- Model training, candidate selection, production publishing (file-based in Volumes).
- Batch inference to produce signals.
- Portfolio report with **correct pair PnL methodology**.

---

## Data layers and artifacts

### UC Volumes (files)
- `dbfs:/Volumes/workspace/mntrading/raw/ohlcv_1h.parquet` — raw 1h batch (uploaded by the orchestrator).
- `dbfs:/Volumes/workspace/mntrading/raw/ohlcv_5m.parquet` — raw 5m batch.
- `dbfs:/Volumes/workspace/mntrading/raw/pairs_screened.json` — screened pairs (Job A output → Local input).
- `dbfs:/Volumes/workspace/mntrading/raw/models/candidates/...` — model candidates and metadata.
- `dbfs:/Volumes/workspace/mntrading/raw/models/production/...` — production model snapshot.

### Unity Catalog (Delta tables)
- `workspace.mntrading.bronze_ohlcv_1h` — 1h bronze.
- `workspace.mntrading.bronze_ohlcv_5m` — 5m bronze.
- `workspace.mntrading.silver_pairs_screened` — pair screening results.
- `workspace.mntrading.gold_features_5m` — features.
- `workspace.mntrading.gold_dataset` — training dataset.
- `workspace.mntrading.gold_signals` — batch inference signals.

### Views
- `workspace.mntrading.silver_features_5m` — a **VIEW** exposing `(ts, symbol, close)` over the 5m source for compatibility.
- `prices_5m` — a **TEMP VIEW** created by `09_portfolio_report.py` for interactive SQL cells.

---

## Databricks Jobs

### Job A — “1h ingest + screening”
1. `01_ingest_1h.py`
   - Reads Volumes `raw/ohlcv_1h.parquet`.
   - Normalizes schema, cleans, writes Delta table `bronze_ohlcv_1h`.
2. `02_screen_pairs.py`
   - Computes correlations/cointegration (Engle–Granger and filters) on 1h.
   - Writes: `silver_pairs_screened` (Delta) and `raw/pairs_screened.json` (Volumes) for the local orchestrator.

### Job B — “5m ingest → features → dataset → train → publish → inference → report”
3. `03_ingest_5m.py`
   - Reads Volumes `raw/ohlcv_5m.parquet`, writes `bronze_ohlcv_5m`.
4. `04_build_features.py`
   - Builds 5m features. Writes `gold_features_5m`.
   - Creates compatibility VIEW `silver_features_5m` with `(ts, symbol, close)` so legacy SQL keeps working.
5. `05_build_dataset.py`
   - Builds `gold_dataset` (features + labels, e.g., next-bar sign or threshold-based reversion).
6. `06_train_models.py`
   - Trains (e.g., Logistic Regression) and stores **candidates** in Volumes; writes `candidates.json`.
7. `07_evaluate_register.py`
   - Picks the best candidate and **publishes** to `raw/models/production/` in Volumes.
8. `08_batch_inference.py`
   - Loads production model, runs batch inference, writes `gold_signals` (Delta) with schema guards.
9. `09_portfolio_report.py`
   - Computes portfolio PnL.
   - **Pairs**: uses **spread increment** on the **next bar**: `Δs = Δln(A) − β·Δln(B)`; strict inner-join on bar times; sanity filters for bad bars (`|Δln| > 50%`); optional normalization by gross exposure; PnL = `signal * Δs`.
   - **Single symbols**: standard next-bar simple return × signal.
   - Outputs daily aggregates, per-pair tables; creates TEMP VIEW `prices_5m`.

---

## Local orchestrator

**File**: `orchestrator.py`

Responsibilities:
- Collect 1h universe via CCXT and save to `stage/ohlcv_1h.parquet`.
- Upload to UC Volumes `raw/ohlcv_1h.parquet`.
- Trigger **Job A** and wait.
- Download `raw/pairs_screened.json` → `stage/`, parse symbols.
- Collect 5m only for selected symbols → `stage/ohlcv_5m.parquet`.
- Upload to UC Volumes `raw/ohlcv_5m.parquet`.
- Trigger **Job B** and wait.

Environment variables from `.env`:
- `DATABRICKS_HOST`, `DATABRICKS_TOKEN`
- `DB_CATALOG=workspace`, `DB_SCHEMA=mntrading`
- `DB_JOB_A=<id>`, `DB_JOB_B=<id>`
- `INTERVAL_MIN` for loop mode (if using `orchestrator_loop`).

---

## Creating/Updating Jobs (REST helper)

1) Fill `.env` with a valid PAT and workspace info.  
2) Review JSON manifests (e.g., `jobA_serverless.json`, `jobB_serverless.json`).  
3) Run:

```bash
python create_jobs_rest.py
```

The script will create/update Jobs A/B and print their IDs. Alternatively, create jobs in the Databricks UI and copy their IDs into `.env`.

---

## Running the pipeline

### One-off run

```bash
# Local (no Docker)
pip install -r requirements.txt
python orchestrator.py

# With Docker
docker compose run --rm orchestrator
```

### Periodic loop

```bash
docker compose up -d orchestrator_loop
# Stop later:
docker compose down
```

---

## Diagnostics and quality checks

- After Job A, verify `silver_pairs_screened` is non-empty and `pairs_screened.json` is well-formed.
- In Job B, `09_portfolio_report.py` uses **spread increments** and **next-bar** logic; this removes unrealistic “astronomical” returns.
- Report includes sanity filters (`|Δln| > 50%` for per-leg 5m jumps) and strict time alignment (inner join on bars).
- If `silver_features_5m` is missing, the notebook creates a **VIEW** for compatibility.

## Model and inference

- Baseline model: **Logistic Regression** (sklearn). Candidates and production snapshots are stored **file-based** in UC Volumes.
- Batch inference writes to `gold_signals` (Delta) with schema normalization.
- Upgrade path: switch the publish step to **MLflow Model Registry** (`07`), use stage aliases instead of file paths.

---


