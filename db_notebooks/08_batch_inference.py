# Databricks notebook source
# MAGIC %pip install -q pyarrow pandas numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# ==========================================
# 08_batch_inference.py — score JSON LR model via mapInPandas, write to gold_signals
# - Serverless-safe (no local filesystem)
# - Auto-aligns to target table schema (symbol↔pair, prob↔proba)
# ==========================================

# ---------- Settings ----------
CATALOG = "workspace"
SCHEMA  = "mntrading"

# Model artifacts produced by 07_publish_file_based.py
PRODUCTION_DIR = "/Volumes/workspace/mntrading/raw/models/production"
MODEL_FILE     = f"{PRODUCTION_DIR}/model.json"
META_FILE      = f"{PRODUCTION_DIR}/meta.json"

# Feature sources (first existing will be used)
FEATURE_TABLES = [
    f"{CATALOG}.{SCHEMA}.gold_dataset",
    f"{CATALOG}.{SCHEMA}.silver_features_5m",
]

# Target table for signals
SIGNALS_TABLE = f"{CATALOG}.{SCHEMA}.gold_signals"

# Inference options
DAYS_BACK = 14
P_LONG  = 0.55
P_SHORT = 0.45

# ---------- Imports ----------
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, TimestampType, StringType, FloatType, IntegerType
from typing import Iterator
import json, numpy as np, pandas as pd

# ---------- Spark / Unity Catalog init ----------
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE {SCHEMA}")

# ---------- Helpers ----------
def path_exists(p: str) -> bool:
    try:
        dbutils.fs.ls(p)
        return True
    except Exception:
        return False

def read_json(dbfs_path: str) -> dict:
    return json.loads(dbutils.fs.head(dbfs_path, 5 * 1024 * 1024))

def table_exists_uc(qualified: str) -> bool:
    """Check table existence for a qualified name like catalog.schema.table"""
    try:
        cat, sch, tbl = qualified.split(".")
        return len(spark.sql(f"SHOW TABLES IN {cat}.{sch} LIKE '{tbl}'").collect()) > 0
    except Exception:
        return False

def pick_input_table() -> str:
    """Pick the first existing feature table from FEATURE_TABLES."""
    for t in FEATURE_TABLES:
        if table_exists_uc(t):
            return t
    raise RuntimeError(f"No feature table found. Tried: {FEATURE_TABLES}")

def table_exists(qualified: str) -> bool:
    """Same as table_exists_uc (alias for readability below)."""
    return table_exists_uc(qualified)

# ---------- Load production model (JSON-LR) ----------
if not path_exists(PRODUCTION_DIR):
    dbutils.notebook.exit(f"Production dir not found: {PRODUCTION_DIR}")

meta = read_json(META_FILE)
model = read_json(MODEL_FILE)
if model.get("format") != "json_lr":
    dbutils.notebook.exit("Only 'json_lr' model format is supported by this notebook.")

feature_cols = list(meta["features"])
label_col = meta.get("label", "label")

# Coefficients / scaler as numpy arrays
coef = np.asarray(model["coef"], dtype="float32")
intercept = np.float32(model["intercept"])
mean = np.asarray(model["scaler"]["mean"], dtype="float32")
scale = np.asarray(model["scaler"]["scale"], dtype="float32")
scale = np.where(scale == 0.0, 1.0, scale)

print(f"[MODEL] Loaded json_lr: {len(feature_cols)} features, AUC={model.get('metrics',{}).get('auc')}")

# ---------- Read recent features ----------
src_tbl = pick_input_table()
df = spark.table(src_tbl)

# Ensure 'ts' exists and is timestamp
if "ts" not in df.columns and "timestamp" in df.columns:
    df = df.withColumnRenamed("timestamp", "ts")
if "ts" not in df.columns:
    dbutils.notebook.exit(f"Input table {src_tbl} must have a 'ts' column.")

# Ensure we have a 'symbol' column (fallback from 'pair' if needed)
if "symbol" not in df.columns and "pair" in df.columns:
    df = df.withColumn("symbol", F.col("pair"))

df = df.withColumn("ts", F.col("ts").cast("timestamp"))
df = df.filter(F.col("ts") >= F.current_timestamp() - F.expr(f"INTERVAL {DAYS_BACK} DAYS"))

id_cols = [c for c in ["ts", "symbol"] if c in df.columns]
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    dbutils.notebook.exit(f"Missing required feature columns: {missing}")

df_in = df.select(*(id_cols + feature_cols))

# ---------- Score with mapInPandas (iterator -> iterator) ----------
schema = StructType([
    StructField("ts", TimestampType(), False),
    StructField("symbol", StringType(), True),
    StructField("prob", FloatType(), True),
    StructField("signal", IntegerType(), True),
])

def score_batches(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """mapInPandas callback: receive iterator of pandas DFs, yield scored DFs."""
    for pdf in batches:
        # Fill any locally missing feature columns
        for c in feature_cols:
            if c not in pdf.columns:
                pdf[c] = np.nan

        # Strict numpy arrays
        X  = np.asarray(pdf[feature_cols].astype("float32").values, dtype="float32")
        mu = np.asarray(mean, dtype="float32")
        sc = np.asarray(scale, dtype="float32")
        w  = np.asarray(coef, dtype="float32")
        b  = np.float32(intercept)

        # Standardize + logistic
        z = (X - mu) / sc
        logits = z.dot(w) + b
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.asarray(p, dtype="float32")

        sig = np.where(p >= P_LONG, 1, np.where(p <= P_SHORT, -1, 0)).astype("int32")

        ts_vals  = pdf["ts"].values
        sym_vals = pdf["symbol"].values if "symbol" in pdf.columns else np.array([None] * len(pdf), dtype=object)

        out = pd.DataFrame({
            "ts": ts_vals,
            "symbol": sym_vals,
            "prob": p,
            "signal": sig,
        })
        yield out

scored = df_in.mapInPandas(score_batches, schema=schema)
display(scored.orderBy(F.col("ts").desc()).limit(10))

# ---------- Create/align target table & MERGE ----------
TARGET = SIGNALS_TABLE  # e.g. workspace.mntrading.gold_signals

if not table_exists(TARGET):
    # Default schema if absent (symbol/prob)
    spark.sql(f"""
    CREATE TABLE {TARGET} (
      ts TIMESTAMP,
      symbol STRING,
      prob FLOAT,
      signal INT
    ) USING DELTA
    """)
    target_cols = ["ts", "symbol", "prob", "signal"]
else:
    target_cols = spark.table(TARGET).columns

# Determine id / prob column names in target
id_tgt   = "symbol" if "symbol" in target_cols else ("pair" if "pair" in target_cols else "symbol")
prob_tgt = "prob"   if "prob"   in target_cols else ("proba" if "proba" in target_cols else "prob")

# Align column names from 'scored' to target schema
select_expr = ["ts"]
if id_tgt == "pair":
    select_expr.append("symbol as pair")
else:
    select_expr.append("symbol as symbol")

if prob_tgt == "proba":
    select_expr.append("prob as proba")
else:
    select_expr.append("prob as prob")

select_expr.append("signal")

scored_aligned = scored.selectExpr(*select_expr)
scored_aligned.createOrReplaceTempView("scored_batch_aligned")

# MERGE using resolved column names
merge_sql = f"""
MERGE INTO {TARGET} AS tgt
USING scored_batch_aligned AS src
ON tgt.{id_tgt} = src.{id_tgt} AND tgt.ts = src.ts
WHEN MATCHED THEN UPDATE SET
  tgt.{prob_tgt} = src.{prob_tgt},
  tgt.signal     = src.signal
WHEN NOT MATCHED THEN INSERT (ts, {id_tgt}, {prob_tgt}, signal)
VALUES (src.ts, src.{id_tgt}, src.{prob_tgt}, src.signal)
"""
spark.sql(merge_sql)

rows_written = spark.sql("SELECT COUNT(*) AS c FROM scored_batch_aligned").first()["c"]
total_rows   = spark.sql(f"SELECT COUNT(*) AS c FROM {TARGET}").first()["c"]
print(f"[OK] Wrote {rows_written} rows into {TARGET}. Total now: {total_rows}")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   MIN(ts) AS ts_min, 
# MAGIC   MAX(ts) AS ts_max, 
# MAGIC   COUNT(*) AS rows
# MAGIC FROM workspace.mntrading.gold_signals;