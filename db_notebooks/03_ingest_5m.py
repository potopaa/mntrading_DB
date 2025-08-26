# Databricks notebook source
# MAGIC %pip install -q pyarrow pandas numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# --- Settings (Unity Catalog) ---
CATALOG = "workspace"
SCHEMA  = "mntrading"
INPUT_PATH_5M = "/Volumes/workspace/mntrading/raw/ohlcv_5m.parquet"

# --- Spark session ---
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE {SCHEMA}")

PAIRS = f"{CATALOG}.{SCHEMA}.silver_pairs_screened"
TABLE = f"{CATALOG}.{SCHEMA}.silver_ohlcv_5m"

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  symbol STRING,
  ts TIMESTAMP,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
  date DATE
) USING DELTA
""")

# --- Robust parquet load via pandas/pyarrow ---
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_integer_dtype,
    is_float_dtype,
)

def normalize_ts(series: pd.Series) -> pd.Series:
    s = series.copy()
    if is_datetime64_any_dtype(s):
        if is_datetime64tz_dtype(s):
            return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        return pd.to_datetime(s)
    if is_integer_dtype(s) or is_float_dtype(s):
        sample = s.dropna()
        if sample.empty:
            return pd.to_datetime(s, errors="coerce")
        v = int(sample.iloc[0])
        unit = "ns" if v > 1_000_000_000_000_000_000 else ("ms" if v > 1_000_000_000_000 else "s")
        out = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        return out.dt.tz_convert("UTC").dt.tz_localize(None)
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out.dt.tz_convert("UTC").dt.tz_localize(None)

pdf = pd.read_parquet(INPUT_PATH_5M)
pdf.columns = [c.lower() for c in pdf.columns]

if "ts" in pdf.columns:
    pdf["ts"] = normalize_ts(pdf["ts"])
elif "timestamp" in pdf.columns:
    pdf["ts"] = normalize_ts(pdf["timestamp"])
else:
    raise ValueError("Parquet must contain 'ts' or 'timestamp'.")

req = ["symbol","open","high","low","close","volume","ts"]
missing = [c for c in req if c not in pdf.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

pdf["symbol"] = pdf["symbol"].astype(str).str.replace("/", "", regex=False)
pdf = (pdf[["symbol","ts","open","high","low","close","volume"]]
       .dropna()
       .sort_values("ts"))

# Optional: filter by screened pairs if present
have_pairs = False
try:
    pairs_pdf = spark.table(PAIRS).select("sym_a","sym_b").distinct().toPandas()
    if len(pairs_pdf) > 0:
        have_pairs = True
        want = sorted(set(pairs_pdf["sym_a"].str.replace("/","")) |
                      set(pairs_pdf["sym_b"].str.replace("/","")))
        pdf = pdf[pdf["symbol"].isin(want)]
except Exception:
    pass

sdf = spark.createDataFrame(pdf).withColumn("date", F.to_date("ts"))

sdf.createOrReplaceTempView("staging_5m")
spark.sql(f"""
MERGE INTO {TABLE} AS t
USING staging_5m AS s
ON  t.symbol = s.symbol AND t.ts = s.ts
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

print(f"[OK] Ingested 5m into {TABLE}. Count=", spark.table(TABLE).count(),
      " filtered_by_pairs=", have_pairs)


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS n5m FROM silver_ohlcv_5m;
# MAGIC SELECT symbol, MIN(ts), MAX(ts), COUNT(*) AS n FROM silver_ohlcv_5m GROUP BY symbol ORDER BY n DESC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS n5m FROM silver_ohlcv_5m;
# MAGIC SELECT symbol, MIN(ts), MAX(ts), COUNT(*) AS n FROM silver_ohlcv_5m GROUP BY symbol ORDER BY n DESC LIMIT 5;
# MAGIC