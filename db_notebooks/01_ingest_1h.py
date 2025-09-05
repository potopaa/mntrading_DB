# Databricks notebook source
# MAGIC %pip install -q pyarrow pandas numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
CATALOG = "workspace"
SCHEMA  = "mntrading"
INPUT_PATH_1H = "/Volumes/workspace/mntrading/raw/ohlcv_1h.parquet"

from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE {SCHEMA}")

TABLE = f"{CATALOG}.{SCHEMA}.bronze_ohlcv_1h"

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
            s = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            s = pd.to_datetime(s, errors="coerce")
        return s

    if is_integer_dtype(s) or is_float_dtype(s):
        sample = s.dropna()
        if sample.empty:
            return pd.to_datetime(s, errors="coerce")
        v = float(sample.iloc[0])
        if v > 1_000_000_000_000_000_000:   # >1e18 -> nanoseconds
            unit = "ns"
        elif v > 1_000_000_000_000:         # >1e12 -> milliseconds
            unit = "ms"
        else:
            unit = "s"
        out = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        return out.dt.tz_convert("UTC").dt.tz_localize(None)

    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out.dt.tz_convert("UTC").dt.tz_localize(None)

pdf = pd.read_parquet(INPUT_PATH_1H)
pdf.columns = [c.lower() for c in pdf.columns]

if "ts" in pdf.columns:
    pdf["ts"] = normalize_ts(pdf["ts"])
elif "timestamp" in pdf.columns:
    pdf["ts"] = normalize_ts(pdf["timestamp"])
else:
    raise ValueError("Parquet must contain 'ts' or 'timestamp' column.")

required = ["symbol","open","high","low","close","volume","ts"]
missing = [c for c in required if c not in pdf.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

pdf["symbol"] = pdf["symbol"].astype(str).str.replace("/", "", regex=False)
pdf = (pdf[["symbol","ts","open","high","low","close","volume"]]
       .dropna()
       .sort_values("ts"))

# Quick sanity check
print("HEAD:\n", pdf.head(3))
print("DTYPES:\n", pdf.dtypes)

sdf = spark.createDataFrame(pdf).withColumn("date", F.to_date("ts"))

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  symbol STRING,
  ts TIMESTAMP,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
  date DATE
) USING DELTA
""")

sdf.createOrReplaceTempView("staging_1h")
spark.sql(f"""
MERGE INTO {TABLE} AS t
USING staging_1h AS s
ON  t.symbol = s.symbol AND t.ts = s.ts
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

cnt = spark.table(TABLE).count()
print(f"[OK] Ingested 1h into {TABLE}. Count={cnt}")