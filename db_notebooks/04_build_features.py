# Databricks notebook source
# MAGIC %pip install -q pyarrow pandas numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# --- Settings ---
CATALOG = "workspace"
SCHEMA  = "mntrading"

BRONZE_5M = f"{CATALOG}.{SCHEMA}.bronze_ohlcv_5m"
SILVER_FEAT = f"{CATALOG}.{SCHEMA}.silver_features_5m"

import json
from pyspark.sql import functions as F, Window as W
from pyspark.sql import types as T

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE {SCHEMA}")

# --- Load screened pairs (defensive) ---
PAIRS_PATH = "/Volumes/workspace/mntrading/raw/pairs_screened.json"
pairs_raw = None
try:
    pairs_raw = dbutils.fs.head(PAIRS_PATH, 1024 * 1024)
except Exception as e:
    print(f"[WARN] Cannot read pairs file: {PAIRS_PATH} -> {e}")

pairs = []
if pairs_raw:
    try:
        data = json.loads(pairs_raw)
        # expected format: {"pairs": [["BTC/USDT","ETH/USDT"], ...]}
        for p in data.get("pairs", []):
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pairs.extend(p)
        pairs = sorted(set(pairs))
    except Exception as e:
        print(f"[WARN] Failed to parse JSON pairs: {e}")

if not pairs:
    print("[INFO] No screened pairs found. Falling back to distinct symbols present in bronze_ohlcv_5m.")
    try:
        pairs = [r["symbol"] for r in spark.table(BRONZE_5M).select("symbol").distinct().limit(50).collect()]
    except Exception as e:
        print(f"[ERROR] No pairs available at all: {e}")
        dbutils.notebook.exit("No pairs to build features.")

print(f"[INFO] Symbols to use: {pairs[:10]} (showing up to 10 of {len(pairs)})")

# --- Read 5m OHLCV (defensive schemas) ---
required_cols = ["ts", "symbol", "open", "high", "low", "close", "volume"]
df = spark.table(BRONZE_5M)
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column '{c}' in {BRONZE_5M}. Found: {df.columns}")

# Cast to expected types
df = (df
      .withColumn("ts", F.col("ts").cast("timestamp"))
      .withColumn("symbol", F.col("symbol").cast("string"))
      .withColumn("open", F.col("open").cast("double"))
      .withColumn("high", F.col("high").cast("double"))
      .withColumn("low",  F.col("low").cast("double"))
      .withColumn("close",F.col("close").cast("double"))
      .withColumn("volume",F.col("volume").cast("double"))
     )

df = df.filter(F.col("symbol").isin(pairs))

cnt = df.count()
print(f"[INFO] Bronze 5m rows after symbol filter: {cnt}")
if cnt == 0:
    dbutils.notebook.exit("No 5m data for selected symbols.")

# --- Basic technical features (serverless-safe) ---
w = W.partitionBy("symbol").orderBy(F.col("ts").cast("long"))

# returns
df = df.withColumn("ret_1", F.log(F.col("close")) - F.log(F.lag("close").over(w)))

# rolling stats
def roll_avg(col, n):
    return F.avg(col).over(w.rowsBetween(-(n-1), 0))
def roll_std(col, n):
    return F.stddev_samp(col).over(w.rowsBetween(-(n-1), 0))

for n in [5, 12, 24, 48]:
    df = df.withColumn(f"ma_{n}", roll_avg(F.col("close"), n))
for n in [12, 48]:
    df = df.withColumn(f"vol_{n}", roll_std(F.col("ret_1"), n))

# z-score of price
df = df.withColumn("price_mean_48", roll_avg(F.col("close"), 48))
df = df.withColumn("price_std_48",  roll_std(F.col("close"), 48))
df = df.withColumn(
    "z_48",
    F.when(F.col("price_std_48") > 1e-12, (F.col("close") - F.col("price_mean_48")) / F.col("price_std_48")).otherwise(F.lit(0.0))
)

# drop helper columns
df = df.drop("price_mean_48", "price_std_48")

# --- Final schema & null handling ---
feature_cols = [c for c in df.columns if c not in ("open","high","low","volume")]  # keep close + features
df_feat = df.select("ts", "symbol", "close", *[c for c in feature_cols if c not in ("ts","symbol","close")])

# Some windows return nulls at the head of each symbol; keep them but you can also drop if needed:
# df_feat = df_feat.na.drop()

print("[INFO] Feature sample:")
display(df_feat.orderBy(F.col("ts").desc()).limit(10))

# --- Write to Delta (mergeSchema to avoid mismatches) ---
(spark
 .createDataFrame(df_feat.rdd, df_feat.schema)  # materialize schema
 .write
 .format("delta")
 .mode("overwrite")              # for simplicity; switch to MERGE for strict incremental
 .option("mergeSchema", "true")
 .saveAsTable(SILVER_FEAT)
)

rows_after = spark.table(SILVER_FEAT).count()
print(f"[OK] Wrote {rows_after} rows into {SILVER_FEAT}")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS nfeats FROM gold_features_5m;
# MAGIC SELECT pair, AVG(zscore) avg_z, COUNT(*) n FROM gold_features_5m GROUP BY pair ORDER BY n DESC LIMIT 5;
# MAGIC