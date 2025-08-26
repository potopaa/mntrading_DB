# Databricks notebook source
# MAGIC %pip install -q pyarrow pandas numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# ---- Settings (UC) ----
CATALOG = "workspace"
SCHEMA  = "mntrading"

# ---- Spark ----
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE {SCHEMA}")

SRC = f"{CATALOG}.{SCHEMA}.bronze_ohlcv_1h"
OUT = f"{CATALOG}.{SCHEMA}.silver_pairs_screened"

# ---- Parameters ----
DAYS_WINDOW = 60
MIN_ABS_CORR = 0.7
PVAL_THRESHOLD = 0.05

# ---- Read last N days and pivot close prices to wide ----
cutoff = F.date_sub(F.current_date(), DAYS_WINDOW)
hourly = spark.table(SRC).where(F.col("date") >= cutoff)

close_wide = (
    hourly.select("ts","symbol","close")
          .groupBy("ts")
          .pivot("symbol")
          .agg(F.first("close"))
          .orderBy("ts")
)

symbols = [c for c in close_wide.columns if c != "ts"]
if len(symbols) < 3:
    raise ValueError("Not enough symbols to screen. Ingest more 1h data first.")

pdf = (close_wide.toPandas()
       .set_index("ts").sort_index().ffill().dropna(axis=1, how="any"))

# ---- Cointegration (Engle–Granger) + correlation prefilter ----
import numpy as np
from statsmodels.tsa.stattools import coint

pairs = []
sym_list = list(pdf.columns)
for i in range(len(sym_list)):
    for j in range(i+1, len(sym_list)):
        a, b = sym_list[i], sym_list[j]
        s1, s2 = pdf[a].values, pdf[b].values
        if np.std(s1) == 0 or np.std(s2) == 0:
            continue
        corr = float(np.corrcoef(s1, s2)[0,1])
        if abs(corr) < MIN_ABS_CORR:
            continue
        try:
            score, pvalue, _ = coint(s1, s2)
        except Exception:
            continue
        if pvalue < PVAL_THRESHOLD:
            pairs.append((a, b, corr, float(pvalue)))

pairs_df = spark.createDataFrame(pairs, schema="sym_a STRING, sym_b STRING, corr DOUBLE, pvalue DOUBLE")
pairs_df.write.mode("overwrite").format("delta").saveAsTable(OUT)

# After silver_pairs_screened is written:
OUT_JSON = "/Volumes/workspace/mntrading/raw/pairs_screened.json"

pairs_pdf = (spark.table(f"{CATALOG}.{SCHEMA}.silver_pairs_screened")
                 .select("sym_a","sym_b").distinct().toPandas())

pairs_pdf["sym_a"] = pairs_pdf["sym_a"].str.replace("/","", regex=False)
pairs_pdf["sym_b"] = pairs_pdf["sym_b"].str.replace("/","", regex=False)
pairs = sorted(set(pairs_pdf["sym_a"]).union(set(pairs_pdf["sym_b"])))

import json, os
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"symbols": pairs}, f, ensure_ascii=False, indent=2)

print(f"[OK] Exported screened symbols to {OUT_JSON} (n={len(pairs)})")


print(f"[OK] Screened pairs -> {OUT}. Count={pairs_df.count()}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS npairs FROM silver_pairs_screened;
# MAGIC SELECT * FROM silver_pairs_screened ORDER BY pvalue ASC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS npairs FROM silver_pairs_screened;
# MAGIC SELECT * FROM silver_pairs_screened ORDER BY pvalue ASC LIMIT 10;
# MAGIC