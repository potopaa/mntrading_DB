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
from pyspark.sql import SparkSession, functions as F, Window
spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE {SCHEMA}")

FEATS = f"{CATALOG}.{SCHEMA}.gold_features_5m"
OUT   = f"{CATALOG}.{SCHEMA}.gold_dataset"

f = spark.table(FEATS).orderBy("ts")
w = Window.partitionBy("pair").orderBy("ts")

HORIZON = 3  # next 3 bars forward return
f2 = (f
      .withColumn("future_a_close", F.lead("a_close", HORIZON).over(w))
      .withColumn("ret_fwd", (F.col("future_a_close") - F.col("a_close"))/F.col("a_close"))
      .withColumn("label", F.when(F.col("ret_fwd")>0, F.lit(1)).otherwise(F.lit(0)))
      .drop("future_a_close"))

ds = f2.select("ts","pair","beta","alpha","spread","zscore","label").dropna()
ds.write.mode("overwrite").format("delta").saveAsTable(OUT)

print(f"[OK] Dataset saved to {OUT}. Count={spark.table(OUT).count()}")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS nds FROM gold_dataset;
# MAGIC SELECT label, COUNT(*) FROM gold_dataset GROUP BY label;
# MAGIC