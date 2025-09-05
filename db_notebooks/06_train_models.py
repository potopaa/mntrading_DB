# Databricks notebook source
# MAGIC %pip install -q pyarrow pandas numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# ==========================================
# 06_train_models.py — scikit-learn Logistic Regression → JSON in UC Volumes
# ==========================================

CATALOG = "workspace"
SCHEMA  = "mntrading"

CANDIDATE_DATASETS = [
    f"{CATALOG}.{SCHEMA}.gold_training_dataset",
    f"{CATALOG}.{SCHEMA}.gold_dataset",
    f"{CATALOG}.{SCHEMA}.silver_features_5m",
]

LABEL_COL = "label"
MAX_ROWS_PANDAS = 300_000
TEST_FRACTION   = 0.2
RANDOM_STATE    = 42

CANDIDATES_DIR  = "/Volumes/workspace/mntrading/raw/models/candidates"
CAND_NAME       = "logreg_json"
CAND_DIR        = f"{CANDIDATES_DIR}/{CAND_NAME}"
CANDIDATES_JSON = "/Volumes/workspace/mntrading/raw/models/last_candidates.json"
LOGS_DIR        = "/Volumes/workspace/mntrading/raw/train_logs"

# ---------- Imports ----------
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import json, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE {SCHEMA}")

def table_exists_uc(qualified: str) -> bool:
    try:
        cat, sch, tbl = qualified.split(".")
        return len(spark.sql(f"SHOW TABLES IN {cat}.{sch} LIKE '{tbl}'").collect()) > 0
    except Exception:
        return False

def log_json(run_name: str, params: dict, metrics: dict) -> None:
    try:
        dbutils.fs.mkdirs(LOGS_DIR)
        p = f"{LOGS_DIR}/{run_name}_{int(time.time())}.json"
        dbutils.fs.put(p, json.dumps({"run_name": run_name, "params": params, "metrics": metrics}), overwrite=True)
        print(f"[LOG] {p}")
    except Exception as e:
        print(f"[WARN] failed to write log JSON: {e}")


data_tbl = next((t for t in CANDIDATE_DATASETS if table_exists_uc(t)), None)
if not data_tbl:
    dbutils.notebook.exit("No training dataset table found.")

df = spark.table(data_tbl)
total_all = df.count()
print(f"[INFO] Using training dataset: {data_tbl}; rows={total_all}")

if LABEL_COL not in df.columns:
    dbutils.notebook.exit(f"'{LABEL_COL}' column is missing in {data_tbl}.")


numeric_types = {"double", "float", "int", "bigint", "smallint", "tinyint"}
skip_cols = {LABEL_COL, "ts", "timestamp", "symbol", "pair", "id"}
feature_cols = [
    f.name for f in df.schema
    if getattr(f.dataType, "simpleString", lambda: f.dataType.simpleString())() in numeric_types
    and f.name not in skip_cols
]
if not feature_cols:
    dbutils.notebook.exit("No numeric feature columns detected.")


for c in feature_cols:
    df = df.withColumn(c, F.col(c).cast("float"))
df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("int"))
df = df.na.drop(subset=feature_cols + [LABEL_COL])


frac = min(1.0, MAX_ROWS_PANDAS / max(1, total_all))
df_sample = df.sample(False, frac, seed=RANDOM_STATE) if frac < 1.0 else df
if frac < 1.0:
    print(f"[INFO] Downsampled to fraction={frac:.4f} (~{int(frac*total_all)} rows)")

pdf = df_sample.select(*(feature_cols + [LABEL_COL])).toPandas()
print(f"[INFO] pandas sample shape: {pdf.shape}")


X = pdf[feature_cols].astype(np.float32).values
y = pdf[LABEL_COL].astype(np.int32).values
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE, stratify=y
)


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

clf = LogisticRegression(max_iter=300, solver="lbfgs")
clf.fit(X_train_s, y_train)
proba = clf.predict_proba(X_val_s)[:, 1]
auc   = float(roc_auc_score(y_val, proba))
print(f"[MODEL] {CAND_NAME} AUC={auc:.4f}")


dbutils.fs.mkdirs(CAND_DIR)

model_json = {
    "name": CAND_NAME,
    "format": "json_lr",
    "framework": "sklearn",
    "features": feature_cols,
    "label": LABEL_COL,
    "coef": clf.coef_[0].astype(float).tolist(),
    "intercept": float(clf.intercept_[0]),
    "scaler": {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_).astype(float).tolist()
    },
    "metrics": {"auc": auc}
}
dbutils.fs.put(f"{CAND_DIR}/model.json", json.dumps(model_json), overwrite=True)

meta = {"features": feature_cols, "label": LABEL_COL, "format": "json_lr"}
dbutils.fs.put(f"{CAND_DIR}/meta.json", json.dumps(meta), overwrite=True)

# log
log_json("cv_logreg_json",
         params={"candidate": CAND_NAME, "dataset_table": data_tbl, "n_features": len(feature_cols), "rows_total": int(total_all)},
         metrics={"auc": auc})


summary = {"best_model": {"name": CAND_NAME, "auc": auc, "uri": CAND_DIR}, "candidates": [{"name": CAND_NAME, "auc": auc, "uri": CAND_DIR}]}
dbutils.fs.mkdirs("/Volumes/workspace/mntrading/raw/models")
dbutils.fs.put(CANDIDATES_JSON, json.dumps(summary), overwrite=True)

print(f"[OK] Best model: {CAND_NAME} AUC={auc:.4f} uri={CAND_DIR}")
