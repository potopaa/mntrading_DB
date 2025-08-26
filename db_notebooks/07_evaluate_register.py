# Databricks notebook source
# MAGIC %pip install -q pyarrow pandas numpy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Databricks notebook source
# ==========================================
# 07_publish_file_based.py — publish champion (file-based, serverless-safe)
# ==========================================

CATALOG = "workspace"
SCHEMA  = "mntrading"

CANDIDATES_SUMMARY = "/Volumes/workspace/mntrading/raw/models/last_candidates.json"
CANDIDATES_ROOT    = "/Volumes/workspace/mntrading/raw/models/candidates"
PRODUCTION_DIR     = "/Volumes/workspace/mntrading/raw/models/production"

import json, time

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE {SCHEMA}")

def path_exists(p: str) -> bool:
    try:
        dbutils.fs.ls(p)
        return True
    except Exception:
        return False

def read_json(dbfs_path: str) -> dict:
    raw = dbutils.fs.head(dbfs_path, 5 * 1024 * 1024)
    return json.loads(raw)

# ---- 1) Read summary with best candidate ----
if not path_exists(CANDIDATES_SUMMARY):
    dbutils.notebook.exit(f"Candidates summary not found: {CANDIDATES_SUMMARY}")

summary = read_json(CANDIDATES_SUMMARY)
best = summary.get("best_model") or {}
name = best.get("name")
uri  = best.get("uri")
auc  = best.get("auc")

if not (name and uri):
    dbutils.notebook.exit(f"Invalid candidates summary in {CANDIDATES_SUMMARY}: {summary}")

print(f"[PUBLISH] Best candidate: name={name}, AUC={auc}, uri={uri}")

# Sanity: candidate folder must exist
if not path_exists(uri):
    dbutils.notebook.exit(f"Candidate folder not found: {uri}")

# ---- 2) Copy candidate -> production (clean replace) ----
try:
    dbutils.fs.rm(PRODUCTION_DIR, True)
except Exception:
    pass

dbutils.fs.mkdirs("/Volumes/workspace/mntrading/raw/models")  # ensure parent
dbutils.fs.cp(uri, PRODUCTION_DIR, recurse=True)
print(f"[PUBLISH] Copied {uri} -> {PRODUCTION_DIR}")

# ---- 3) Write champion_meta.json in production ----
champion_meta = {
    "published_ts": int(time.time()),
    "candidate": {
        "name": name,
        "auc": auc,
        "source_uri": uri
    }
}
dbutils.fs.put(f"{PRODUCTION_DIR}/champion_meta.json", json.dumps(champion_meta), overwrite=True)
print(f"[PUBLISH] Wrote {PRODUCTION_DIR}/champion_meta.json")

# ---- 4) Quick check of expected files ----
expected_any = ["model.json", "model.pkl"]  # format may vary (JSON LR or sklearn pickle)
present = {f.name for f in dbutils.fs.ls(PRODUCTION_DIR)}
has_model = any(x in present for x in expected_any)
has_meta  = "meta.json" in present

print(f"[CHECK] production contents: {sorted(list(present))}")
if not has_model:
    print("[WARN] No model file detected (model.json/model.pkl). Verify previous step (06) wrote correct artifacts.")
if not has_meta:
    print("[WARN] meta.json not found in production. If you use JSON-LR model, 06 should have written meta.json.")

print("[OK] Champion published (file-based).")
