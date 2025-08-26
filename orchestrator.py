# orchestrator.py
# ==========================================
# Local orchestrator:
# - fetch 1h OHLCV locally via CCXT
# - upload to UC Volumes
# - trigger Databricks Job A (01->02)
# - download pairs_screened.json, parse symbols robustly
# - fetch 5m OHLCV for selected symbols
# - upload 5m
# - trigger Databricks Job B (03..09)
# ==========================================

from __future__ import annotations

import os
import json
import time
import pathlib
from typing import List, Dict, Any, Iterable, Set

import pandas as pd

# local modules
from tools.databricks_io import upload, download
import fetchers

# Databricks SDK (env DATABRICKS_HOST/TOKEN must be set)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

# ---------- Config ----------
STAGE_DIR = pathlib.Path("./stage"); STAGE_DIR.mkdir(exist_ok=True, parents=True)

# UC volume paths
VOL_1H    = "/Volumes/workspace/mntrading/raw/ohlcv_1h.parquet"
VOL_5M    = "/Volumes/workspace/mntrading/raw/ohlcv_5m.parquet"
VOL_PAIRS = "/Volumes/workspace/mntrading/raw/pairs_screened.json"

# local stage paths
LOC_1H    = str(STAGE_DIR / "ohlcv_1h.parquet")
LOC_5M    = str(STAGE_DIR / "ohlcv_5m.parquet")
LOC_PAIRS = str(STAGE_DIR / "pairs_screened.json")

# symbols universe (1h fetch)
SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
    "ADA/USDT","DOGE/USDT","LTC/USDT","TRX/USDT","MATIC/USDT",
]

# jobs (set these in environment)
JOB_ID_A = int(os.environ["DB_JOB_A"])  # 01_ingest_1h -> 02_screen_pairs
JOB_ID_B = int(os.environ["DB_JOB_B"])  # 03..09

# fetch params
LIMIT_1H   = 1000
LIMIT_5M   = 1000
SLEEP_MS   = 350
EXCHANGE   = "binance"

# ---------- Helpers ----------
def uniq_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def parse_screened_symbols(json_obj: Any) -> List[str]:
    """
    Accepts many shapes:
    - dict with keys: symbols | selected_symbols | top_symbols -> list[str]
    - dict with keys: pairs | selected_pairs | cointegrated_pairs -> list[str] OR list[dict]
      * dict item may have: 'pair' (str) OR ('symbol','symbol2') OR ('symbol1','symbol2')
      * если это действительно пары активов, берём объединённое множество символов
    - list[str]
    - list[dict] (как выше)
    Returns list[str] (distinct, original order if possible).
    """
    def from_pairs_list(lst: List[Any]) -> List[str]:
        acc: List[str] = []
        for it in lst:
            if isinstance(it, str):
                acc.append(it)
            elif isinstance(it, dict):
                if "symbol" in it and isinstance(it["symbol"], str):
                    acc.append(it["symbol"])
                elif "pair" in it and isinstance(it["pair"], str):
                    acc.append(it["pair"])
                else:
                    # cointegrated pair as two symbols
                    s1 = it.get("symbol1") or it.get("base") or it.get("s1")
                    s2 = it.get("symbol2") or it.get("quote") or it.get("s2")
                    if isinstance(s1, str):
                        acc.append(s1)
                    if isinstance(s2, str):
                        acc.append(s2)
        return acc

    if isinstance(json_obj, dict):
        # direct lists of strings
        for key in ("symbols", "selected_symbols", "top_symbols"):
            val = json_obj.get(key)
            if isinstance(val, list):
                strs = [x for x in val if isinstance(x, str)]
                if strs:
                    return uniq_keep_order(strs)

        # pairs-like
        for key in ("pairs", "selected_pairs", "cointegrated_pairs"):
            val = json_obj.get(key)
            if isinstance(val, list):
                got = from_pairs_list(val)
                if got:
                    return uniq_keep_order(got)

        # sometimes inside nested 'data' or similar
        data = json_obj.get("data")
        if data is not None:
            return parse_screened_symbols(data)

        return []

    if isinstance(json_obj, list):
        # list[str]
        if all(isinstance(x, str) for x in json_obj):
            return uniq_keep_order(json_obj)  # type: ignore[arg-type]
        # list[dict/any]
        return uniq_keep_order(from_pairs_list(json_obj))

    # unknown shape
    return []

def load_screened_symbols_from_file(path: str, fallback: List[str]) -> List[str]:
    # read file (try JSON first; if fails — try JSON Lines)
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        # debug preview
        print(f"[pairs] local file bytes={len(txt.encode('utf-8'))}, head={txt[:240].replace(os.linesep,' ')}")
        try:
            obj = json.loads(txt)
            syms = parse_screened_symbols(obj)
            if syms:
                print(f"[pairs] parsed symbols: {len(syms)} (e.g. {syms[:10]})")
                return syms
        except json.JSONDecodeError:
            # try json lines
            syms_acc: List[str] = []
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                syms_acc.extend(parse_screened_symbols(obj))
            syms_acc = uniq_keep_order(syms_acc)
            if syms_acc:
                print(f"[pairs] parsed symbols (jsonl): {len(syms_acc)} (e.g. {syms_acc[:10]})")
                return syms_acc
    except FileNotFoundError:
        print(f"[pairs] file not found: {path}")

    # fallback
    print("[pairs] No symbols parsed from file; falling back to 1h universe.")
    fb = uniq_keep_order(fallback)
    if not fb:
        raise RuntimeError("No screened symbols found and fallback is empty.")
    return fb

def run_job_and_wait(w: WorkspaceClient, job_id: int, timeout_min: int = 90) -> None:
    # start
    resp = w.jobs.run_now(job_id=job_id)
    run_id = resp.run_id
    print(f"[job {job_id}] started run_id={run_id}")
    print(f"[job {job_id}] UI: {w.config.host.rstrip('/')}/#job/{job_id}/run/{run_id}")

    # poll
    deadline = time.time() + timeout_min * 60
    last_lc, last_res = None, None
    while True:
        r = w.jobs.get_run(run_id=run_id, include_history=True, include_resolved_values=True)
        lc = r.state.life_cycle_state
        rs = r.state.result_state
        # task summary (best-effort)
        try:
            tasks = w.jobs.list_run_tasks(run_id=run_id).tasks or []
            parts = []
            for t in tasks:
                parts.append(f"{t.task_key}={t.state.life_cycle_state}/{t.state.result_state}")
            if parts:
                print(f"[job {job_id}] state={lc} result={rs} :: " + ", ".join(parts))
            else:
                print(f"[job {job_id}] state={lc} result={rs}")
        except Exception:
            print(f"[job {job_id}] state={lc} result={rs}")

        if lc == RunLifeCycleState.TERMINATED:
            if rs == RunResultState.SUCCESS:
                print(f"[job {job_id}] SUCCESS")
                return
            else:
                raise RuntimeError(f"Job {job_id} failed: {rs}")

        if time.time() > deadline:
            raise TimeoutError(f"Job {job_id} timed out after {timeout_min} minutes.")

        time.sleep(10)

# ---------- Main ----------
def main():
    # 0) Databricks client
    w = WorkspaceClient()

    # 1) Fetch 1h locally
    df_1h = fetchers.fetch_ohlcv_batch(
        symbols=SYMBOLS,
        timeframe="1h",
        limit=LIMIT_1H,
        sleep_ms=SLEEP_MS,
        exchange_id=EXCHANGE,
        verbose=True,
    )
    if df_1h.empty:
        raise RuntimeError("Fetched 1h DataFrame is empty.")
    df_1h.to_parquet(LOC_1H, index=False)
    print(f"[write] {LOC_1H} rows={len(df_1h)}")

    # 2) Upload 1h and run Job A
    upload(LOC_1H, VOL_1H)
    run_job_and_wait(w, JOB_ID_A, timeout_min=60)

    # 3) Download pairs_screened.json and parse symbols
    download(VOL_PAIRS, LOC_PAIRS)
    # fallback symbols — all unique from 1h pull
    fallback = df_1h["symbol"].dropna().astype(str).tolist()
    fallback = uniq_keep_order(fallback)
    screened = load_screened_symbols_from_file(LOC_PAIRS, fallback=fallback)
    if not screened:
        raise RuntimeError("No screened symbols found after parsing and fallback.")

    # 4) Fetch 5m for screened symbols
    df_5m = fetchers.fetch_ohlcv_batch(
        symbols=screened,
        timeframe="5m",
        limit=LIMIT_5M,
        sleep_ms=SLEEP_MS,
        exchange_id=EXCHANGE,
        verbose=True,
    )
    if df_5m.empty:
        raise RuntimeError("Fetched 5m DataFrame is empty.")
    df_5m.to_parquet(LOC_5M, index=False)
    print(f"[write] {LOC_5M} rows={len(df_5m)}")

    # 5) Upload 5m and run Job B
    upload(LOC_5M, VOL_5M)
    run_job_and_wait(w, JOB_ID_B, timeout_min=120)

    print("[DONE] Orchestration complete.")

if __name__ == "__main__":
    main()
