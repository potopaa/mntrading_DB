# Databricks notebook source
# 09_portfolio_report — robust price source selection + next-bar PnL
# Pairs mode (A,B,beta) with spread-diff; falls back to single-symbol mode if pair fields are absent.
# All comments are in English. Do not paste emojis or non-printable characters.

from pyspark.sql import SparkSession, functions as F, Window
from typing import Tuple, Optional, List

# -------------------- Settings --------------------
CATALOG = "workspace"
SCHEMA  = "mntrading"

# Signals table (produced by 08 notebook / inference)
SIGNALS_TBL = f"{CATALOG}.{SCHEMA}.gold_signals"

# Price sources to try (in this priority)
PRICE_TBL_CANDIDATES = [
    f"{CATALOG}.{SCHEMA}.silver_features_5m",  # preferred if present
    f"{CATALOG}.{SCHEMA}.bronze_ohlcv_5m",     # fallback table if present
]

# Last-resort parquet fallback (Volumes). For Spark SQL use dbfs: path.
PARQUET_FALLBACK = "dbfs:/Volumes/workspace/mntrading/raw/ohlcv_5m.parquet"

# Sanity thresholds
THRESH_ABS_LOGRET = 0.5  # drop bars where |Δln(price)| > 50% on 5m (very likely bad tick)
BETA_MIN, BETA_MAX = 0.1, 10.0

# -------------------- Spark --------------------
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.ansi.enabled", "true")

# -------------------- Helpers --------------------
def try_table(tbl_name: str):
    """Return DataFrame if table exists, otherwise None (no exception propagates)."""
    try:
        return spark.table(tbl_name)
    except Exception:
        return None

def normalize_prices_source(df):
    """
    Normalize any price-like source to schema:
      - columns: ts (timestamp), symbol (string), close (double)
    Accepts sources with alternative names and casts types explicitly.
    """
    cols = set(df.columns)

    # timestamp column -> 'ts'
    if "ts" not in cols:
        for alt in ("timestamp", "time", "datetime"):
            if alt in cols:
                df = df.withColumnRenamed(alt, "ts")
                break
    if "ts" not in df.columns:
        raise RuntimeError("Prices source must contain a timestamp column (ts/timestamp/time/datetime).")

    # symbol column -> 'symbol'
    if "symbol" not in df.columns:
        for alt in ("pair", "ticker", "instrument"):
            if alt in df.columns:
                df = df.withColumnRenamed(alt, "symbol")
                break
    if "symbol" not in df.columns:
        raise RuntimeError("Prices source must contain a symbol column (symbol/pair/ticker/instrument).")

    # close column -> 'close'
    if "close" not in df.columns:
        # Derive from OHLC if possible (no self-reference to 'close' here)
        if all(c in df.columns for c in ("open", "high", "low")):
            df = df.withColumn("close", (F.col("open") + F.col("high") + F.col("low")) / F.lit(3.0))
        else:
            raise RuntimeError("Prices source must have 'close' or OHLC to derive it.")

    return (df
            .withColumn("ts", F.col("ts").cast("timestamp"))
            .withColumn("symbol", F.col("symbol").cast("string"))
            .withColumn("close", F.col("close").cast("double")))

def pick_prices_source():
    """
    Pick the first available price table from candidates; otherwise read parquet fallback.
    Returns (DataFrame, source_name_string).
    """
    for t in PRICE_TBL_CANDIDATES:
        df = try_table(t)
        if df is not None:
            return normalize_prices_source(df), t
    df = spark.read.parquet(PARQUET_FALLBACK)
    return normalize_prices_source(df), PARQUET_FALLBACK

def ensure_signal_column(df_sig):
    """
    Ensure presence of 'signal' column:
      - use 'signal' if exists
      - else 'action' or 'pred'
      - else derive from 'proba' via thresholds (long/short/flat)
    """
    if "signal" in df_sig.columns:
        return df_sig
    if "action" in df_sig.columns:
        return df_sig.withColumnRenamed("action", "signal")
    if "pred" in df_sig.columns:
        return df_sig.withColumnRenamed("pred", "signal")
    if "proba" in df_sig.columns:
        return df_sig.withColumn(
            "signal",
            F.when(F.col("proba") > F.lit(0.55), F.lit(1))
             .when(F.col("proba") < F.lit(0.45), F.lit(-1))
             .otherwise(F.lit(0))
        )
    raise RuntimeError("Signals must contain one of: signal/action/pred/proba.")

def floor_to_5m(col_ts):
    """Floor timestamp to a 5-minute grid using integer division on UNIX seconds."""
    return F.from_unixtime((F.unix_timestamp(col_ts) / 300).cast("bigint") * 300).cast("timestamp")

def detect_pairs_schema(df):
    """
    Return a normalized DataFrame for pairs if pair fields are present; otherwise None.
    Normalized columns: ts (timestamp), symbol_a (string), symbol_b (string), beta (double), signal (int)
    Also supports alternative names: a/b, leg_a/leg_b, base/quote; beta could be hedge_ratio.
    If only a 'pair' string like 'AAA/USDT|BBB/USDT' is present, split it.
    """
    cols = set(df.columns)
    # Try to standardize timestamp
    if "ts" not in cols and "timestamp" in cols:
        df = df.withColumnRenamed("timestamp", "ts")
        cols = set(df.columns)

    # Try to standardize signal
    df = ensure_signal_column(df)
    cols = set(df.columns)

    # Try symbol_a / symbol_b
    candidates_a = [c for c in ("symbol_a", "sym_a", "a", "leg_a", "base") if c in cols]
    candidates_b = [c for c in ("symbol_b", "sym_b", "b", "leg_b", "quote") if c in cols]
    candidates_beta = [c for c in ("beta", "hedge_ratio", "hr", "b_hat") if c in cols]

    if candidates_a and candidates_b and candidates_beta:
        df2 = (df
               .withColumnRenamed(candidates_a[0], "symbol_a")
               .withColumnRenamed(candidates_b[0], "symbol_b")
               .withColumnRenamed(candidates_beta[0], "beta"))
        return (df2
                .withColumn("ts", F.col("ts").cast("timestamp"))
                .withColumn("symbol_a", F.col("symbol_a").cast("string"))
                .withColumn("symbol_b", F.col("symbol_b").cast("string"))
                .withColumn("beta", F.col("beta").cast("double"))
                .withColumn("signal", F.col("signal").cast("int"))
                .select("ts", "symbol_a", "symbol_b", "beta", "signal"))

    # Try to parse 'pair' like 'AAA/USDT|BBB/USDT' or 'AAAUSDT,BBBUSDT'
    if "pair" in cols or "pair_id" in cols:
        pair_col = "pair" if "pair" in cols else "pair_id"
        df2 = df
        # Heuristic split into two symbols
        df2 = df2.withColumn(
            "_split",
            F.split(F.regexp_replace(F.col(pair_col), r"[,\|\s]+", "|"), r"\|")
        )
        df2 = (df2
               .withColumn("symbol_a", F.col("_split").getItem(0))
               .withColumn("symbol_b", F.col("_split").getItem(1)))
        # Try beta column or set null (will be filtered later)
        if "beta" not in df2.columns:
            df2 = df2.withColumn("beta", F.lit(None).cast("double"))
        return (df2
                .withColumn("ts", F.col("ts").cast("timestamp"))
                .withColumn("symbol_a", F.col("symbol_a").cast("string"))
                .withColumn("symbol_b", F.col("symbol_b").cast("string"))
                .withColumn("beta", F.col("beta").cast("double"))
                .withColumn("signal", F.col("signal").cast("int"))
                .select("ts", "symbol_a", "symbol_b", "beta", "signal"))

    return None

def summarize_span(df, label: str):
    r = df.agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts"), F.count("*").alias("rows")).collect()[0]
    print(f"[INFO] {label} rows={r['rows']}, span=({r['min_ts']}, {r['max_ts']})")

# -------------------- Load signals --------------------
df_sig = try_table(SIGNALS_TBL)
if df_sig is None:
    raise RuntimeError(f"Signals table not found: {SIGNALS_TBL}")

# First, try pairs mode
df_pairs = detect_pairs_schema(df_sig)

# Also prepare single-symbol fallback
df_single = None
if df_pairs is None:
    # Standardize minimal schema: ts, symbol, signal
    tmp = df_sig
    if "ts" not in tmp.columns and "timestamp" in tmp.columns:
        tmp = tmp.withColumnRenamed("timestamp", "ts")
    if "symbol" not in tmp.columns and "pair" in tmp.columns:
        tmp = tmp.withColumnRenamed("pair", "symbol")
    tmp = ensure_signal_column(tmp)
    df_single = (tmp
                 .withColumn("ts", F.col("ts").cast("timestamp"))
                 .withColumn("symbol", F.col("symbol").cast("string"))
                 .withColumn("signal", F.col("signal").cast("int"))
                 .select("ts", "symbol", "signal"))

# -------------------- Load prices (auto-pick source) --------------------
df_px, px_src = pick_prices_source()
print(f"[INFO] price source: {px_src}")
summarize_span(df_px, "prices")

# -------------------- Common preprocessing: time overlap & 5m grid --------------------
# Compute spans
rng_sig = (df_pairs if df_pairs is not None else df_single) \
    .agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts")).collect()[0]
rng_px  = df_px.agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts")).collect()[0]

overlap_start = max(rng_sig["min_ts"], rng_px["min_ts"]) if rng_sig["min_ts"] and rng_px["min_ts"] else None
overlap_end   = min(rng_sig["max_ts"], rng_px["max_ts"]) if rng_sig["max_ts"] and rng_px["max_ts"] else None
print("[INFO] overlap=({}, {})".format(overlap_start, overlap_end))

if overlap_start is None or overlap_end is None or overlap_end <= overlap_start:
    print("[WARN] No time overlap between signals and prices; aborting PnL computation.")
    display(spark.createDataFrame([], "date date, rows bigint, sum_pnl double, avg_pnl double"))
    dbutils.notebook.exit("No overlap; update 08_batch_inference or extend price window.")

# Align prices to 5m and create next-bar columns
df_px5 = (df_px
          .withColumn("ts5", floor_to_5m(F.col("ts")))
          .select("symbol", "ts5", "close"))

w_sym = Window.partitionBy("symbol").orderBy("ts5")
df_px5 = (df_px5
          .withColumn("close_next", F.lead("close").over(w_sym))
          .withColumn("ts5_next", F.lead("ts5").over(w_sym))
          .filter(F.col("close").isNotNull()))

# -------------------- Pairs mode --------------------
if df_pairs is not None:
    print("[INFO] running PAIRS mode")

    # Clip signals to overlap and floor to 5m grid
    df_sig5 = (df_pairs
               .withColumn("ts5", floor_to_5m(F.col("ts")))
               .filter((F.col("ts5") >= F.lit(overlap_start)) & (F.col("ts5") <= F.lit(overlap_end)))
               .select("ts5", "symbol_a", "symbol_b", "beta", "signal"))

    # Drop invalid beta
    df_sig5 = df_sig5.filter(F.col("beta").isNotNull() & (F.col("beta") >= F.lit(BETA_MIN)) & (F.col("beta") <= F.lit(BETA_MAX)))

    # Reduce price table to only symbols that appear in signals
    syms = (df_sig5
            .select(F.col("symbol_a").alias("symbol")).unionByName(df_sig5.select(F.col("symbol_b").alias("symbol")))
            .distinct())
    df_px5_sub = df_px5.join(syms, on="symbol", how="inner")

    # Join A leg
    pxa = df_px5_sub.select(
        F.col("symbol").alias("symbol_a"),
        F.col("ts5").alias("ts5"),
        F.col("close").alias("close_a"),
        F.col("close_next").alias("close_next_a"),
        F.col("ts5_next").alias("ts5_next_a"),
    )
    # Join B leg
    pxb = df_px5_sub.select(
        F.col("symbol").alias("symbol_b"),
        F.col("ts5").alias("ts5"),
        F.col("close").alias("close_b"),
        F.col("close_next").alias("close_next_b"),
        F.col("ts5_next").alias("ts5_next_b"),
    )

    df_join = (df_sig5
               .join(pxa, on=["symbol_a", "ts5"], how="inner")
               .join(pxb, on=["symbol_b", "ts5"], how="inner"))

    # Require both next bars present
    df_join = df_join.filter(F.col("close_next_a").isNotNull() & F.col("close_next_b").isNotNull())

    # Compute next-bar log returns and spread diff
    def safe_log(x):
        return F.log(F.when(x <= 0, None).otherwise(x))  # guard against non-positive prices

    ln_a_now  = safe_log(F.col("close_a"))
    ln_a_next = safe_log(F.col("close_next_a"))
    ln_b_now  = safe_log(F.col("close_b"))
    ln_b_next = safe_log(F.col("close_next_b"))

    dln_a = ln_a_next - ln_a_now
    dln_b = ln_b_next - ln_b_now

    df_join = df_join.withColumn("dln_a", dln_a).withColumn("dln_b", dln_b)
    # Sanity filter on raw legs
    df_join = df_join.filter(F.abs(F.col("dln_a")) <= F.lit(THRESH_ABS_LOGRET)).filter(F.abs(F.col("dln_b")) <= F.lit(THRESH_ABS_LOGRET))

    # Spread increment Δs_t = ΔlnA - beta * ΔlnB
    df_join = df_join.withColumn("ret_pair", F.col("dln_a") - F.col("beta") * F.col("dln_b"))

    # Optional normalization by previous gross exposure; keeps units closer to % of notional
    # gross_exposure_t = |1|*close_a + |beta|*close_b (using current bar prices as "previous" for next-bar PnL)
    df_join = df_join.withColumn("gross_expo", F.abs(F.lit(1.0)) * F.col("close_a") + F.abs(F.col("beta")) * F.col("close_b"))
    # Avoid division by zero
    df_join = df_join.withColumn("ret_norm", F.when(F.col("gross_expo") > 0, F.col("ret_pair")).otherwise(F.lit(0.0)))

    # Next-bar PnL using current signal (no peeking, signal at t applied to t->t+1 move)
    df_join = df_join.withColumn("pnl", F.col("signal") * F.col("ret_norm"))

    # Labels for grouping
    df_join = df_join.withColumn("pair", F.concat_ws("||", F.col("symbol_a"), F.col("symbol_b")))

    # Summaries
    summary_total = (df_join
                     .agg(F.count("*").alias("rows"),
                          F.sum("pnl").alias("sum_pnl"),
                          F.avg("pnl").alias("avg_pnl")))
    display(summary_total)

    df_daily = (df_join
                .withColumn("date", F.to_date("ts5"))
                .groupBy("date")
                .agg(F.count("*").alias("rows"),
                     F.sum("pnl").alias("sum_pnl"),
                     F.avg("pnl").alias("avg_pnl"))
                .orderBy("date"))
    display(df_daily)

    df_by_pair = (df_join
                  .groupBy("pair")
                  .agg(F.count("*").alias("rows"),
                       F.sum("pnl").alias("sum_pnl"),
                       F.avg("pnl").alias("avg_pnl"))
                  .orderBy(F.desc("sum_pnl")))
    display(df_by_pair)

    # Expose useful temp views
    spark.sql("DROP VIEW IF EXISTS pair_pnl_5m")
    df_join.select("ts5", "symbol_a", "symbol_b", "beta", "signal", "ret_pair", "pnl").createOrReplaceTempView("pair_pnl_5m")
    print("[INFO] Created TEMP VIEW pair_pnl_5m (ts5, symbol_a, symbol_b, beta, signal, ret_pair, pnl).")

# -------------------- Single-symbol fallback mode --------------------
else:
    print("[INFO] running SINGLE-SYMBOL mode")

    df_sig5 = (df_single
               .withColumn("ts5", floor_to_5m(F.col("ts")))
               .filter((F.col("ts5") >= F.lit(overlap_start)) & (F.col("ts5") <= F.lit(overlap_end)))
               .select("symbol", "ts5", "signal"))

    df_join = (df_sig5.alias("s")
               .join(df_px5.alias("p"), on=["symbol", "ts5"], how="inner")
               .filter(F.col("p.close_next").isNotNull())
               .select("s.symbol", "s.ts5", "s.signal", "p.close", "p.close_next"))

    # Next-bar simple return
    ret = (F.col("close_next") - F.col("close")) / F.col("close")
    df_join = (df_join
               .withColumn("ret", ret)
               .withColumn("pnl", F.col("ret") * F.col("signal")))

    summary_total = (df_join
                     .agg(F.count("*").alias("rows"),
                          F.sum("pnl").alias("sum_pnl"),
                          F.avg("pnl").alias("avg_pnl")))
    display(summary_total)

    df_daily = (df_join
                .withColumn("date", F.to_date("ts5"))
                .groupBy("date")
                .agg(F.count("*").alias("rows"),
                     F.sum("pnl").alias("sum_pnl"),
                     F.avg("pnl").alias("avg_pnl"))
                .orderBy("date"))
    display(df_daily)

    df_by_symbol = (df_join
                    .groupBy("symbol")
                    .agg(F.count("*").alias("rows"),
                         F.sum("pnl").alias("sum_pnl"),
                         F.avg("pnl").alias("avg_pnl"))
                    .orderBy(F.desc("sum_pnl")))
    display(df_by_symbol)

# -------------------- Unified price view for any SQL cells --------------------
# Create a temp view 'prices_5m' so that subsequent %sql cells can safely use it.
spark.sql("DROP VIEW IF EXISTS prices_5m")
(spark.createDataFrame([], schema="symbol string, ts5 timestamp, close double")  # placeholder schema
     .limit(0)
     .createOrReplaceTempView("prices_5m"))
df_px5.select("symbol", "ts5", "close").createOrReplaceTempView("prices_5m")
print("[INFO] Created TEMP VIEW prices_5m with unified schema (symbol, ts5, close).")
print("[INFO] Use it in SQL: SELECT * FROM prices_5m LIMIT 10;")

# -------------------- Diagnostics --------------------
matches = ( (df_pairs.select("symbol_a", "symbol_b") if df_pairs is not None else df_single.select("symbol").distinct())
           if (df_pairs is not None or df_single is not None) else spark.createDataFrame([], "dummy int") )

# For brevity, show a random slice if available
try:
    display(df_px5.orderBy(F.rand()).limit(50))
except Exception:
    pass