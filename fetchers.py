from __future__ import annotations
import os
import time
from typing import Iterable, List, Optional, Dict, Any
import pandas as pd

try:
    import ccxt
except Exception as e:
    raise RuntimeError(
        "CCXT is required. Install with: pip install ccxt"
    ) from e


# ---------- Exchange factory ----------

def make_exchange(
    exchange_id: str = "binance",
    enable_rate_limit: bool = True,
    **kwargs: Any,
):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange_id: {exchange_id}")

    klass = getattr(ccxt, exchange_id)
    args: Dict[str, Any] = {
        "enableRateLimit": enable_rate_limit,
        "timeout": kwargs.pop("timeout", 30000),
    }

    api_key = os.getenv("EXCHANGE_API_KEY")
    api_secret = os.getenv("EXCHANGE_API_SECRET")
    if api_key and api_secret:
        args["apiKey"] = api_key
        args["secret"] = api_secret

    args.update(kwargs)

    ex = klass(args)
    return ex


# ---------- Core fetch loop per symbol ----------

def _fetch_symbol_ohlcv(
    ex,
    symbol: str,
    timeframe: str,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    limit: Optional[int] = 1000,
    sleep_ms: int = 400,
    max_rounds: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    all_rows: List[List[float]] = []
    total_fetched = 0
    loops = 0

    next_since = since_ms

    while True:
        loops += 1
        if loops > max_rounds:
            break

        request_limit = 1000
        if limit is not None:
            request_limit = min(request_limit, max(1, limit - total_fetched))
            if request_limit <= 0:
                break

        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=request_limit)

        if not data:
            break

        if until_ms is not None:
            data = [row for row in data if row and row[0] <= until_ms]
            if not data:
                break

        all_rows.extend(data)
        total_fetched += len(data)

        if limit is not None and total_fetched >= limit:
            break

        last_ts = data[-1][0]
        next_since = last_ts + 1

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        if len(data) < request_limit:
            break

    if not all_rows:
        if verbose:
            print(f"[{timeframe}] {symbol} fetched 0 bars")
        df_empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "symbol", "timeframe"])
        return df_empty

    # Build DataFrame
    df = pd.DataFrame(all_rows, columns=cols)
    df.rename(columns={"timestamp": "ts"}, inplace=True)
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    # Ensure types
    df["ts"] = df["ts"].astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    if verbose and not df.empty:
        print(f"[{timeframe}] {symbol} fetched {len(df)} bars, last_ts={int(df['ts'].max())}")

    return df


# ---------- Public API: batch fetchers ----------

def fetch_ohlcv_generic(
    symbols: Iterable[str],
    timeframe: str,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    limit: Optional[int] = 1000,
    exchange_id: str = "binance",
    exchange_kwargs: Optional[Dict[str, Any]] = None,
    sleep_ms: int = 400,
    verbose: bool = True,
) -> pd.DataFrame:

    exchange_kwargs = exchange_kwargs or {}
    ex = make_exchange(exchange_id, **exchange_kwargs)

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        try:
            df_sym = _fetch_symbol_ohlcv(
                ex,
                symbol=sym,
                timeframe=timeframe,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                sleep_ms=sleep_ms,
                verbose=verbose,
            )
            if not df_sym.empty:
                frames.append(df_sym)
        except Exception as e:
            if verbose:
                print(f"[{timeframe}] {sym} fetch error: {e}")

    if not frames:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "symbol", "timeframe"])

    out = pd.concat(frames, ignore_index=True)

    out.sort_values(by=["symbol", "ts"], inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


def fetch_ohlcv_1h(
    symbols: Iterable[str],
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    limit: Optional[int] = 1000,
    exchange_id: str = "binance",
    exchange_kwargs: Optional[Dict[str, Any]] = None,
    sleep_ms: int = 400,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:

    return fetch_ohlcv_generic(
        symbols=symbols,
        timeframe="1h",
        since_ms=since_ms,
        until_ms=until_ms,
        limit=limit,
        exchange_id=exchange_id,
        exchange_kwargs=exchange_kwargs,
        sleep_ms=sleep_ms,
        verbose=verbose,
    )


def fetch_ohlcv_5m(
    symbols: Iterable[str],
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    limit: Optional[int] = 1000,
    exchange_id: str = "binance",
    exchange_kwargs: Optional[Dict[str, Any]] = None,
    sleep_ms: int = 400,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:

    return fetch_ohlcv_generic(
        symbols=symbols,
        timeframe="5m",
        since_ms=since_ms,
        until_ms=until_ms,
        limit=limit,
        exchange_id=exchange_id,
        exchange_kwargs=exchange_kwargs,
        sleep_ms=sleep_ms,
        verbose=verbose,
    )


def fetch_ohlcv_batch(
    symbols: Iterable[str],
    timeframe: str = "1h",
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    limit: Optional[int] = 1000,
    exchange_id: str = "binance",
    exchange_kwargs: Optional[Dict[str, Any]] = None,
    sleep_ms: int = 400,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:

    tf = str(timeframe).lower()
    if tf in ("1h", "60m", "h1"):
        return fetch_ohlcv_1h(
            symbols=symbols,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            exchange_id=exchange_id,
            exchange_kwargs=exchange_kwargs,
            sleep_ms=sleep_ms,
            verbose=verbose,
            **kwargs,
        )
    if tf in ("5m", "m5"):
        return fetch_ohlcv_5m(
            symbols=symbols,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            exchange_id=exchange_id,
            exchange_kwargs=exchange_kwargs,
            sleep_ms=sleep_ms,
            verbose=verbose,
            **kwargs,
        )
    raise ValueError(f"Unsupported timeframe: {timeframe}. Use '1h' or '5m'.")