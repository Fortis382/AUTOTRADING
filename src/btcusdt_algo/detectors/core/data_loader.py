from typing import List, Optional
import pandas as pd
from loguru import logger
import ccxt
import time

def download_klines_ccxt(symbol: str = "BTC/USDT", timeframe: str = "1m", since: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    all_rows: List[dict] = []
    fetch_since = since
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        if not batch:
            break
        for ts, o, h, l, c, v in batch:
            all_rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        logger.info("Fetched {} rows", len(batch))
        if len(batch) < limit:
            break
        fetch_since = batch[-1][0] + 1
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def download_days_segmented(symbol: str = "BTC/USDT", timeframe: str = "1m", days: int = 180, step_days: int = 10) -> pd.DataFrame:
    now_ms = int(time.time()*1000)
    seg_ms = step_days*24*60*60*1000
    start_ms = now_ms - days*24*60*60*1000
    parts = []
    cursor = start_ms
    while cursor < now_ms:
        df = download_klines_ccxt(symbol, timeframe, since=cursor, limit=1000)
        if not df.empty:
            parts.append(df)
            last_ts = int(df['timestamp'].iloc[-1].value/1e6)  # ms
            cursor = last_ts + 1
        else:
            cursor += seg_ms
    if parts:
        out = pd.concat(parts, ignore_index=True)
        out = out.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return out
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
