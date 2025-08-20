# src/btcusdt_algo/core/mtf.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Iterable

# ---------------------------
# small, self-contained TA
# ---------------------------
def _ema(close: pd.Series, length: int) -> pd.Series:
    L = max(1, int(length))
    return pd.Series(close, copy=False).ewm(span=L, adjust=False).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    L = max(1, int(length))
    c = pd.Series(close, copy=False)
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1 / L, adjust=False).mean()
    ad = dn.ewm(alpha=1 / L, adjust=False).mean()
    rs = au / (ad + 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out

# ---------------------------
# utils
# ---------------------------
def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp 컬럼 존재/UTC/정렬 보장"""
    if "timestamp" not in df.columns:
        raise KeyError("DataFrame must contain 'timestamp' column")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.sort_values("timestamp").reset_index(drop=True)

def _ind_cfg(settings: Dict) -> Tuple[int, int]:
    """
    엔진 호환성 유지: rsi/ema 길이만 반환 (정확히 2개!)
    """
    ind = (settings or {}).get("indicators", {}) or {}
    rsi_len = int(ind.get("rsi_length", ind.get("rsi_period", 14)))
    ema_len = int(ind.get("ema_length", ind.get("ema21_period", 21)))
    return rsi_len, ema_len

def _freq_alias(freq: str) -> str:
    """
    1m/5m/15m 같은 별칭을 pandas 유효 alias로 변환.
    'm'(minute) 경고 회피 위해 'min'으로 변환.
    """
    f = str(freq).lower().strip()
    # 가장 흔한 표현만 엄격 처리
    map_ = {"1m": "1min", "5m": "5min", "15m": "15min"}
    if f in map_:
        return map_[f]
    # 이미 'min' 쓰면 그대로, 'm'로 끝나면 'min' 대체
    if f.endswith("m") and not f.endswith("min"):
        return f + "in"  # '5m' → '5min' 은 위에서 처리, 여기선 드물게 '30m' 같은 경우
    return f

def _suffix_from_freq(freq: str) -> str:
    """표기 통일: '5min' → '5m', '15min' → '15m'"""
    f = _freq_alias(freq)
    if f.endswith("min"):
        return f.replace("min", "m")
    return f

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    열이 없으면 만들고(NaN), 중복(동일 이름 다수)일 경우 마지막 열을 택해 1-D Series로 강제 변환.
    그다음 to_numeric(coerce)로 숫자화.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
            continue
        obj = df[c]
        # 중복 컬럼이면 DataFrame이 들어올 수 있음 → 마지막 열 선택
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, -1]
        df[c] = pd.to_numeric(obj, errors="coerce")
    return df
# ---------------------------
# public API
# ---------------------------
def resample_with_indicators(
    df: pd.DataFrame,
    freq: str,
    settings: dict,
    cols: Tuple[str, ...] = ("rsi", "ema21"),
) -> pd.DataFrame:
    """
    1분봉 df를 freq('5min','15min' 등)으로 OHLCV 리샘플 → 요청 컬럼만 계산.
    반환: ['timestamp','close'] + requested cols

    - 항상 timestamp/close 포함 (머지/진단용)
    - RSI/EMA 길이는 settings['indicators']에서 읽음(하위키 호환)
    """
    base = _ensure_ts(df)
    pfreq = _freq_alias(freq)

    # 1) OHLCV 리샘플
    ohlcv = (
        base.set_index("timestamp")
        .resample(pfreq)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )

    want_rsi = "rsi" in cols
    want_ema = "ema21" in cols
    rsi_len, ema_len = _ind_cfg(settings)

    out = ohlcv[["timestamp", "close"]].copy()
    if want_rsi:
        out["rsi"] = _rsi(out["close"], rsi_len)
    if want_ema:
        out["ema21"] = _ema(out["close"], ema_len)

    # numeric 위생
    _safe_numeric(out, ["close", "rsi", "ema21"])

    keep = ["timestamp", "close"] + [c for c in ("rsi", "ema21") if c in out.columns]
    return out[keep].reset_index(drop=True)

def merge_mtf(base: pd.DataFrame, higher: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    1분 타임라인(base)에 상위TF(higher) 지표를 backward asof 머지.
    rsi_5m, ema21_5m 같은 *_suffix 컬럼을 보장.
    """
    b = _ensure_ts(base)
    h = _ensure_ts(higher)

    merged = pd.merge_asof(
        b, h, on="timestamp", direction="backward", suffixes=("", f"_{suffix}")
    )

    # 지표 이름 보정 (혹시 suffix가 자동으로 안 붙은 경우 대비)
    for c in ("rsi", "ema21", "close", "macd_hist", "adx", "ema21_slope", "atr"):
        dst = f"{c}_{suffix}"
        if dst not in merged.columns and c in merged.columns:
            merged[dst] = merged[c]

    # 숫자형 강제 + 중복 이름 방어
    _safe_numeric(merged, [f"{c}_{suffix}" for c in ("rsi", "ema21", "close")])
    return merged

# 편의 함수: 여러 TF를 한 번에 붙이고 싶을 때(엔진이 굳이 쓸 필요는 없음)
def add_mtf(
    df: pd.DataFrame,
    settings: dict,
    freqs: Tuple[str, ...] = ("5min", "15min"),
    cols: Tuple[str, ...] = ("rsi", "ema21"),
) -> pd.DataFrame:
    out = _ensure_ts(df)
    for f in freqs:
        ht = resample_with_indicators(out, f, settings, cols=cols)
        out = merge_mtf(out, ht, _suffix_from_freq(f))
    return out
