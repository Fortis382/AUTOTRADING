# src/btcusdt_algo/core/regime.py
from __future__ import annotations
import pandas as pd
import numpy as np

EPS = 1e-8

def _safe_num(s, default=np.nan):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        # s가 없으면 index 맞춰 Series 생성
        idx = getattr(s, "index", None)
        return pd.Series(default, index=idx)

def _ensure_bb_width(df: pd.DataFrame) -> pd.Series:
    """
    bb_width( (high-low)/mid ) 를 최대한 보수적으로 복원.
    우선순위:
      1) df['bb_width']가 있으면 그것
      2) df['bb_high'],['bb_low'],['bb_mid']로 계산
      3) close의 단순 BB(20, 2.0)로 대체 계산
    """
    if "bb_width" in df.columns:
        w = _safe_num(df["bb_width"], 0.0)
        return w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if {"bb_high", "bb_low", "bb_mid"}.issubset(df.columns):
        hi = _safe_num(df["bb_high"], np.nan)
        lo = _safe_num(df["bb_low"], np.nan)
        mid = _safe_num(df["bb_mid"], np.nan)
        denom = mid.abs().where(mid.abs() > EPS, EPS)
        w = (hi - lo) / denom
        return w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 간단 볼밴 재계산(윈도우 20, 표준편차 2.0)
    close = _safe_num(df.get("close", pd.Series(np.nan, index=df.index)), np.nan)
    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std(ddof=0)
    std = std.replace(0, np.nan).bfill().ffill().fillna(1.0)
    hi = mid + 2.0 * std
    lo = mid - 2.0 * std
    denom = mid.abs().where(mid.abs() > EPS, EPS)
    w = (hi - lo) / denom
    return w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _ensure_slope_abs(
    df: pd.DataFrame,
    *,
    prefer_precomputed: bool = True,
    precomputed_col: str = "ema_slope",
    lookback: int = 5,
) -> pd.Series:
    """
    상대 EMA 기울기(|ΔEMA/EMA_prev|). precomputed가 있으면 활용.
    """
    if prefer_precomputed and precomputed_col in df.columns:
        s = _safe_num(df[precomputed_col], np.nan).abs()
        # 전부 NaN이면 재계산
        if s.notna().any():
            return s.fillna(0.0)

    # 폴백: ema21이 있으면 그것으로, 없으면 close로 근사
    base = df["ema21"] if "ema21" in df.columns else _safe_num(df.get("close", np.nan), np.nan)
    base = _safe_num(base, np.nan)
    prior = base.shift(int(max(1, lookback)))
    slope_rel = (base - prior) / (prior.abs().where(prior.abs() > EPS, np.nan))
    return slope_rel.replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)

def classify_regime(
    df: pd.DataFrame,
    trend_adx_min: int = 22,
    bb_squeeze_th: float = 0.028,
    range_bb_width_max: float | None = 0.00165,
    ema_slope_lookback: int = 5,
    ema_slope_min: float = 0.0007,
    # --- 안정화 옵션 ---
    use_hysteresis: bool = True,
    slope_hyst_ratio: float = 0.7,
    adx_hyst_delta: float = 2.0,
    bbw_hyst_ratio: float = 0.9,
    smooth_min_bars: int = 2,
    # --- 실무 편의 ---
    prefer_precomputed_slope: bool = True,
    precomputed_slope_col: str = "ema_slope",
) -> pd.Series:
    """
    Regime classifier: returns 'trend' | 'range'

    Trend if ANY:
      - |EMA21 slope| ≥ ema_slope_min
      - BB width > bb_squeeze_th
      - ADX ≥ trend_adx_min   (존재할 때만)

    Strong RANGE override (옵션):
      - |slope| < ema_slope_min AND
      - BB width ≤ range_bb_width_max AND
      - (ADX < trend_adx_min or NaN)

    Stabilizers:
      - Hysteresis: 직전이 trend면 range로 뒤집힐 때 더 엄격(기울기/ADX/BB폭 조건 동시)
      - Minimal run smoothing: 1~N바짜리 고립 전환 제거
    """
    idx = df.index

    # ---------- inputs ----------
    slope_abs = _ensure_slope_abs(
        df,
        prefer_precomputed=prefer_precomputed_slope,
        precomputed_col=precomputed_slope_col,
        lookback=ema_slope_lookback,
    )

    bb_width = _ensure_bb_width(df)
    adx = _safe_num(df.get("adx", pd.Series(np.nan, index=idx)), np.nan)

    # ---------- base decisions ----------
    cond_trend = (
        (slope_abs >= float(ema_slope_min)) |
        (bb_width > float(bb_squeeze_th)) |
        (adx >= float(trend_adx_min))
    )

    if range_bb_width_max is not None:
        cond_range_strong = (
            (slope_abs < float(ema_slope_min)) &
            (bb_width <= float(range_bb_width_max)) &
            ((adx < float(trend_adx_min)) | adx.isna())
        )
        base = np.where(cond_range_strong, "range", np.where(cond_trend, "trend", "range"))
    else:
        base = np.where(cond_trend, "trend", "range")

    arr = pd.Series(base, index=idx, dtype="object").to_numpy()

    # ---------- hysteresis (trend stickiness) ----------
    if use_hysteresis and len(arr) > 1:
        slope_exit = float(ema_slope_min) * float(slope_hyst_ratio)
        adx_exit   = max(0.0, float(trend_adx_min) - float(adx_hyst_delta))
        bbw_exit   = float(bb_squeeze_th) * float(bbw_hyst_ratio)

        s_abs = slope_abs.to_numpy()
        a = adx.to_numpy()
        bbw = bb_width.to_numpy()

        for i in range(1, len(arr)):
            if arr[i-1] == "trend" and arr[i] == "range":
                ok_flip = (s_abs[i] < slope_exit) and ( (np.isnan(a[i])) or (a[i] < adx_exit) ) and (bbw[i] <= bbw_exit)
                if not ok_flip:
                    arr[i] = "trend"

    # ---------- minimal run smoothing ----------
    # 1바 고립 제거
    if len(arr) >= 3:
        mid = arr[1:-1]
        prev = arr[:-2]
        nxt = arr[2:]
        mask = (mid != prev) & (mid != nxt)
        arr[1:-1][mask] = prev[mask]

    # 2바 고립까지 제거(옵션)
    if smooth_min_bars >= 2 and len(arr) >= 4:
        # 패턴: A B B A → BB를 A로
        for i in range(2, len(arr)):
            if arr[i-2] == arr[i] and arr[i-1] != arr[i]:
                arr[i-1] = arr[i]

    return pd.Series(arr, index=idx, dtype="category")
