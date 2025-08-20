# src/btcusdt_algo/detectors/microtrigger.py
from __future__ import annotations
import pandas as pd
import numpy as np

EPS = 1e-9

def _to_num(s, default=np.nan):
    s = pd.to_numeric(s, errors="coerce")
    return s.fillna(default)

def _cooloff_mask(sig: pd.Series, bars: int) -> pd.Series:
    """
    True(시그널) 이후 bars개 봉 동안 추가 시그널을 억제.
    bars<=0 이면 원본 반환.
    """
    if bars <= 0:
        return sig.astype(bool)

    arr = sig.astype(bool).to_numpy(copy=True)
    n = arr.shape[0]
    block = 0
    for i in range(n):
        if block > 0:
            arr[i] = False
            block -= 1
        if arr[i]:
            block = int(bars)
    return pd.Series(arr, index=sig.index, dtype=bool)

def micro_triggers(
    df: pd.DataFrame,
    lookback_break: int = 5,
    rsi_on: float = 52.0,
    *,
    # --- 동작 모드/게이트 (기존 옵션 유지) ---
    mode: str = "pierce",             # "pierce"(고가/저가 관통) | "close"(종가 돌파)
    use_regime_gate: bool = True,     # regime=="trend" 만 활성 (엔진도 trend에서만 쓰지만 안전장치)
    regime_col: str = "regime",

    use_mtf_align: bool = False,      # rsi_5m/15m 정렬 강제
    mtf_rsi_min: float = 55.0,
    mtf_mode: str = "and",            # "and" | "or"

    use_ema_align: bool = False,      # EMA 정렬 강제 (롱: close>=ema21, 숏: close<=ema21)
    ema_col: str = "ema21",

    min_atr_pct: float = 0.0,         # ATR/close 하한 (0=미사용)
    max_atr_pct: float = 0.0,         # ATR/close 상한 (0=무한대)

    vol_spike_mult: float = 0.0,      # volume >= mult * vol_ma (0=미사용)
    vol_ma_len: int = 50,             # vol_ma 없을 때 롤링 길이

    darvas_confirm: bool = False,     # darvas_up/dn 플래그 동의 필요
    cooloff_bars: int = 0,            # 시그널 발생 후 N봉 억제

    # --- NEW: RSI 극단/교차 + 1·2차 조건(FOD/SOD) 옵션 ---
    rsi_extrema: dict | None = None,  # {enable: True, peak_thr:70, trough_thr:30, window:3, require_sod:True}
) -> pd.DataFrame:
    """
    1m 마이크로 트리거(추세 엔트리 타이밍 보조) + RSI 극단/교차 + FOD/SOD 극점.

    기본 규칙:
      - 돌파:   high >= prior_N_high  (mode="pierce")  |  close >= prior_N_high (mode="close")
      - 역돌파: low  <= prior_N_low   (mode="pierce")  |  close <= prior_N_low
      - RSI:    long: RSI >= rsi_on,  short: RSI <= 100-rsi_on

    선택 게이트:
      - Regime == "trend"
      - MTF RSI 정렬(rsi_5m/15m, and/or)
      - EMA 정렬(close vs ema21)
      - ATR% 범위(min/max)
      - 볼륨 스파이크(volume vs vol_ma)
      - 다르바스 확인(darvas_up/dn)
      - 시그널 쿨오프(bars)

    NEW (항상 컬럼 생성):
      - rsi_cross_70_down / rsi_cross_30_up
      - rsi_cross_50_up / rsi_cross_50_down
      - rsi_peak_ovb / rsi_trough_ovs (FOD/SOD 충족 극점)
      - rsi_peak_recent / rsi_trough_recent (최근 window 바 내 극점 발생여부)
        ↳ 엔진의 extreme RSI 하드게이트에서 사용 가능
    """
    n = int(max(2, lookback_break))
    out_index = df.index

    # ---------- 기본 시리즈 ----------
    high  = _to_num(df.get("high"),  np.nan)
    low   = _to_num(df.get("low"),   np.nan)
    close = _to_num(df.get("close"), np.nan)
    rsi   = _to_num(df.get("rsi"),   np.nan)

    # ---------- prior N 극값 (현재봉 제외: shift(1)) ----------
    prior_high = high.shift(1).rolling(n, min_periods=n).max()
    prior_low  = low.shift(1).rolling(n, min_periods=n).min()

    # ---------- 돌파 판정 ----------
    m = (mode or "pierce").lower()
    if m not in ("pierce", "close"):
        m = "pierce"
    if m == "pierce":
        brk_up   = (high >= prior_high)
        brk_down = (low  <= prior_low)
    else:  # "close"
        brk_up   = (close >= prior_high)
        brk_down = (close <= prior_low)

    # ---------- RSI 게이트 ----------
    rsi_long_ok  = (rsi >= float(rsi_on))
    rsi_short_ok = (rsi <= (100.0 - float(rsi_on)))

    # ---------- Regime 게이트 ----------
    if use_regime_gate and (regime_col in df.columns):
        reg = df[regime_col].astype(str).str.lower()
        regime_ok = reg.eq("trend")
    else:
        regime_ok = pd.Series(True, index=out_index)

    # ---------- MTF 정렬 ----------
    if use_mtf_align:
        r5  = _to_num(df.get("rsi_5m"),  np.nan)
        r15 = _to_num(df.get("rsi_15m"), np.nan)
        thr = float(mtf_rsi_min)

        long_5   = r5.ge(thr)                 if r5.notna().any()  else pd.Series(True, index=out_index)
        long_15  = r15.ge(thr)                if r15.notna().any() else pd.Series(True, index=out_index)
        short_5  = r5.le(100.0 - thr)         if r5.notna().any()  else pd.Series(True, index=out_index)
        short_15 = r15.le(100.0 - thr)        if r15.notna().any() else pd.Series(True, index=out_index)

        if (mtf_mode or "and").lower() == "or":
            mtf_long_ok  = long_5 | long_15
            mtf_short_ok = short_5 | short_15
        else:
            mtf_long_ok  = long_5 & long_15
            mtf_short_ok = short_5 & short_15
    else:
        mtf_long_ok  = pd.Series(True, index=out_index)
        mtf_short_ok = pd.Series(True, index=out_index)

    # ---------- EMA 정렬 ----------
    if use_ema_align and (ema_col in df.columns):
        ema = _to_num(df.get(ema_col), np.nan)
        ema_ok_long  = close.ge(ema)
        ema_ok_short = close.le(ema)
    else:
        ema_ok_long  = pd.Series(True, index=out_index)
        ema_ok_short = pd.Series(True, index=out_index)

    # ---------- ATR% 필터 ----------
    if (min_atr_pct or max_atr_pct) and ("atr" in df.columns):
        atr = _to_num(df.get("atr"), np.nan)
        atr_pct = (atr / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        lo = float(min_atr_pct) if min_atr_pct and min_atr_pct > 0 else 0.0
        hi = float(max_atr_pct) if max_atr_pct and max_atr_pct > 0 else np.inf
        atr_ok = atr_pct.ge(lo) & atr_pct.le(hi)
    else:
        atr_ok = pd.Series(True, index=out_index)

    # ---------- 볼륨 스파이크 ----------
    if vol_spike_mult and vol_spike_mult > 0:
        vol = _to_num(df.get("volume"), np.nan)
        if "vol_ma" in df.columns:
            vol_ma = _to_num(df.get("vol_ma"), np.nan)
        else:
            L = int(max(5, vol_ma_len))
            vol_ma = vol.rolling(L, min_periods=max(3, L//3)).mean()
        vol_ok = vol.ge(float(vol_spike_mult) * vol_ma.fillna(np.inf))
    else:
        vol_ok = pd.Series(True, index=out_index)

    # ---------- 다르바스 확인 ----------
    if darvas_confirm:
        up = df.get("darvas_up")
        dn = df.get("darvas_dn")
        darvas_ok_long  = up.fillna(False).astype(bool) if up is not None else pd.Series(True, index=out_index)
        darvas_ok_short = dn.fillna(False).astype(bool) if dn is not None else pd.Series(True, index=out_index)
    else:
        darvas_ok_long  = pd.Series(True, index=out_index)
        darvas_ok_short = pd.Series(True, index=out_index)

    # ---------- 종합(원시) ----------
    micro_long_raw  = (brk_up   & rsi_long_ok  & regime_ok & mtf_long_ok  & ema_ok_long  & atr_ok & vol_ok & darvas_ok_long)
    micro_short_raw = (brk_down & rsi_short_ok & regime_ok & mtf_short_ok & ema_ok_short & atr_ok & vol_ok & darvas_ok_short)

    # ---------- 쿨오프 적용 ----------
    micro_long  = _cooloff_mask(micro_long_raw,  int(max(0, cooloff_bars)))
    micro_short = _cooloff_mask(micro_short_raw, int(max(0, cooloff_bars)))

    # ======================================================
    # NEW: RSI 극단/교차 + FOD/SOD 극점 + 최근 N바 플래그
    # ======================================================
    rex = dict(rsi_extrema or {})
    peak_thr    = float(rex.get("peak_thr",   70.0))
    trough_thr  = float(rex.get("trough_thr", 30.0))
    ext_win     = int(max(1, rex.get("window", 3)))
    require_sod = bool(rex.get("require_sod", True))

    rsi_prev       = rsi.shift(1)
    rsi_diff       = rsi - rsi_prev           # 1차 차분(FOD)
    rsi_diff_prev  = rsi_diff.shift(1)
    rsi_sod        = (rsi_diff - rsi_diff_prev)  # 2차 차분(SOD) 근사

    # 단순 교차
    rsi_cross_70_down = ((rsi_prev >= peak_thr)   & (rsi < peak_thr)).fillna(False)
    rsi_cross_30_up   = ((rsi_prev <= trough_thr) & (rsi > trough_thr)).fillna(False)
    rsi_cross_50_up   = ((rsi_prev <= 50.0)       & (rsi > 50.0)).fillna(False)
    rsi_cross_50_down = ((rsi_prev >= 50.0)       & (rsi < 50.0)).fillna(False)

    # 극점(FOD/SOD)
    rsi_peak_ovb   = (rsi_diff_prev > 0) & (rsi_diff <= 0) & (rsi >= peak_thr)
    rsi_trough_ovs = (rsi_diff_prev < 0) & (rsi_diff >= 0) & (rsi <= trough_thr)
    if require_sod:
        rsi_peak_ovb   &= (rsi_sod < 0)
        rsi_trough_ovs &= (rsi_sod > 0)
    rsi_peak_ovb   = rsi_peak_ovb.fillna(False)
    rsi_trough_ovs = rsi_trough_ovs = rsi_trough_ovs.fillna(False)

    # 최근 N바 내 극점 발생 여부(엔진 게이트에서 사용)
    rsi_peak_recent   = rsi_peak_ovb.rolling(ext_win, min_periods=1).max().astype(bool)
    rsi_trough_recent = rsi_trough_ovs.rolling(ext_win, min_periods=1).max().astype(bool)

    # ---------- 반환 ----------
    out = pd.DataFrame({
        # 엔진이 직접 쓰는 플래그
        "micro_long":  micro_long.astype(bool),
        "micro_short": micro_short.astype(bool),

        # 디버깅/로깅용(원시 + 게이트 마스크들)
        "micro_long_raw":   micro_long_raw.astype(bool),
        "micro_short_raw":  micro_short_raw.astype(bool),
        "regime_ok":        regime_ok.astype(bool),
        "mtf_ok_long":      mtf_long_ok.astype(bool),
        "mtf_ok_short":     mtf_short_ok.astype(bool),
        "ema_ok_long":      ema_ok_long.astype(bool),
        "ema_ok_short":     ema_ok_short.astype(bool),
        "atr_ok":           atr_ok.astype(bool),
        "vol_ok":           vol_ok.astype(bool),
        "darvas_ok_long":   darvas_ok_long.astype(bool),
        "darvas_ok_short":  darvas_ok_short.astype(bool),

        # NEW: RSI 극단/교차 & FOD/SOD 극점
        "rsi_cross_70_down": rsi_cross_70_down.astype(bool),
        "rsi_cross_30_up":   rsi_cross_30_up.astype(bool),
        "rsi_cross_50_up":   rsi_cross_50_up.astype(bool),
        "rsi_cross_50_down": rsi_cross_50_down.astype(bool),
        "rsi_peak_ovb":      rsi_peak_ovb.astype(bool),
        "rsi_trough_ovs":    rsi_trough_ovs.astype(bool),
        "rsi_peak_recent":   rsi_peak_recent.astype(bool),
        "rsi_trough_recent": rsi_trough_recent.astype(bool),
    }, index=out_index)

    return out
