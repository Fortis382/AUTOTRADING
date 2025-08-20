# src/btcusdt_algo/detectors/fib.py
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd

EPS = 1e-9

def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _fractal_swings(high: pd.Series, low: pd.Series, wing: int = 2) -> tuple[pd.Series, pd.Series]:
    """확정형 프랙탈 스윙: 가운데가 좌우 wing 포함 창의 극값이면 True."""
    w = int(max(1, wing))
    win = 2 * w + 1
    sh = (high == high.rolling(win, center=True).max()).fillna(False)
    sl = (low  == low.rolling(win,  center=True).min()).fillna(False)
    return sh.astype(bool), sl.astype(bool)

def fib_proximity(
    df: pd.DataFrame,
    lookback: int = 120,
    eps_pct: float = 0.15,           # 주 레벨(0.382/0.5/0.618)은 구간 길이 * eps_pct 로 판정
    wing: int = 2,
    # --- 추가(허점) 레벨 관련 설정 ---
    include_extra: bool = True,
    extra_retracements: Iterable[float] = (0.786,),   # 예: (0.786, 0.707, 0.236)
    extra_extensions:   Iterable[float] = (),         # 예: (2.0,)
    extra_level_eps_pct: float = 0.06,                # 레벨 자체의 ±비율(0.05~0.07 권장)
    atr_floor_mult: float = 0.25,                     # 최소 허용폭: max( tol, atr*이값 )
) -> pd.DataFrame:
    """
    최근 스윙(lo<->hi) 구간으로 피보나치 근접 여부 플래그 생성.

    반환:
      - 기본: near_fib_382, near_fib_500, near_fib_618, near_ext_127, near_ext_161 (bool)
      - 추가(옵션): near_fib_786, near_fib_707, near_fib_236, near_ext_200 등 (bool)
    """
    if len(df) == 0:
        return pd.DataFrame(index=df.index)

    high = _as_float(df.get("high"))
    low  = _as_float(df.get("low"))
    close= _as_float(df.get("close"))
    atr  = _as_float(df.get("atr")) if "atr" in df.columns else pd.Series(np.nan, index=df.index)

    n = len(df)
    sh_mask, sl_mask = _fractal_swings(high, low, wing=wing)
    sh_mask = sh_mask.values; sl_mask = sl_mask.values

    out = {
        "near_fib_382": np.zeros(n, dtype=bool),
        "near_fib_500": np.zeros(n, dtype=bool),
        "near_fib_618": np.zeros(n, dtype=bool),
        "near_ext_127": np.zeros(n, dtype=bool),
        "near_ext_161": np.zeros(n, dtype=bool),
    }

    # extra 레벨 준비
    extra_retr = tuple(float(x) for x in (extra_retracements or ()))
    extra_ext  = tuple(float(x) for x in (extra_extensions or ()))
    for r in extra_retr:
        out[f"near_fib_{int(round(r*1000)):03d}"] = np.zeros(n, dtype=bool)  # 0.786 -> near_fib_786
    for e in extra_ext:
        out[f"near_ext_{int(round(e*100)):03d}"] = np.zeros(n, dtype=bool)   # 2.0 -> near_ext_200

    last_high_idx = -10**9; last_high = np.nan
    last_low_idx  = -10**9; last_low  = np.nan

    look = int(max(2, lookback))
    eps_pct = float(max(0.0, eps_pct))
    extra_level_eps_pct = float(max(0.0, extra_level_eps_pct))
    atr_floor_mult = float(max(0.0, atr_floor_mult))

    for i in range(n):
        # 최신 스윙 갱신
        if sh_mask[i] and np.isfinite(high.iat[i]):
            last_high_idx = i; last_high = float(high.iat[i])
        if sl_mask[i] and np.isfinite(low.iat[i]):
            last_low_idx  = i; last_low  = float(low.iat[i])

        # 최근 lookback 내 유효 스윙쌍 없으면 skip
        if (i - last_high_idx) > look or (i - last_low_idx) > look:
            continue
        if not (np.isfinite(last_high) and np.isfinite(last_low)):
            continue
        if last_high_idx == last_low_idx:
            continue

        # 현재 방향과 구간
        if last_low_idx < last_high_idx:
            # up leg: low -> high
            lo, hi = last_low, last_high
            rng = hi - lo
            if rng <= EPS: continue
            # 기본 되돌림
            lv = {
                "near_fib_382": hi - 0.382 * rng,
                "near_fib_500": hi - 0.500 * rng,
                "near_fib_618": hi - 0.618 * rng,
            }
            # 확장
            ex = {
                "near_ext_127": lo + 1.272 * rng,
                "near_ext_161": lo + 1.618 * rng,
            }
            # extra retr
            lv_extra = { f"near_fib_{int(round(r*1000)):03d}": hi - float(r) * rng for r in extra_retr }
            # extra ext
            ex_extra = { f"near_ext_{int(round(e*100)):03d}": lo + float(e) * rng for e in extra_ext }
        else:
            # down leg: high -> low
            hi, lo = last_high, last_low
            rng = hi - lo
            if rng <= EPS: continue
            # 기본 되돌림
            lv = {
                "near_fib_382": lo + 0.382 * rng,
                "near_fib_500": lo + 0.500 * rng,
                "near_fib_618": lo + 0.618 * rng,
            }
            # 확장
            ex = {
                "near_ext_127": hi - 1.272 * rng,
                "near_ext_161": hi - 1.618 * rng,
            }
            # extra retr
            lv_extra = { f"near_fib_{int(round(r*1000)):03d}": lo + float(r) * rng for r in extra_retr }
            # extra ext
            ex_extra = { f"near_ext_{int(round(e*100)):03d}": hi - float(e) * rng for e in extra_ext }

        # 허용오차
        tol_base = eps_pct * rng
        atr_i = float(atr.iat[i]) if i < len(atr) and np.isfinite(atr.iat[i]) else np.nan
        if np.isfinite(atr_i):
            tol_base = max(tol_base, atr_floor_mult * atr_i)

        px = float(close.iat[i]) if np.isfinite(close.iat[i]) else np.nan
        if not np.isfinite(px):
            continue

        # 기본 레벨 판정 (구간 기반 tol)
        for k, lvpx in lv.items():
            out[k][i] = abs(px - lvpx) <= tol_base
        for k, lvpx in ex.items():
            out[k][i] = abs(px - lvpx) <= tol_base

        # extra 레벨 판정 (레벨 자체의 ±% tol)
        if include_extra and (extra_retr or extra_ext):
            for k, lvpx in {**lv_extra, **ex_extra}.items():
                tol_extra = extra_level_eps_pct * abs(lvpx)
                if np.isfinite(atr_i):
                    tol_extra = max(tol_extra, atr_floor_mult * atr_i)
                out[k][i] = abs(px - lvpx) <= tol_extra

    return pd.DataFrame({k: v for k, v in out.items()}, index=df.index)
