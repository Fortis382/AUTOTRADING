# src/btcusdt_algo/detectors/darvas.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

EPS = 1e-12

def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def darvas_signals(
    df: pd.DataFrame,
    *,
    box_len: int = 20,
    confirm_bars: int = 2,
    min_range_pct: float = 0.003,   # 박스 높이 최소(가격의 0.3%) 필터
    atr_min_mult: float = 0.0,      # ATR 기반 최소 높이(예: 0.8 → 0.8*ATR)
    touch_eps_pct: float = 0.0005,  # 경계선 근접 허용(0.05%)
    return_box_levels: bool = False # 디버깅용 상단/하단 레벨 컬럼도 반환
) -> pd.DataFrame:
    """
    Darvas Box 기반 브레이크아웃 신호.

    아이디어(실전형 간소화):
      - 박스 상단 = 직전 N봉(high) 최고값, 하단 = 직전 N봉(low) 최저값 (현재봉 제외)
      - 박스 높이가 너무 얇으면 무시(%) + (선택) ATR 최소폭
      - 브레이크아웃은 '연속 confirm_bars개'의 종가가 경계 외부로 마감되어야 확정

    반환 컬럼(bool):
      - darvas_up, darvas_dn
    옵션:
      - return_box_levels=True이면 box_top/box_bottom/box_width도 같이 반환
    """
    if len(df) == 0:
        cols = ["darvas_up","darvas_dn"]
        if return_box_levels:
            cols += ["box_top","box_bottom","box_width"]
        return pd.DataFrame({c: [] for c in cols})

    box_len = int(max(2, box_len))
    confirm_bars = int(max(1, confirm_bars))

    high = _as_float(df.get("high"))
    low  = _as_float(df.get("low"))
    close= _as_float(df.get("close"))
    atr  = _as_float(df["atr"]) if "atr" in df.columns else pd.Series(np.nan, index=df.index)

    # 박스 상/하단: 현재봉 제외하려고 shift(1)
    box_top = high.shift(1).rolling(box_len, min_periods=box_len).max()
    box_bot = low.shift(1).rolling(box_len,  min_periods=box_len).min()

    # 유효 박스 높이(절대/상대)
    width = (box_top - box_bot).astype(float)
    # 최소 높이 절대값: max(비율 기반, ATR 기반)
    min_abs = (close.abs() * float(min_range_pct)).fillna(0.0)
    if np.isfinite(atr_min_mult) and atr_min_mult > 0:
        min_abs = np.maximum(min_abs, atr * float(atr_min_mult))
    valid_box = (width >= (min_abs.replace([np.inf, -np.inf], np.nan).fillna(0.0) + EPS))

    # 근접 허용치
    eps_top = box_top * float(touch_eps_pct)
    eps_bot = box_bot * float(touch_eps_pct)

    # 1차 브레이크아웃 판정(연속성 적용 전)
    prelim_up = (close >= (box_top - eps_top)) & valid_box
    prelim_dn = (close <= (box_bot + eps_bot)) & valid_box
    prelim_up = prelim_up.fillna(False).to_numpy()
    prelim_dn = prelim_dn.fillna(False).to_numpy()

    # 연속 confirm_bars개 종가가 경계 외부인 경우만 True
    n = len(df)
    up = np.zeros(n, dtype=bool)
    dn = np.zeros(n, dtype=bool)
    run_up = 0
    run_dn = 0
    for i in range(n):
        if prelim_up[i]:
            run_up += 1
        else:
            run_up = 0
        if prelim_dn[i]:
            run_dn += 1
        else:
            run_dn = 0
        if run_up >= confirm_bars:
            up[i] = True
        if run_dn >= confirm_bars:
            dn[i] = True

    out = pd.DataFrame({
        "darvas_up": up,
        "darvas_dn": dn,
    }, index=df.index)

    if return_box_levels:
        out["box_top"] = box_top
        out["box_bottom"] = box_bot
        out["box_width"] = width

    return out
