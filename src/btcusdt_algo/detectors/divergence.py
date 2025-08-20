# src/btcusdt_algo/detectors/divergence.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

EPS = 1e-12

def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _fractal_swings(high: pd.Series, low: pd.Series, wing: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    확정형 프랙탈 스윙 (repaint 방지):
      - swing high: 현재 high == centered rolling max
      - swing low : 현재 low  == centered rolling min
    """
    w = int(max(1, wing))
    win = 2 * w + 1
    sh = (high == high.rolling(win, center=True).max()).fillna(False)
    sl = (low  == low.rolling(win,  center=True).min()).fillna(False)
    return sh.astype(bool), sl.astype(bool)

def _mark_with_confirmation(n: int, signal_idx: int, direction: str,
                            close: pd.Series, low: pd.Series, high: pd.Series,
                            confirm_bars: int) -> int | None:
    """
    다이버전스가 pivot에서 발생했을 때, `confirm_bars` 이후에도 무효화되지 않았으면
    (bull: pivot low보다 더 낮은 저가가 나오지 않음, bear: pivot high보다 더 높은 고가가 나오지 않음)
    신호를 pivot+confirm_bars 위치에 표기. 무효화되면 None.
    """
    if confirm_bars <= 0:
        return signal_idx

    j = min(n - 1, signal_idx + confirm_bars)
    if direction == "bull":
        # 대기 구간에서 pivot low 갱신되면 무효화
        if low.iloc[signal_idx+1 : j+1].min(skipna=True) < low.iloc[signal_idx] - EPS:
            return None
        return j
    else:
        if high.iloc[signal_idx+1 : j+1].max(skipna=True) > high.iloc[signal_idx] + EPS:
            return None
        return j

def detect_divergence(
    df: pd.DataFrame,
    *,
    lookback_swings: int = 60,
    wing: int = 2,
    confirm_bars: int = 1,
    use_hidden: bool = False,
    rsi_col: str = "rsi",
    macd_hist_col: str = "macd_hist",
) -> pd.DataFrame:
    """
    스윙 피벗 간의 오실레이터 다이버전스 탐지 (재도장 없음).

    정의(클래식):
      - Bullish:   가격 LL (더 낮은 저점) vs 오실레이터 HL (더 높은 저점)
      - Bearish:   가격 HH (더 높은 고점) vs 오실레이터 LH (더 낮은 고점)

    옵션(히든):
      - Hidden Bull:  가격 HL vs 오실레이터 LL
      - Hidden Bear:  가격 LH vs 오실레이터 HH

    파라미터
      - lookback_swings: 최근 몇 바 내의 이전 스윙과 비교할지
      - wing: 프랙탈 좌우 날개 길이(2~3 권장)
      - confirm_bars: 신호 확정까지 기다리는 바 수(무효화 체크 포함)
      - rsi_col / macd_hist_col: 사용할 컬럼명

    반환 컬럼(bool)
      - bull_div_rsi, bear_div_rsi, bull_div_macd, bear_div_macd
      - (use_hidden=True면) hidden_bull_div_rsi, hidden_bear_div_rsi, hidden_bull_div_macd, hidden_bear_div_macd
    """
    n = len(df)
    if n == 0:
        cols = ["bull_div_rsi","bear_div_rsi","bull_div_macd","bear_div_macd"]
        if use_hidden:
            cols += ["hidden_bull_div_rsi","hidden_bear_div_rsi","hidden_bull_div_macd","hidden_bear_div_macd"]
        return pd.DataFrame({c: [] for c in cols})

    high = _as_float(df.get("high"))
    low  = _as_float(df.get("low"))
    close = _as_float(df.get("close"))
    rsi  = _as_float(df.get(rsi_col, pd.Series(np.nan, index=df.index)))
    macd = _as_float(df.get(macd_hist_col, pd.Series(np.nan, index=df.index)))

    sh_mask, sl_mask = _fractal_swings(high, low, wing=wing)
    sh_mask = sh_mask.values
    sl_mask = sl_mask.values

    bull_rsi = np.zeros(n, dtype=bool)
    bear_rsi = np.zeros(n, dtype=bool)
    bull_macd = np.zeros(n, dtype=bool)
    bear_macd = np.zeros(n, dtype=bool)

    h_bull_rsi = np.zeros(n, dtype=bool)
    h_bear_rsi = np.zeros(n, dtype=bool)
    h_bull_macd = np.zeros(n, dtype=bool)
    h_bear_macd = np.zeros(n, dtype=bool)

    # 최신 스윙 인덱스 유지
    last_low_idx  = -10**9
    prev_low_idx  = -10**9
    last_high_idx = -10**9
    prev_high_idx = -10**9

    look = int(max(2, lookback_swings))

    for i in range(n):
        # 스윙 업데이트
        if sl_mask[i]:
            prev_low_idx, last_low_idx = last_low_idx, i
        if sh_mask[i]:
            prev_high_idx, last_high_idx = last_high_idx, i

        # --- Low (클래식 Bull / Hidden Bull)
        if sl_mask[i] and prev_low_idx > -10**8:
            j, k = prev_low_idx, last_low_idx  # j < k == i
            if (k - j) <= look:
                # 두 저점 값
                p_j, p_k = low.iat[j], low.iat[k]
                r_j, r_k = rsi.iat[j], rsi.iat[k]
                m_j, m_k = macd.iat[j], macd.iat[k]

                # Bullish (LL in price, HL in oscillator)
                if np.isfinite(p_j) and np.isfinite(p_k) and (p_k < p_j - EPS):
                    # RSI
                    if np.isfinite(r_j) and np.isfinite(r_k) and (r_k > r_j + EPS):
                        idx = _mark_with_confirmation(n, k, "bull", close, low, high, confirm_bars)
                        if idx is not None: bull_rsi[idx] = True
                    # MACD hist
                    if np.isfinite(m_j) and np.isfinite(m_k) and (m_k > m_j + EPS):
                        idx = _mark_with_confirmation(n, k, "bull", close, low, high, confirm_bars)
                        if idx is not None: bull_macd[idx] = True

                if use_hidden:
                    # Hidden Bull (HL in price, LL in oscillator)
                    if np.isfinite(p_j) and np.isfinite(p_k) and (p_k > p_j + EPS):
                        if np.isfinite(r_j) and np.isfinite(r_k) and (r_k < r_j - EPS):
                            idx = _mark_with_confirmation(n, k, "bull", close, low, high, confirm_bars)
                            if idx is not None: h_bull_rsi[idx] = True
                        if np.isfinite(m_j) and np.isfinite(m_k) and (m_k < m_j - EPS):
                            idx = _mark_with_confirmation(n, k, "bull", close, low, high, confirm_bars)
                            if idx is not None: h_bull_macd[idx] = True

        # --- High (클래식 Bear / Hidden Bear)
        if sh_mask[i] and prev_high_idx > -10**8:
            j, k = prev_high_idx, last_high_idx
            if (k - j) <= look:
                p_j, p_k = high.iat[j], high.iat[k]
                r_j, r_k = rsi.iat[j], rsi.iat[k]
                m_j, m_k = macd.iat[j], macd.iat[k]

                # Bearish (HH in price, LH in oscillator)
                if np.isfinite(p_j) and np.isfinite(p_k) and (p_k > p_j + EPS):
                    if np.isfinite(r_j) and np.isfinite(r_k) and (r_k < r_j - EPS):
                        idx = _mark_with_confirmation(n, k, "bear", close, low, high, confirm_bars)
                        if idx is not None: bear_rsi[idx] = True
                    if np.isfinite(m_j) and np.isfinite(m_k) and (m_k < m_j - EPS):
                        idx = _mark_with_confirmation(n, k, "bear", close, low, high, confirm_bars)
                        if idx is not None: bear_macd[idx] = True

                if use_hidden:
                    # Hidden Bear (LH in price, HH in oscillator)
                    if np.isfinite(p_j) and np.isfinite(p_k) and (p_k < p_j - EPS):
                        if np.isfinite(r_j) and np.isfinite(r_k) and (r_k > r_j + EPS):
                            idx = _mark_with_confirmation(n, k, "bear", close, low, high, confirm_bars)
                            if idx is not None: h_bear_rsi[idx] = True
                        if np.isfinite(m_j) and np.isfinite(m_k) and (m_k > m_j + EPS):
                            idx = _mark_with_confirmation(n, k, "bear", close, low, high, confirm_bars)
                            if idx is not None: h_bear_macd[idx] = True

    cols = {
        "bull_div_rsi": bull_rsi,
        "bear_div_rsi": bear_rsi,
        "bull_div_macd": bull_macd,
        "bear_div_macd": bear_macd,
    }
    if use_hidden:
        cols.update({
            "hidden_bull_div_rsi": h_bull_rsi,
            "hidden_bear_div_rsi": h_bear_rsi,
            "hidden_bull_div_macd": h_bull_macd,
            "hidden_bear_div_macd": h_bear_macd,
        })
    return pd.DataFrame(cols, index=df.index)
