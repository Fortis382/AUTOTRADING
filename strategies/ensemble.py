# src/btcusdt_algo/strategies/ensemble.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

EPS = 1e-12

def _f(x, default=np.nan) -> float:
    """robust float extractor for Series/Scalar/None"""
    try:
        if isinstance(x, pd.Series):
            x = x.iloc[-1]
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _b(x) -> bool:
    """bool-ish with NaN safe"""
    try:
        if isinstance(x, pd.Series):
            x = x.iloc[-1]
        return bool(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else False
    except Exception:
        return False

def _band_pos(close: float, bb_low: float, bb_high: float) -> float:
    """position inside BB [0..1], NaN-safe"""
    if not (np.isfinite(close) and np.isfinite(bb_low) and np.isfinite(bb_high)):
        return np.nan
    span = bb_high - bb_low
    if not np.isfinite(span) or abs(span) < EPS:
        return np.nan
    return float((close - bb_low) / span)

# ---------------------------------------------------------------------
# Base Strategy Interface
# ---------------------------------------------------------------------
@dataclass
class BaseStrategy:
    name: str = "BaseStrategy"

    def should_long(self, row: pd.Series) -> bool:
        return False

    def should_short(self, row: pd.Series) -> bool:
        return False

# ---------------------------------------------------------------------
# Trend Strategy
# ---------------------------------------------------------------------
@dataclass
class TrendStrategy(BaseStrategy):
    """
    방향 배정만 깔끔하게:
      - 핵심 정렬: close vs ema21, MACD hist sign
      - 보조: RSI(1m/5m/15m) 정렬, Darvas/다이버전스 보너스·패널티
    실제 엔트리 타이밍은 엔진의 마이크로 트리거/게이트가 보장.
    """
    name: str = "TrendStrategy"

    # 임곗값(엔진 게이트와 과도하게 겹치지 않도록 완만하게)
    rsi_neutral: float = 50.0
    rsi_bias: float = 55.0

    def _bias_score(self, row: pd.Series) -> float:
        score = 0.0

        c = _f(row.get("close"), np.nan)
        e = _f(row.get("ema21"), np.nan)
        mh = _f(row.get("macd_hist"), np.nan)
        r = _f(row.get("rsi"), np.nan)
        r5 = _f(row.get("rsi_5m"), np.nan)
        r15 = _f(row.get("rsi_15m"), np.nan)

        # 1) 방향 정렬(강)
        if np.isfinite(c) and np.isfinite(e):
            score += 1.2 if c >= e else -1.2
        if np.isfinite(mh):
            score += 1.2 if mh >= 0 else -1.2

        # 2) RSI(중)
        if np.isfinite(r):
            score += 0.8 if r >= self.rsi_bias else (-0.8 if r <= (100 - self.rsi_bias) else 0.0)

        # 3) MTF 정렬(약)
        if np.isfinite(r5):
            score += 0.4 if ((r >= self.rsi_neutral and r5 >= self.rsi_neutral) or
                             (r <  self.rsi_neutral and r5 <  self.rsi_neutral)) else -0.2
        if np.isfinite(r15):
            score += 0.4 if ((r >= self.rsi_neutral and r15 >= self.rsi_neutral) or
                             (r <  self.rsi_neutral and r15 <  self.rsi_neutral)) else -0.2

        # 4) 패턴/디텍터 보너스(아주 약)
        if _b(row.get("darvas_up")):   score += 0.4
        if _b(row.get("darvas_dn")):   score -= 0.4
        if _b(row.get("bull_div_rsi")):  score += 0.3
        if _b(row.get("bear_div_rsi")):  score -= 0.3
        if _b(row.get("bull_div_macd")): score += 0.2
        if _b(row.get("bear_div_macd")): score -= 0.2

        return float(score)

    def should_long(self, row: pd.Series) -> bool:
        s = self._bias_score(row)
        # 충분한 롱 성향
        return s >= 1.2

    def should_short(self, row: pd.Series) -> bool:
        s = self._bias_score(row)
        # 충분한 숏 성향
        return s <= -1.2

# ---------------------------------------------------------------------
# Range Strategy
# ---------------------------------------------------------------------
@dataclass
class RangeStrategy(BaseStrategy):
    """
    박스/밴드 안에서 리버전:
      - 하단 근처 + 저RSI → long
      - 상단 근처 + 고RSI → short
    엔진은 range 모드일 때도 추가로 자체 규칙(밴드 엣지+RSI)로 마이크로 확인함.
    """
    name: str = "RangeStrategy"

    edge_lo: float = 0.25
    edge_hi: float = 0.75
    rsi_long_max: float = 45.0
    rsi_short_min: float = 55.0

    def should_long(self, row: pd.Series) -> bool:
        c  = _f(row.get("close"), np.nan)
        bl = _f(row.get("bb_low"), np.nan)
        bh = _f(row.get("bb_high"), np.nan)
        r  = _f(row.get("rsi"), np.nan)

        pos = _band_pos(c, bl, bh)
        if not np.isfinite(pos) or not np.isfinite(r):
            return False
        # 하단 에지 + 낮은 RSI
        if pos <= self.edge_lo and r <= self.rsi_long_max:
            # 과도 확장(near_ext_161 등)시 살짝 보수적
            if _b(row.get("near_ext_161")) or _b(row.get("near_ext_127")):
                return r <= (self.rsi_long_max - 2.0)
            return True
        return False

    def should_short(self, row: pd.Series) -> bool:
        c  = _f(row.get("close"), np.nan)
        bl = _f(row.get("bb_low"), np.nan)
        bh = _f(row.get("bb_high"), np.nan)
        r  = _f(row.get("rsi"), np.nan)

        pos = _band_pos(c, bl, bh)
        if not np.isfinite(pos) or not np.isfinite(r):
            return False
        # 상단 에지 + 높은 RSI
        if pos >= self.edge_hi and r >= self.rsi_short_min:
            if _b(row.get("near_ext_161")) or _b(row.get("near_ext_127")):
                return r >= (self.rsi_short_min + 2.0)
            return True
        return False

# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------
# 싱글톤 인스턴스(불필요한 객체 생성 방지)
_TREND = TrendStrategy()
_RANGE = RangeStrategy()

def route_by_regime(regime: str | None) -> BaseStrategy:
    """
    엔진에서 row['regime']을 넘겨 호출.
    - 'trend' → TrendStrategy
    - 그 외 → RangeStrategy
    """
    rg = (regime or "").lower()
    if rg == "trend":
        return _TREND
    return _RANGE
