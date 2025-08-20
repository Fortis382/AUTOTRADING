# src/btcusdt_algo/core/scoring.py
from __future__ import annotations
from typing import Dict, Any, Optional
import math
import numpy as np
import pandas as pd

EPS = 1e-12

def _f(x, default: float = np.nan) -> float:
    try:
        if isinstance(x, pd.Series):
            x = x.iloc[-1]
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _exists(x) -> bool:
    try:
        v = _f(x)
        return np.isfinite(v)
    except Exception:
        return False

class SignalScorer:
    """
    합성 스코어(0~100)
      - 5 컴포넌트(변동성/모멘텀/볼륨/구조/FIB)
      - 레짐별 가중 리밸런싱
      - 컨텍스트 보너스/패널티(캡)
      - (옵션) 로지스틱 캡
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        squeeze_th: float = 0.02,
        bonuses: Optional[Dict[str, float]] = None,
        max_bonus: float = 14.0,
        max_penalty: float = 14.0,
        # ↓ 기본을 False로 바꿔 진입 스코어가 과도하게 눌리는 문제 방지
        logistic_cap: bool = False,
        logistic_mid: float = 60.0,
        logistic_k: float = 12.0,
    ):
        # 1) 가중치 정규화
        default_w = {"volatility":0.25, "momentum":0.25, "volume":0.20, "structure":0.20, "fib":0.10}
        w_in = dict(weights or {})
        w_clean: Dict[str, float] = {}
        for k, dv in default_w.items():
            v = _f(w_in.get(k, dv), dv)
            v = 0.0 if not np.isfinite(v) or v < 0 else float(v)
            w_clean[k] = v
        s = sum(w_clean.values()) or 1.0
        self.weights = {k: (v / s) for k, v in w_clean.items()}

        self.squeeze_th = float(squeeze_th)

        # 2) 디텍터 보너스 기본값
        det_default = {
            "micro_long": 2.0, "micro_short": 2.0,
            "bull_div_rsi": 3.0, "bear_div_rsi": 3.0,
            "bull_div_macd": 2.0, "bear_div_macd": 2.0,
            "darvas_up": 4.0, "darvas_dn": 4.0,
            "near_fib_382": 1.0, "near_fib_500": 0.8, "near_fib_618": 1.0,
            "near_fib_707": 0.7, "near_fib_786": 0.8,
            "near_ext_127": 0.8, "near_ext_161": 1.0,
        }
        self.bonuses = det_default
        if bonuses:
            for k, v in bonuses.items():
                self.bonuses[str(k)] = _f(v, det_default.get(k, 0.0))

        self.max_bonus = float(max_bonus)
        self.max_penalty = float(max_penalty)
        self.logistic_cap = bool(logistic_cap)
        self.logistic_mid = float(logistic_mid)
        self.logistic_k = float(logistic_k)

    # ---------- components in [0,1] ----------
    def _volatility(self, row: pd.Series) -> float:
        vs = _f(row.get("volatility_score"), np.nan)
        if np.isfinite(vs):
            return float(np.clip(vs/100.0, 0.0, 1.0))
        bbw = _f(row.get("bb_width"), np.nan)
        if not np.isfinite(bbw): return 0.0
        lo, hi = self.squeeze_th * 0.7, self.squeeze_th * 1.4
        if bbw <= lo: return 0.0
        if bbw >= hi: return 1.0
        return float((bbw - lo) / (hi - lo))

    def _momentum(self, row: pd.Series) -> float:
        ms = _f(row.get("momentum_score"), np.nan)
        if np.isfinite(ms):
            return float(np.clip(ms/100.0, 0.0, 1.0))
        r = _f(row.get("rsi"), 50.0)
        core = min(1.0, abs(r - 50.0) / 15.0)  # ±15 → 1.0
        r5  = _f(row.get("rsi_5m"), np.nan)
        r15 = _f(row.get("rsi_15m"), np.nan)
        align = 0.0
        if np.isfinite(r5):
            same = (r >= 50 and r5 >= 50) or (r < 50 and r5 < 50)
            align += 0.1 if same else -0.06
        if np.isfinite(r15):
            same = (r >= 50 and r15 >= 50) or (r < 50 and r15 < 50)
            align += 0.1 if same else -0.06
        return float(np.clip(core + align, 0.0, 1.0))

    def _volume(self, row: pd.Series) -> float:
        vs = _f(row.get("volume_score"), np.nan)
        if np.isfinite(vs):
            return float(np.clip(vs/100.0, 0.0, 1.0))
        v = _f(row.get("volume"), np.nan); vma = _f(row.get("vol_ma"), np.nan)
        if not (np.isfinite(v) and np.isfinite(vma) and vma > 0): return 0.0
        return float(np.clip((v/vma - 1.0) / 1.5, 0.0, 1.0))  # 1.0→0, 2.5x→1.0

    def _structure(self, row: pd.Series) -> float:
        ss = _f(row.get("structure_score"), np.nan)
        if np.isfinite(ss):
            return float(np.clip(ss/100.0, 0.0, 1.0))
        c = _f(row.get("close")); mid = _f(row.get("bb_mid")); ema = _f(row.get("ema21"), mid)
        if not (np.isfinite(c) and np.isfinite(mid) and np.isfinite(ema)): return 0.5
        sgn_mid = np.sign(c - mid); sgn_ema = np.sign(c - ema)
        aligned = 1.0 if (sgn_mid == sgn_ema and sgn_mid != 0.0) else 0.6
        bbw = _f(row.get("bb_width"), np.nan)
        if np.isfinite(bbw) and bbw < self.squeeze_th*0.6:
            aligned = max(0.3, aligned - 0.2)
        return float(aligned)

    def _fib(self, row: pd.Series) -> float:
        fs = _f(row.get("fib_score"), np.nan)
        if np.isfinite(fs):
            return float(np.clip(fs/100.0, 0.0, 1.0))
        if bool(row.get("near_fib_618", False)): return 1.0
        if bool(row.get("near_fib_786", False)) or bool(row.get("near_fib_707", False)): return 0.85
        if bool(row.get("near_fib_382", False)) or bool(row.get("near_fib_500", False)): return 0.7
        if bool(row.get("near_ext_161", False)) or bool(row.get("near_ext_127", False)): return 0.8
        return 0.5

    def _detector_bonus(self, row: pd.Series) -> float:
        b = 0.0
        for k, pts in self.bonuses.items():
            try:
                if bool(row.get(k, False)):
                    b += float(pts)
            except Exception:
                continue
        return float(min(b, self.max_bonus))

    def _regime_rebalance(self, regime: str) -> Dict[str, float]:
        w = dict(self.weights)
        rg = (regime or "").lower()
        if rg == "trend":
            w["momentum"]   *= 1.30
            w["fib"]        *= 1.15
            w["volatility"] *= 1.10
            w["structure"]  *= 0.90
        elif rg == "range":
            w["structure"]  *= 1.35
            w["volatility"] *= 1.10
            w["momentum"]   *= 0.92
            w["fib"]        *= 0.92
        s = sum(w.values()) or 1.0
        for k in w: w[k] /= s
        return w

    # ---------- public ----------
    def score_row(self, row: pd.Series, session_mult: float = 1.0) -> float:
        regime = str(row.get("regime", "")).lower()

        # 컴포넌트
        v  = self._volatility(row)
        m  = self._momentum(row)
        vo = self._volume(row)
        st = self._structure(row)
        fb = self._fib(row)

        # 레짐 가중 재조정
        w = self._regime_rebalance(regime)
        base01 = (w["volatility"]*v + w["momentum"]*m + w["volume"]*vo + w["structure"]*st + w["fib"]*fb)
        base01 = float(np.clip(base01, 0.0, 1.0))

        # 보너스/패널티
        bonus = 0.0; penalty = 0.0

        # squeeze 초입 + EMA/RSI 정렬 약보너스
        bbw = _f(row.get("bb_width"), np.nan)
        if np.isfinite(bbw) and bbw <= self.squeeze_th:
            bonus += 2.0
            rsi = _f(row.get("rsi"), np.nan)
            if np.isfinite(rsi) and rsi >= 56.0: bonus += 0.8
            c = _f(row.get("close")); e = _f(row.get("ema21"), np.nan)
            if np.isfinite(c) and np.isfinite(e) and c >= e: bonus += 0.8

        # MACD 부호 vs EMA 방향 상충(추세에서만)
        mh = _f(row.get("macd_hist"), np.nan)
        c = _f(row.get("close"), np.nan); e = _f(row.get("ema21"), np.nan)
        if regime == "trend" and np.isfinite(mh) and np.isfinite(c) and np.isfinite(e):
            dir_up = c >= e
            if (dir_up and mh < 0) or ((not dir_up) and mh > 0):
                penalty += 2.5

        # MTF RSI 정렬
        r = _f(row.get("rsi"), np.nan)
        r5 = _f(row.get("rsi_5m"), np.nan); r15 = _f(row.get("rsi_15m"), np.nan)
        if np.isfinite(r):
            if np.isfinite(r5)  and ((r>=50 and r5>=50) or (r<50 and r5<50)):  bonus += 0.6
            if np.isfinite(r15) and ((r>=50 and r15>=50) or (r<50 and r15<50)): bonus += 0.6

        # 볼륨 스파이크(정규화 피처 없을 때 보조)
        if not np.isfinite(_f(row.get("volume_score"), np.nan)):
            vma = _f(row.get("vol_ma"), np.nan); vr = _f(row.get("volume"), np.nan)
            if np.isfinite(vma) and vma > 0 and np.isfinite(vr) and vr > 1.5*vma:
                bonus += 1.2

        # 디텍터 보너스(캡)
        bonus += self._detector_bonus(row)

        # 레인지 극단 RSI 감점
        if regime == "range" and np.isfinite(r):
            if r >= 72.0 or r <= 28.0:
                penalty += 2.0

        # 합산 (0~1)*100 + bonus - penalty
        score = 100.0 * base01 + max(0.0, min(self.max_bonus, bonus)) - max(0.0, min(self.max_penalty, penalty))

        # 세션 배수
        score *= float(session_mult if np.isfinite(session_mult) else 1.0)

        # 로지스틱 캡(옵션)
        if self.logistic_cap:
            z = max(0.0, min(120.0, score))  # 0~120 영역 가정
            score = 100.0 / (1.0 + math.exp(-(z - self.logistic_mid)/self.logistic_k))

        return round(float(max(0.0, min(100.0, score))), 2)

def adaptive_threshold(
    scores: pd.Series,
    *,
    window_bars: int = 4320,
    pct: float = 0.7,
    floor: float = 65.0,
) -> pd.Series:
    """
    분위수 기반 적응 임계값.
      - 초기 구간: expanding quantile(min_periods=10)
      - 정상 구간: rolling quantile(min_periods=max(30, win/10))
      - ffill + 글로벌 분위수 백업 + floor 하한
    """
    s = pd.to_numeric(scores, errors="coerce")
    if s.empty:
        return pd.Series([], dtype=float)

    win = max(30, int(window_bars))
    exp_q  = s.expanding(min_periods=10).quantile(float(pct))
    roll_q = s.rolling(window=win, min_periods=max(30, win//10)).quantile(float(pct))
    q = roll_q.combine_first(exp_q)

    if s.notna().any():
        global_q = float(s.quantile(float(pct)))
    else:
        global_q = float(floor)

    q = q.ffill().fillna(global_q)
    q = q.clip(lower=float(floor))
    return q
