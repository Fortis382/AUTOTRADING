# src/btcusdt_algo/core/position.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import math

def _clamp(x, lo, hi): 
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo

@dataclass
class Position:
    side: str = ""      # 'long' or 'short'
    entry: float = 0.0  # R 계산 기준 base entry(고정)
    sl: float = 0.0
    tp: float = 0.0
    open: bool = False
    entry_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    rsi_trailing: bool = False
    size: float = 1.0              # 현재 잔여 비중(1.0=100%)
    init_risk: float = 0.0         # 가격단위 R (ATR*mult), 진입 시 고정
    add_on_count: int = 0
    ladder: List[Dict] = field(default_factory=list)
    # tracking
    mfe_R: float = 0.0
    mae_R: float = 0.0
    regime: str = ""
    strategy: str = ""
    session: str = ""
    # flags
    be_armed: bool = False
    trail_armed: bool = False

class PositionManager:
    def __init__(self, rr=(1,2), atr_sl_mult=1.2, trail_conf=None, max_hold_minutes=240, fees_bps_per_side=6,
                 add_on_max:int=0, add_on_trigger_R:float=0.8, add_on_size_pct:float=33.0,
                 ladder_scheme:Optional[List[Dict]]=None, be_at_R: float = 1.2, **kwargs):
        """
        kwargs 백워드 호환:
          - partials: 과거 엔진에서 분할체결 라더를 'partials' 키로 넘긴 경우 매핑
          - 기타 알 수 없는 인자는 무시
        """
        # 호환 매핑
        if ladder_scheme is None and kwargs.get("partials"):
            ladder_scheme = kwargs.get("partials")

        self.pos = Position()
        self.rr = tuple(rr)
        try:
            self.atr_sl_mult = float(atr_sl_mult)
        except Exception:
            self.atr_sl_mult = 1.2

        # trailing 설정 기본값
        self.trail_conf = {
            "enable_rsi": True, "rsi_on_long": 65, "rsi_on_short": 35,
            "atr_trail_mult": 1.0, "trail_start_R": 0.8, "wait_first_ladder": False
        }
        if isinstance(trail_conf, dict):
            self.trail_conf.update(trail_conf)

        self.max_hold = timedelta(minutes=int(max_hold_minutes))
        self.fees_bps = float(fees_bps_per_side)
        self.add_on_max = int(add_on_max)
        self.add_on_trigger_R = float(add_on_trigger_R)
        self.add_on_size_pct = float(add_on_size_pct) / 100.0
        self.be_at_R = float(be_at_R)

        # 라더: pct는 0~1(비중)로 사용
        if ladder_scheme is None:
            self.default_ladder = [
                {"R":1.0, "pct":0.30, "done":False},
                {"R":2.0, "pct":0.30, "done":False},
                {"R":4.0, "pct":0.40, "done":False},
            ]
        else:
            self.default_ladder = [dict(x) for x in ladder_scheme]

    # ---------- 기본 유틸 ----------
    def flat(self) -> bool:
        return (not self.pos.open) or (self.pos.size <= 0.0)

    def _R_value(self) -> float:
        return max(float(self.pos.init_risk), 1e-12)

    def _current_R(self, price: float) -> float:
        if self.pos.init_risk <= 0: 
            return 0.0
        if self.pos.side == "long":
            return (float(price) - self.pos.entry) / self.pos.init_risk
        else:
            return (self.pos.entry - float(price)) / self.pos.init_risk

    def _fees_R(self, price_ref: float, fraction: float) -> float:
        """왕복 수수료를 R 단위로 환산하여 fraction 비중만큼 차감."""
        px = abs(float(price_ref))
        bps = max(0.0, float(self.fees_bps))
        roundtrip_px = px * (bps / 10000.0) * 2.0
        return (roundtrip_px / self._R_value()) * float(max(0.0, fraction))

    # ---------- 엔트리/업데이트 ----------
    def open_position(self, side: str, price: float, atr: float, now: datetime, size: float=1.0,
                      regime: str="", strategy: str="", session: str=""):
        atr = abs(float(atr))
        sl_dist = max(atr * self.atr_sl_mult, 1e-8)  # ATR 0 방어
        tp_dist = sl_dist * (float(self.rr[1]) / max(float(self.rr[0]), 1e-8))
        price = float(price)

        if side == "long":
            sl = price - sl_dist; tp = price + tp_dist
        else:
            sl = price + sl_dist; tp = price - tp_dist

        self.pos = Position(
            side=str(side), entry=price, sl=sl, tp=tp, open=True,
            entry_time=now, last_update=now, rsi_trailing=False,
            size=float(size), init_risk=sl_dist, add_on_count=0,
            ladder=[dict(x) for x in self.default_ladder],
            mfe_R=0.0, mae_R=0.0, regime=str(regime), strategy=str(strategy), session=str(session),
            be_armed=False, trail_armed=False
        )

    def update_excursions(self, price: float):
        r = self._current_R(price)
        if r > self.pos.mfe_R: self.pos.mfe_R = r
        if r < self.pos.mae_R: self.pos.mae_R = r

    def update_trailing(self, price: float, rsi: float, atr: float):
        if not self.pos.open:
            return

        # BE 승격
        curR = self._current_R(price)
        if (not self.pos.be_armed) and curR >= self.be_at_R:
            if self.pos.side == "long":
                self.pos.sl = max(self.pos.sl, self.pos.entry)
            else:
                self.pos.sl = min(self.pos.sl, self.pos.entry)
            self.pos.be_armed = True

        # 트레일 시작 조건
        start_R = float(self.trail_conf.get("trail_start_R", 0.0))
        if (not self.pos.trail_armed) and curR >= start_R:
            if self.trail_conf.get("enable_rsi", True):
                try:
                    rsi_val = float(rsi)
                except Exception:
                    rsi_val = math.nan
                if (self.pos.side == "long" and rsi_val >= float(self.trail_conf.get("rsi_on_long", 65))) or \
                   (self.pos.side == "short" and rsi_val <= float(self.trail_conf.get("rsi_on_short", 35))):
                    self.pos.trail_armed = True
            else:
                self.pos.trail_armed = True

        # ATR 트레일
        if self.pos.trail_armed:
            atr = abs(float(atr))
            atr_mult = float(self.trail_conf.get("atr_trail_mult", 1.0))
            trail_dist = max(atr * atr_mult, 1e-8)
            price = float(price)
            if self.pos.side == "long":
                self.pos.sl = max(self.pos.sl, price - trail_dist)
            else:
                self.pos.sl = min(self.pos.sl, price + trail_dist)

    # ---------- 유지보수(분할/애드온) ----------
    def _apply_ladder_steps(self, price: float, now: datetime) -> List[Dict]:
        """한 캔들에서 여러 스텝 동시 충족시 연속 처리."""
        events: List[Dict] = []
        if self.flat(): 
            return events

        curR = self._current_R(price)
        progressed = True
        while progressed and not self.flat():
            progressed = False
            for lvl in self.pos.ladder:
                if lvl.get("done"): 
                    continue
                trig = float(lvl.get("R", 0.0))
                pct  = _clamp(lvl.get("pct", 0.0), 0.0, 1.0)
                if curR >= trig and self.pos.size > 0.0 and pct > 0.0:
                    lvl["done"] = True
                    # 현재 잔여 대비 실청산 비중
                    fraction = _clamp(self.pos.size * pct, 0.0, self.pos.size)
                    pnl_R = curR * fraction
                    fee_R = self._fees_R(price, fraction)
                    pnl_R_net = pnl_R - fee_R
                    trade = {
                        "reason": f"TP@{trig:.2f}R",
                        "R": round(pnl_R_net, 6),
                        "side": self.pos.side,
                        "exit": float(price), "entry": self.pos.entry, "time": now,
                        "partial": True, "partial_pct": round(fraction, 6),  # 0~1 비중
                        "strategy": self.pos.strategy, "regime": self.pos.regime, "session": self.pos.session,
                        "mfe_R": self.pos.mfe_R, "mae_R": self.pos.mae_R
                    }
                    events.append(trade)
                    self.pos.size = round(self.pos.size - fraction, 6)
                    progressed = True
                    curR = self._current_R(price)
                    if self.pos.size <= 1e-9:
                        # 전량 종료(라더로 모두 소진)
                        self.pos = Position()
                        return events
        return events

    def _maybe_add_on(self, price: float):
        """애드온: R 참조는 base_entry 고정. 사이즈만 늘려서 간단 운용(기대R 과대/과소 방지)."""
        if self.pos.add_on_count >= self.add_on_max or self.pos.size <= 0.0:
            return
        curR = self._current_R(price)
        if curR >= self.add_on_trigger_R:
            add_frac = _clamp(self.add_on_size_pct, 0.0, 1.0)
            self.pos.size = _clamp(self.pos.size * (1.0 + add_frac), 0.0, 2.0)
            self.pos.add_on_count += 1
            # 엔트리/SL/TP는 R계산 보존 위해 유지

    def try_maintenance(self, price: float, now: datetime) -> List[Dict]:
        if self.flat(): 
            return []
        self.update_excursions(price)
        ev = self._apply_ladder_steps(price, now)
        if self.add_on_max > 0:
            self._maybe_add_on(price)
        return ev

    # ---------- 소프트 TP ----------
    def maybe_soft_tp(self, price: float, now: datetime, rsi: float, macd_hist: float, conf: dict):
        if not self.pos.open: 
            return None
        conf = conf or {}

        r_min  = float(conf.get("r_min", 1.0))
        r_hard = float(conf.get("r_hard", float("inf")))
        macd_ok = bool(conf.get("macd_flip", True))
        rsi_fade = float(conf.get("rsi_fade", 52.0 if "rsi_fade" in conf else 52.0))

        curR = self._current_R(price)
        if curR >= r_hard:
            return self._close_all(price, now, "SOFT_TP_HARD")

        # 모멘텀 페이드
        try:
            rsi_val = float(rsi)
        except Exception:
            rsi_val = math.nan
        try:
            macd_val = float(macd_hist)
        except Exception:
            macd_val = math.nan

        if curR >= r_min:
            bad_rsi = (rsi_val < rsi_fade) if self.pos.side == "long" else (rsi_val > (100.0 - rsi_fade))
            bad_macd = (macd_val < 0.0) if self.pos.side == "long" else (macd_val > 0.0)
            if bad_rsi or (macd_ok and bad_macd):
                return self._close_all(price, now, "SOFT_TP")
        return None

    # ---------- 하드 클로즈 ----------
    def close_reason(
        self,
        price: float,
        now: datetime,
        bar_high: float | None = None,
        bar_low: float | None = None,
    ) -> Optional[str]:
        if not self.pos.open:
            return None

        # 보수적: SL 터치가 보이면 SL 우선 처리
        if self.pos.side == "long":
            if bar_low is not None and bar_low <= self.pos.sl:
                return "SL"
            if bar_high is not None and bar_high >= self.pos.tp:
                return "TP"
            # 극단값이 없을 때만 종가로 폴백
            if price <= self.pos.sl:
                return "SL"
            if price >= self.pos.tp:
                return "TP"
        else:
            if bar_high is not None and bar_high >= self.pos.sl:
                return "SL"
            if bar_low is not None and bar_low <= self.pos.tp:
                return "TP"
            if price >= self.pos.sl:
                return "SL"
            if price <= self.pos.tp:
                return "TP"

        if now - (self.pos.entry_time or now) >= self.max_hold:
            return "TIME"
        return None

    def try_close(
        self,
        price: float,
        now: datetime,
        rsi: float,
        atr: float,
        bar_high: float | None = None,
        bar_low: float | None = None,
    ) -> Optional[Dict]:
        rsn = self.close_reason(price, now, bar_high=bar_high, bar_low=bar_low)
        if not rsn:
            return None
        exit_price = self.pos.sl if rsn == "SL" else (self.pos.tp if rsn == "TP" else price)
        return self._close_all(exit_price, now, rsn)

    # ---------- 내부: 전량 청산 ----------
    def _close_all(self, price: float, now: datetime, reason: str) -> Dict:
        curR = self._current_R(price)
        frac = _clamp(self.pos.size, 0.0, 1.0)
        fee_R = self._fees_R(price, frac)
        pnl_R_net = curR * frac - fee_R
        trade = {
            "reason": str(reason), "R": round(pnl_R_net, 6), "side": self.pos.side,
            "exit": float(price), "entry": self.pos.entry, "time": now,
            "partial": False, "partial_pct": 1.0,
            "strategy": self.pos.strategy, "regime": self.pos.regime, "session": self.pos.session,
            "mfe_R": self.pos.mfe_R, "mae_R": self.pos.mae_R
        }
        self.pos = Position()
        return trade
