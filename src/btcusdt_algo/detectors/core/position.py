from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime, timedelta

@dataclass
class Position:
    side: str = ""      # 'long' or 'short'
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    open: bool = False
    entry_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    rsi_trailing: bool = False

class PositionManager:
    def __init__(self, rr=(1,2), atr_sl_mult=1.2, trail_conf=None, max_hold_minutes=240, fees_bps_per_side=6):
        self.pos = Position()
        self.rr = rr
        self.atr_sl_mult = atr_sl_mult
        self.trail_conf = trail_conf or {
            "enable_rsi": True,
            "rsi_on_long": 65,
            "rsi_on_short": 35,
            "atr_trail_mult": 1.0,
        }
        self.max_hold = timedelta(minutes=int(max_hold_minutes))
        self.fees_bps = fees_bps_per_side

    def flat(self) -> bool:
        return not self.pos.open

    def open_position(self, side: str, price: float, atr: float, now: datetime):
        sl_dist = max(atr * self.atr_sl_mult, 1e-8)
        tp_dist = sl_dist * (self.rr[1] / max(self.rr[0], 1e-8))
        if side == "long":
            sl = price - sl_dist; tp = price + tp_dist
        else:
            sl = price + sl_dist; tp = price - tp_dist
        self.pos = Position(side=side, entry=price, sl=sl, tp=tp, open=True, entry_time=now, last_update=now, rsi_trailing=False)

    def close_reason(self, price: float, now: datetime, rsi: float, atr: float) -> Optional[str]:
        if not self.pos.open: return None
        if self.pos.side == "long":
            if price <= self.pos.sl: return "SL"
            if price >= self.pos.tp: return "TP"
        else:
            if price >= self.pos.sl: return "SL"
            if price <= self.pos.tp: return "TP"
        if now - (self.pos.entry_time or now) >= self.max_hold:
            return "TIME"
        return None

    def update_trailing(self, price: float, rsi: float, atr: float):
        if not self.pos.open or not self.trail_conf.get("enable_rsi", True): return
        atr_trail = max(atr * float(self.trail_conf.get("atr_trail_mult", 1.0)), 1e-8)
        if self.pos.side == "long" and rsi >= float(self.trail_conf.get("rsi_on_long", 65)):
            self.pos.rsi_trailing = True
        if self.pos.side == "short" and rsi <= float(self.trail_conf.get("rsi_on_short", 35)):
            self.pos.rsi_trailing = True
        if self.pos.rsi_trailing:
            if self.pos.side == "long":
                self.pos.sl = max(self.pos.sl, price - atr_trail)
            else:
                self.pos.sl = min(self.pos.sl, price + atr_trail)

    def settle_trade_R(self, exit_price: float) -> float:
        init_risk = abs(self.pos.entry - self.pos.sl)
        if init_risk <= 0: return 0.0
        r_mult = (exit_price - self.pos.entry)/init_risk if self.pos.side == "long" else (self.pos.entry - exit_price)/init_risk
        fee_frac = 2 * (self.fees_bps / 10000.0)
        r_mult -= fee_frac
        return r_mult

    def try_close(self, price: float, now: datetime, rsi: float, atr: float) -> Optional[Dict]:
        reason = self.close_reason(price, now, rsi, atr)
        if reason:
            r = self.settle_trade_R(price)
            side = self.pos.side; entry = self.pos.entry
            self.pos = Position()
            return {"reason": reason, "R": round(r, 4), "side": side, "exit": price, "entry": entry, "time": now}
        return None
