from .base import Strategy

class RangeRevert(Strategy):
    def should_long(self, row) -> bool:
        return (row['regime']=='range') and (row['close']<=row['bb_low']) and (row['rsi']<=35)
    def should_short(self, row) -> bool:
        return (row['regime']=='range') and (row['close']>=row['bb_high']) and (row['rsi']>=65)
