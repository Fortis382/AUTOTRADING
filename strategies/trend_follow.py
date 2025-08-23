from .base import Strategy

class TrendFollow(Strategy):
    def should_long(self, row) -> bool:
        return (row['regime']=='trend') and (row['close']>=row['ema21']) and (row['rsi']>=50)
    def should_short(self, row) -> bool:
        return (row['regime']=='trend') and (row['close']<=row['ema21']) and (row['rsi']<50)
