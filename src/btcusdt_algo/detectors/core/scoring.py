import pandas as pd

class SignalScorer:
    def __init__(self, weights=None, squeeze_th=0.02):
        self.weights = weights or {
            "volatility": 0.25,
            "momentum": 0.25,
            "volume": 0.20,
            "structure": 0.20,
            "fib": 0.10,
        }
        self.squeeze_th = squeeze_th

    def score_row(self, row: pd.Series) -> float:
        score = 0.0
        vol = 1.0 if row.get("bb_width", 0) > self.squeeze_th else 0.0
        rsi = row.get("rsi", 50)
        mom = 1.0 if (rsi >= 55 or rsi <= 45) else 0.0
        vol_ma = row.get("vol_ma", None); vol_raw = row.get("volume", None)
        vol_spike = 1.0 if (vol_ma is not None and vol_raw is not None and vol_ma > 0 and vol_raw > 1.5*vol_ma) else 0.0
        close = row.get("close", None); mid = row.get("bb_mid", None); ema = row.get("ema21", None)
        struct = 1.0 if (close is not None and mid is not None and ((close > mid and close > (ema or mid)) or (close < mid and close < (ema or mid)))) else 0.8
        fib = 0.5
        score += vol * self.weights["volatility"]
        score += mom * self.weights["momentum"]
        score += vol_spike * self.weights["volume"]
        score += struct * self.weights["structure"]
        score += fib * self.weights["fib"]
        return round(score * 100, 2)

def adaptive_threshold(scores, *, window_bars:int=4320, pct:float=0.7, floor:float=65.0) -> float:
    if len(scores) < max(10, window_bars):
        return max(floor, scores.quantile(pct)) if len(scores) else floor
    th = scores.tail(window_bars).quantile(pct)
    return max(floor, th)
