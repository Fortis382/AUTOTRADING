import pandas as pd

def classify_regime(df: pd.DataFrame, adx_min: float, bb_width_th: float) -> pd.Series:
    cond_trend = (df['adx'] >= float(adx_min)) | (df['bb_width'] >= float(bb_width_th))
    return pd.Series(['trend' if t else 'range' for t in cond_trend], index=df.index, name='regime')
