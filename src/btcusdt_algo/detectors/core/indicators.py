import pandas as pd
import pandas_ta as pta

def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ind = cfg.get("indicators", {})
    out = df.copy()

    # Core
    out['rsi'] = pta.rsi(out['close'], length=int(ind.get("rsi_length",14)))
    bb_len = int(ind.get("bb_length",20)); bb_std = float(ind.get("bb_std",2.0))
    bb = pta.bbands(out['close'], length=bb_len, std=bb_std)
    out['bb_low']  = bb.get(f'BBL_{bb_len}_{bb_std}', bb.iloc[:,0])
    out['bb_mid']  = bb.get(f'BBM_{bb_len}_{bb_std}', bb.iloc[:,1])
    out['bb_high'] = bb.get(f'BBU_{bb_len}_{bb_std}', bb.iloc[:,2])
    out['bb_width'] = (out['bb_high'] - out['bb_low']) / out['bb_mid']
    out['atr'] = pta.atr(high=out['high'], low=out['low'], close=out['close'], length=int(ind.get("atr_length",14)))
    out['vol_ma'] = out['volume'].rolling(int(ind.get("vol_ma_length",50))).mean()

    # Trend/MTF helpers
    out['ema21'] = pta.ema(out['close'], length=int(ind.get("ema_length",21)))
    out['adx'] = pta.adx(high=out['high'], low=out['low'], close=out['close'], length=int(ind.get("adx_length",14)))['ADX_14']

    # MACD
    macd = pta.macd(out['close'],
                    fast=int(ind.get("macd",{}).get("fast",12)),
                    slow=int(ind.get("macd",{}).get("slow",26)),
                    signal=int(ind.get("macd",{}).get("signal",9)))
    out['macd'] = macd.iloc[:,0]
    out['macd_signal'] = macd.iloc[:,1]
    out['macd_hist'] = macd.iloc[:,2]

    # Stochastic
    st = pta.stoch(high=out['high'], low=out['low'], close=out['close'],
                   k=int(ind.get("stoch",{}).get("k",14)),
                   d=int(ind.get("stoch",{}).get("d",3)),
                   smooth_k=int(ind.get("stoch",{}).get("smooth",3)))
    out['stoch_k'] = st.iloc[:,0]; out['stoch_d'] = st.iloc[:,1]

    # CCI
    out['cci'] = pta.cci(high=out['high'], low=out['low'], close=out['close'], length=20)

    # Ichimoku (basic)
    ich = pta.ichimoku(conversion=int(ind.get("ichimoku",{}).get("conv",9)),
                       base=int(ind.get("ichimoku",{}).get("base",26)),
                       span_b=int(ind.get("ichimoku",{}).get("spanB",52)))[0]
    out['ich_conv'] = ich['ITS_9']; out['ich_base'] = ich['IKS_26']

    return out
