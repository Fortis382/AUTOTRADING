# src/btcusdt_algo/core/indicators.py
import numpy as np
import pandas as pd

# --- optional pandas_ta backend ---
try:
    import pandas_ta as pta
    _HAS_PTA = True
except Exception:
    _HAS_PTA = False

EPS = 1e-8


# --------- Native helpers (Wilder-aligned) ----------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.Series(series, copy=False).ewm(span=int(max(1, span)), adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    # Wilder's RMA
    return pd.Series(series, copy=False).ewm(alpha=1 / int(max(1, period)), adjust=False).mean()


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    c = pd.Series(close, copy=False)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_gain = _rma(up, period)
    avg_loss = _rma(down, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    c = pd.Series(close, copy=False)
    ema_fast = _ema(c, fast)
    ema_slow = _ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bb(close: pd.Series, window=20, sigma=2.0):
    c = pd.Series(close, copy=False)
    mid = c.rolling(int(max(1, window)), min_periods=1).mean()
    std = c.rolling(int(max(1, window)), min_periods=1).std(ddof=0)
    std = std.replace(0, np.nan).bfill().ffill()
    std = std.fillna(std.rolling(5, min_periods=1).mean())  # 마지막 안전망
    high = mid + float(sigma) * std
    low  = mid - float(sigma) * std
    denom = mid.abs().where(mid.abs() > EPS, EPS)
    width = (high - low) / denom
    return mid, high, low, width.abs()


# --------- Robust helpers ----------
def _ensure_in_cols(df: pd.DataFrame):
    """보편적인 OHLCV 컬럼을 강제한다(없으면 생성/캐스팅)."""
    out = df.copy()
    # 기본 컬럼 존재 보장
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise KeyError(f"Input data missing required column: '{col}'")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # volume 없으면 0으로 생성 (볼륨 스파이크 게이트가 있어도 안전)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    else:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    return out


def _pta_pick(df_like: pd.DataFrame, candidates: list[str], fallback: pd.Series | None = None) -> pd.Series:
    """pandas_ta 결과의 가변 컬럼명을 안전하게 선택."""
    for k in candidates:
        if isinstance(df_like, pd.DataFrame) and k in df_like.columns:
            return df_like[k]
    # 마지막 안전: 첫 컬럼
    if isinstance(df_like, pd.DataFrame) and df_like.shape[1] > 0:
        return df_like.iloc[:, 0]
    if fallback is not None:
        return fallback
    # 빈 시리즈 방지
    return pd.Series(np.nan, index=(df_like.index if hasattr(df_like, "index") else None))


# -------------- Unified API --------------
def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    엔진이 요구하는 필수 컬럼을 항상 생성한다.
      - rsi, ema21, atr, macd, macd_signal, macd_hist
      - bb_mid, bb_high, bb_low, bb_width
      - volume, vol_ma (vol_ma_length 기준)
      - (옵션) adx, stoch_k, stoch_d, cci, ich_conv, ich_base
    pandas_ta가 있으면 우선 사용하고, 실패 시 네이티브로 폴백한다.
    """
    ind = (cfg or {}).get("indicators", {}) or {}
    use_pta = bool(ind.get("use_pandas_ta", False)) and _HAS_PTA

    rsi_len  = int(ind.get("rsi_length", 14))
    bb_len   = int(ind.get("bb_length", 20))
    bb_std   = float(ind.get("bb_std", 2.0))
    atr_len  = int(ind.get("atr_length", 14))
    ema_len  = int(ind.get("ema_length", 21))
    macd_fast   = int(ind.get("macd", {}).get("fast", 12))
    macd_slow   = int(ind.get("macd", {}).get("slow", 26))
    macd_signal = int(ind.get("macd", {}).get("signal", 9))
    vol_ma_len  = int(ind.get("vol_ma_length", 50))

    out = _ensure_in_cols(df)

    # ---------- Try pandas_ta backend ----------
    if use_pta:
        try:
            # RSI
            out["rsi"] = pta.rsi(out["close"], length=rsi_len)

            # BBANDS
            bb = pta.bbands(out["close"], length=bb_len, std=bb_std)
            out["bb_low"]  = _pta_pick(bb, [f"BBL_{bb_len}_{bb_std}", "BBL"])
            out["bb_mid"]  = _pta_pick(bb, [f"BBM_{bb_len}_{bb_std}", "BBM"])
            out["bb_high"] = _pta_pick(bb, [f"BBU_{bb_len}_{bb_std}", "BBU"])
            denom = out["bb_mid"].abs().where(out["bb_mid"].abs() > EPS, EPS)
            out["bb_width"] = (out["bb_high"] - out["bb_low"]) / denom

            # ATR
            out["atr"] = pta.atr(out["high"], out["low"], out["close"], length=atr_len)

            # EMA
            out["ema21"] = pta.ema(out["close"], length=ema_len)

            # MACD
            macd_df = pta.macd(out["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            out["macd"]        = _pta_pick(macd_df, [f"MACD_{macd_fast}_{macd_slow}_{macd_signal}", "MACD"])
            out["macd_signal"] = _pta_pick(macd_df, [f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}", "MACDs"])
            out["macd_hist"]   = _pta_pick(macd_df, [f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}", "MACDh"])

            # Optional: ADX
            if ind.get("with_adx", False):
                adx_len = int(ind.get("adx_length", 14))
                adx_df = pta.adx(out["high"], out["low"], out["close"], length=adx_len)
                out["adx"] = _pta_pick(adx_df, [f"ADX_{adx_len}", "ADX"])

            # Optional: Stoch
            if ind.get("with_stoch", False):
                k = int(ind.get("stoch", {}).get("k", 14))
                d = int(ind.get("stoch", {}).get("d", 3))
                sm = int(ind.get("stoch", {}).get("smooth", 3))
                st = pta.stoch(out["high"], out["low"], out["close"], k=k, d=d, smooth_k=sm)
                out["stoch_k"] = _pta_pick(st, [f"STOCHk_{k}_{d}_{sm}", "STOCHk"])
                out["stoch_d"] = _pta_pick(st, [f"STOCHd_{k}_{d}_{sm}", "STOCHd"])

            # Optional: CCI
            if ind.get("with_cci", False):
                cci_len = int(ind.get("cci_length", 20))
                out["cci"] = pta.cci(out["high"], out["low"], out["close"], length=cci_len)

            # Optional: Ichimoku (보수적 키 선택)
            if ind.get("with_ichimoku", False):
                conv = int(ind.get("ichimoku", {}).get("conv", 9))
                base = int(ind.get("ichimoku", {}).get("base", 26))
                spanB= int(ind.get("ichimoku", {}).get("spanB", 52))
                try:
                    ich = pta.ichimoku(out["high"], out["low"], out["close"],
                                       tenkan=conv, kijun=base, senkou=spanB)
                    ich_df = ich[0] if isinstance(ich, tuple) else ich
                    out["ich_conv"] = _pta_pick(ich_df, ["ITS_9", "ITS", "TENKAN"])
                    out["ich_base"] = _pta_pick(ich_df, ["IKS_26", "IKS", "KIJUN"])
                except Exception:
                    # 폴백: 수동 계산
                    out["ich_conv"] = (out["high"].rolling(conv).max() + out["low"].rolling(conv).min()) / 2.0
                    out["ich_base"] = (out["high"].rolling(base).max() + out["low"].rolling(base).min()) / 2.0

        except Exception:
            # pandas_ta 경로에서 하나라도 깨지면 네이티브로 전체 폴백
            use_pta = False

    # ---------- Native fallback ----------
    if not use_pta:
        out["rsi"] = _rsi_wilder(out["close"], rsi_len)
        out["ema21"] = _ema(out["close"], ema_len)
        out["atr"] = _atr_wilder(out, atr_len)
        macd_line, macd_sig, macd_hist = _macd(out["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        out["macd"] = macd_line
        out["macd_signal"] = macd_sig
        out["macd_hist"] = macd_hist
        bb_mid, bb_hi, bb_lo, bb_w = _bb(out["close"], bb_len, bb_std)
        out["bb_mid"], out["bb_high"], out["bb_low"], out["bb_width"] = bb_mid, bb_hi, bb_lo, bb_w

        # 옵션들(ADX/스토캐스틱/CCI/이치모쿠)은 네이티브 미구현 시 생략해도 무방
        if ind.get("with_ichimoku", False):
            conv = int(ind.get("ichimoku", {}).get("conv", 9))
            base = int(ind.get("ichimoku", {}).get("base", 26))
            out["ich_conv"] = (out["high"].rolling(conv).max() + out["low"].rolling(conv).min()) / 2.0
            out["ich_base"] = (out["high"].rolling(base).max() + out["low"].rolling(base).min()) / 2.0

    # ---------- Hygiene & guarantees ----------
    # bb_width 결측/무한 안전화
    out["bb_width"] = pd.to_numeric(out.get("bb_width", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # RSI/EMA/ATR/MACD/BB 필수 보장 (없으면 NaN)
    for col in ["rsi", "ema21", "atr", "macd", "macd_signal", "macd_hist", "bb_mid", "bb_high", "bb_low"]:
        if col not in out.columns:
            out[col] = np.nan

    # 거래량 이동평균 (볼륨 스파이크 게이트용)
    if vol_ma_len > 1:
        out["vol_ma"] = pd.Series(out["volume"], copy=False).rolling(vol_ma_len, min_periods=1).mean()
    else:
        out["vol_ma"] = out["volume"].astype(float)

    # 타입 캐스팅 및 최종 NA 처치(엔진 쪽에서 notna 체크를 하므로 과도한 채우기는 지양)
    for col in ["rsi", "ema21", "atr", "macd", "macd_signal", "macd_hist", "bb_mid", "bb_high", "bb_low", "bb_width", "vol_ma"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out
