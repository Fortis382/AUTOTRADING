# src/btcusdt_algo/core/features.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

EPS = 1e-8

# ---------------- helpers ----------------
def _qnorm(s: pd.Series, qlo: float = 0.01, qhi: float = 0.99) -> pd.Series:
    """
    분위수 기반 0~100 스케일링. NaN/상수 구간 안전.
    """
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(qlo)
    hi = s.quantile(qhi)
    if not np.isfinite(lo): lo = s.min()
    if not np.isfinite(hi): hi = s.max()
    if not np.isfinite(lo): lo = 0.0
    if (not np.isfinite(hi)) or (hi - lo < EPS):
        return pd.Series(50.0, index=s.index)
    s = s.clip(lower=lo, upper=hi)
    out = (s - lo) / (hi - lo + EPS) * 100.0
    return out.fillna(50.0)

def _safe_col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """
    항상 df.index에 맞는 1-D Series를 반환(스칼라/DF 금지).
    """
    try:
        idx = df.index
    except Exception:
        idx = None

    if isinstance(df, pd.DataFrame) and (name in df.columns):
        s = df[name]
    else:
        s = pd.Series(default, index=idx)

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    try:
        s = pd.to_numeric(s, errors="coerce")
    except Exception:
        s = pd.Series(default, index=idx)

    if pd.api.types.is_numeric_dtype(s):
        s = s.fillna(default)

    return s

def _get_det_cfg(cfg: dict, key: str) -> dict:
    """
    상위 레벨(cfg[key]) 우선, detectors[key]가 있으면 덮어씀.
    """
    top  = (cfg or {}).get(key, {}) or {}
    nest = ((cfg or {}).get("detectors", {}) or {}).get(key, {}) or {}
    out = dict(top); out.update(nest)
    return out

# --------------- basics -------------------
def _ensure_basics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    엔진이 기대하는 최소 컬럼 보장 + 일관된 bb_width 정의.
    """
    out = df.copy()
    ind = (cfg or {}).get("indicators", {}) or {}

    for col in ("open","high","low","close"):
        if col not in out.columns:
            raise KeyError(f"build_features: input missing column '{col}'")

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.sort_values("timestamp").reset_index(drop=True)

    if "volume" not in out.columns:
        out["volume"] = 0.0

    # BB 구성요소 폴백
    win = int(ind.get("bb_length", 20))
    if "bb_mid" not in out.columns:
        out["bb_mid"] = out["close"].rolling(win, min_periods=1).mean()
    if "bb_low" not in out.columns:
        out["bb_low"] = out["close"].rolling(win, min_periods=1).min()
    if "bb_high" not in out.columns:
        out["bb_high"] = out["close"].rolling(win, min_periods=1).max()

    # width = (high-low)/|mid|
    denom = out["bb_mid"].abs().where(out["bb_mid"].abs() > EPS, EPS)
    out["bb_width"] = ((out["bb_high"] - out["bb_low"]) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # EMA/RSI 폴백(지표 모듈에서 못 올렸을 때 안전)
    if "ema21" not in out.columns:
        ema_p = int(ind.get("ema_length", ind.get("ema21_period", 21)))
        out["ema21"] = out["close"].ewm(span=ema_p, adjust=False).mean()
    if "rsi" not in out.columns:
        out["rsi"] = 50.0

    # macd_hist 폴백
    if "macd_hist" not in out.columns:
        out["macd_hist"] = 0.0

    # vol_ma 기준선
    if "vol_ma" not in out.columns:
        vlen = int(ind.get("vol_ma_length", 50))
        out["vol_ma"] = out["volume"].rolling(vlen, min_periods=max(5, vlen//3)).mean()

    return out

# --------------- auxiliary features -------------------
def _aux_block(out: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    스코어 외에 엔진/디버그에 도움되는 보조 피처.
    """
    c   = _safe_col(out, "close", np.nan)
    mid = _safe_col(out, "bb_mid", np.nan)
    ema = _safe_col(out, "ema21", np.nan)
    hi  = _safe_col(out, "bb_high", np.nan)
    lo  = _safe_col(out, "bb_low", np.nan)

    # 밴드 내 위치(0~1)
    span = (hi - lo).replace(0, np.nan)
    out["bb_pos01"] = ((c - lo) / (span + EPS)).clip(0.0, 1.0).fillna(0.5)

    # ATR 퍼센트(있으면), 없으면 0
    atr = _safe_col(out, "atr", 0.0)
    out["atr_pct"] = (atr / (c.abs().where(c.abs() > EPS, EPS))).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ema/중심선까지 거리(%)
    out["ema_dist_pct"] = ((c - ema) / (ema.abs().where(ema.abs() > EPS, EPS))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["mid_dist_pct"] = ((c - mid) / (mid.abs().where(mid.abs() > EPS, EPS))).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # RSI 중심(50) 거리
    rsi = _safe_col(out, "rsi", 50.0)
    out["rsi_dist"] = (rsi - 50.0).fillna(0.0)

    # MTF 합의 점수(-1~+1): 없으면 0
    r5  = _safe_col(out, "rsi_5m", np.nan)
    r15 = _safe_col(out, "rsi_15m", np.nan)
    agree = pd.Series(0.0, index=out.index)
    agree += np.where(np.isfinite(r5),  np.where(((rsi>=50)&(r5>=50)) | ((rsi<50)&(r5<50)),  0.5, -0.5), 0.0)
    agree += np.where(np.isfinite(r15), np.where(((rsi>=50)&(r15>=50))| ((rsi<50)&(r15<50)), 0.5, -0.5), 0.0)
    out["mtf_agree01"] = (agree / 2.0).clip(-1.0, 1.0)

    return out

# --------------- scores (0~100) ---------------
def _score_block(out: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ind = (cfg or {}).get("indicators", {}) or {}
    reg = (cfg or {}).get("regime", {}) or {}

    # 1) Volatility: squeeze_th 주변 선형 + 분위수 블렌드
    bbw = _safe_col(out, "bb_width", 0.0)
    squeeze_th = float(ind.get("bb_squeeze_th", 0.028))
    lo, hi = squeeze_th * 0.7, squeeze_th * 1.4
    vol01 = ((bbw - lo) / (hi - lo + EPS)).clip(0.0, 1.0)
    vol_q = (_qnorm(bbw, 0.01, 0.99) / 100.0).clip(0.0, 1.0)
    vol01 = (0.75 * vol01 + 0.25 * vol_q).clip(0.0, 1.0)
    out["volatility_score"] = (vol01 * 100.0).round(2)

    # 2) Momentum: |RSI-50|/15 + 5m/15m 정렬 보정
    rsi  = _safe_col(out, "rsi", 50.0)
    r5   = _safe_col(out, "rsi_5m", np.nan)
    r15  = _safe_col(out, "rsi_15m", np.nan)
    core01 = (rsi.sub(50.0).abs() / 15.0).clip(0.0, 1.0)
    align  = pd.Series(0.0, index=out.index)
    align += np.where(np.isfinite(r5),  np.where(((rsi>=50)&(r5>=50)) | ((rsi<50)&(r5<50)), 0.15, -0.10), 0.0)
    align += np.where(np.isfinite(r15), np.where(((rsi>=50)&(r15>=50))| ((rsi<50)&(r15<50)),0.15, -0.10), 0.0)
    mom01 = (core01 + align).clip(0.0, 1.0)
    out["momentum_score"] = (mom01 * 100.0).round(2)

    # 3) Volume: ratio(1.0→0, 2.5x→1.0) + z-score 20%
    v   = _safe_col(out, "volume", np.nan)
    if "vol_ma" not in out.columns:
        vlen = int(ind.get("vol_ma_length", 50))
        out["vol_ma"] = v.rolling(vlen, min_periods=max(5, vlen//3)).mean()
    vma = _safe_col(out, "vol_ma", np.nan)
    ratio01 = ((v / vma) - 1.0) / 1.5
    ratio01 = ratio01.replace([np.inf,-np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)
    win = int((cfg.get("features", {}) or {}).get("volume_window", 100))
    mu = v.rolling(win, min_periods=max(10, win//5)).mean()
    sd = v.rolling(win, min_periods=max(10, win//5)).std(ddof=0).replace(0, np.nan)
    z  = ((v - mu) / (sd + EPS)).clip(-3, 3).fillna(0.0)
    z01 = ((z + 3.0) / 6.0).clip(0.0, 1.0)
    vol01 = (0.8 * ratio01 + 0.2 * z01).clip(0.0, 1.0)
    out["volume_score"] = (vol01 * 100.0).round(2)

    # 4) Structure: bb_mid/ema 정렬 + squeeze 패널티 + ema slope 보너스
    close = _safe_col(out, "close", np.nan)
    mid   = _safe_col(out, "bb_mid", np.nan)
    ema   = _safe_col(out, "ema21", np.nan)
    sgn_mid = np.sign(close - mid); sgn_ema = np.sign(close - ema)
    aligned = pd.Series(0.6, index=out.index)
    aligned[(sgn_mid == sgn_ema) & (sgn_mid != 0.0)] = 1.0
    aligned[(sgn_mid == 0.0) | (sgn_ema == 0.0)] = 0.8
    bbw2 = _safe_col(out, "bb_width", np.nan)
    aligned = np.where(bbw2 < squeeze_th * 0.6, np.maximum(0.3, aligned - 0.2), aligned)

    look = int(reg.get("ema_slope_lookback", 5))
    ema_prev = ema.shift(look)
    ema_slope_rel = ((ema - ema_prev) / (ema_prev.abs().where(ema_prev.abs() > EPS, EPS))).replace([np.inf,-np.inf], np.nan).fillna(0.0).abs()
    slope_bonus = (ema_slope_rel / float(reg.get("ema_slope_min", 0.0006))).clip(0.0, 1.0) * 0.1

    struct01 = np.clip(pd.Series(aligned, index=out.index) + slope_bonus, 0.0, 1.0)
    out["structure_score"] = (struct01 * 100.0).round(2)
    out["ema_slope_rel"] = ema_slope_rel  # 참고용

    # 5) Fib: 플래그 우선, 없으면 slope 프록시
    fib01 = pd.Series(0.5, index=out.index, dtype=float)
    has_flag = False
    if "near_fib_618" in out.columns:
        has_flag = True
        fib01[out["near_fib_618"].fillna(False)] = 1.0

    idx_382_500 = None
    if "near_fib_382" in out.columns or "near_fib_500" in out.columns:
        a = out.get("near_fib_382", False)
        b = out.get("near_fib_500", False)
        a = a.fillna(False) if isinstance(a, pd.Series) else a
        b = b.fillna(False) if isinstance(b, pd.Series) else b
        idx_382_500 = (a | b)

    if isinstance(idx_382_500, (pd.Series, np.ndarray)):
        mask = idx_382_500 if isinstance(idx_382_500, pd.Series) else pd.Series(idx_382_500, index=out.index)
        fib01[mask] = np.maximum(fib01[mask], 0.7)

    idx_ext = None
    if "near_ext_127" in out.columns or "near_ext_161" in out.columns:
        a = out.get("near_ext_127", False)
        b = out.get("near_ext_161", False)
        a = a.fillna(False) if isinstance(a, pd.Series) else a
        b = b.fillna(False) if isinstance(b, pd.Series) else b
        idx_ext = (a | b)

    if isinstance(idx_ext, (pd.Series, np.ndarray)):
        mask = idx_ext if isinstance(idx_ext, pd.Series) else pd.Series(idx_ext, index=out.index)
        fib01[mask] = np.maximum(fib01[mask], 0.8)

    if not has_flag:
        fib01 = (_qnorm(ema_slope_rel.abs().fillna(0.0), 0.01, 0.99) / 100.0).clip(0.0, 1.0)

    out["fib_score"] = (fib01 * 100.0).round(2)

    # 위생
    for c in ["volatility_score","momentum_score","volume_score","structure_score","fib_score"]:
        out[c] = out[c].replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(0, 100)

    return out

# -------------- detectors (safe) --------------
def _attach_detectors(out: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    탐지기가 실패하더라도 전체 파이프는 멈추지 않도록 try/except로 보호.
    설정 키는 detectors.{divergence,fib,darvas} 를 우선 사용.
    """
    # Divergence
    div_cfg = _get_det_cfg(cfg, "divergence")
    if div_cfg.get("enable", True):
        try:
            from btcusdt_algo.detectors.divergence import detect_divergence
            out = pd.concat([out, detect_divergence(
                out,
                lookback_swings=int(div_cfg.get("lookback_swings", 60)),
                wing=int(div_cfg.get("wing", 2)),
                confirm_bars=int(div_cfg.get("confirm_bars", 1)),
                use_hidden=bool(div_cfg.get("use_hidden", False)),
                rsi_col=str(div_cfg.get("rsi_col", "rsi")),
                macd_hist_col=str(div_cfg.get("macd_hist_col", "macd_hist")),
            )], axis=1)
        except Exception:
            pass

    # Fib proximity
    fib_cfg = _get_det_cfg(cfg, "fib")
    if fib_cfg.get("enable", True):
        try:
            from btcusdt_algo.detectors.fib import fib_proximity
            out = pd.concat([out, fib_proximity(
                out,
                lookback=int(fib_cfg.get("lookback", 120)),
                eps_pct=float(fib_cfg.get("epsilon_pct", 0.15)),
                wing=int(fib_cfg.get("wing", 2)),
                include_extra=bool(fib_cfg.get("include_extra", True)),
                extra_retracements=tuple(fib_cfg.get("extra_retracements", (0.786,))),
                extra_extensions=tuple(fib_cfg.get("extra_extensions", ())),
                extra_level_eps_pct=float(fib_cfg.get("extra_level_eps_pct", 0.06)),
                atr_floor_mult=float(fib_cfg.get("atr_floor_mult", 0.25)),
            )], axis=1)
        except Exception:
            pass

    # Darvas (키 이름을 'darvas'로 통일)
    dv_cfg = _get_det_cfg(cfg, "darvas")
    if dv_cfg.get("enable", True):
        try:
            from btcusdt_algo.detectors.darvas import darvas_signals
            out = pd.concat([out, darvas_signals(
                out,
                box_len=int(dv_cfg.get("channel_min_bars", 20)),
                confirm_bars=int(dv_cfg.get("breakout_confirm_bars", 2)),
                min_range_pct=float(dv_cfg.get("min_range_pct", 0.003)),
                atr_min_mult=float(dv_cfg.get("atr_min_mult", 0.0)),
                touch_eps_pct=float(dv_cfg.get("touch_eps_pct", 0.0005)),
                return_box_levels=bool(dv_cfg.get("return_box_levels", False)),
            )], axis=1)
        except Exception:
            pass

    # 누락 플래그는 False로 채움(엔진 안전)
    needed = [
        "bull_div_rsi","bear_div_rsi","bull_div_macd","bear_div_macd",
        "near_fib_382","near_fib_500","near_fib_618","near_ext_127","near_ext_161",
        "darvas_up","darvas_dn"
    ]
    if bool(div_cfg.get("use_hidden", False)):
        needed += ["hidden_bull_div_rsi","hidden_bear_div_rsi","hidden_bull_div_macd","hidden_bear_div_macd"]

    for col in needed:
        if col not in out.columns:
            out[col] = False

    return out

# -------------- public API --------------
def build_features(df: pd.DataFrame, use_detectors: bool = True, cfg: dict | None = None) -> pd.DataFrame:
    """
    - 필수 컬럼 정비 및 폴백 (_ensure_basics)
    - 보조 피처 생성 (_aux_block)
    - 점수(0~100) 계산 (_score_block)
    - 탐지기 부착(옵션) (_attach_detectors)
    """
    cfg = cfg or {}
    out = _ensure_basics(df, cfg)
    out = _aux_block(out, cfg)
    out = _score_block(out, cfg)
    if use_detectors:
        out = _attach_detectors(out, cfg)

    # 마지막 위생: 무한대/NaN 제거
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out
