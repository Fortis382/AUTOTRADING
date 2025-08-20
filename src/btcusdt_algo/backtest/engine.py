# src/btcusdt_algo/backtest/engine.py
"""
Backtest Orchestrator (BTCUSDT) — MTF Driver + RSI-Extrema Aware

핵심 포인트
- 15m 'driver'가 방향(롱/숏/중립)과 엔트리 윈도우를 결정, 1m는 타이밍만.
- 1h/4h 상위TF 동조(선택) + 5m/15m MTF RSI & EMA 게이트(레거시) 동시 지원.
- MACD deadband: abs / pct_of_close / pct_of_atr 유지.
- Micro triggers: 돌파 + (옵션) Regime/MTF/EMA/ATR/Volume/Darvas + RSI 극단(70/30) FOD/SOD 플래그.
- Extreme RSI block: rsi_peak_recent / rsi_trough_recent 기반 과열 직후 진입 억제(옵션).
- 세션 점수 배수, 리스크 조정, BE 승격, 분할익절, 트레일, 소프트TP 등 유지.
- 디버그 JSON에 모든 게이트 카운터와 샘플 포함.

안전 기본값
- settings['mtf_driver']['enable']=False 이면 과거(레거시) 엔진처럼 동작.
- 'entries' 섹션을 사용 중이라면, entries.decision_tf / align_mode / scan_window_mins를 읽어 driver에 자동 반영.
"""

from __future__ import annotations

import os, json
from datetime import datetime, timezone
from collections import Counter
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from loguru import logger

from btcusdt_algo.core.indicators import add_indicators
from btcusdt_algo.core.regime import classify_regime
from btcusdt_algo.core.features import build_features
from btcusdt_algo.core.scoring import SignalScorer, adaptive_threshold
from btcusdt_algo.core.position import PositionManager
from btcusdt_algo.core.session import score_mult, which as which_session
from btcusdt_algo.core.mtf import resample_with_indicators, merge_mtf
from btcusdt_algo.detectors.microtrigger import micro_triggers
from btcusdt_algo.strategies.ensemble import route_by_regime
from btcusdt_algo.backtest.metrics import perf_headline, breakdowns


# ============================================================
# Helpers
# ============================================================

def _fallback_side(row: pd.Series) -> str:
    """전략이 중립일 때 간단한 히유리스틱(추세 편향)."""
    ema = row.get("ema21", np.nan)
    mh  = row.get("macd_hist", np.nan)
    close = row.get("close", np.nan)
    regime = str(row.get("regime", "")).lower()

    if regime == "trend":
        if pd.notna(mh) and mh > 0: return "long"
        if pd.notna(mh) and mh < 0: return "short"
        if pd.notna(ema) and pd.notna(close):
            return "long" if close >= ema else "short"

    if regime == "range":
        hi = row.get("bb_high", np.nan); lo = row.get("bb_low", np.nan)
        if pd.notna(hi) and pd.notna(lo) and pd.notna(close) and hi > lo:
            pos = (close - lo) / max(hi - lo, 1e-8)
            if pos <= 0.25: return "long"
            if pos >= 0.75: return "short"
    return ""


def _macd_deadband_threshold(row: pd.Series, f: Dict[str, Any]) -> float:
    """
    MACD histogram deadband:
      - 'abs'          : |mh| > deadband
      - 'pct_of_close' : |mh| > pct * close
      - 'pct_of_atr'   : |mh| > pct * atr
    """
    mode = str(f.get("macd_deadband_mode", "abs")).lower()
    if mode == "abs":
        return float(f.get("macd_deadband", 0.0))

    pct = float(f.get("macd_deadband_pct", 0.0))
    if pct <= 0:
        return 0.0

    if mode == "pct_of_close":
        base = row.get("close", None)
    elif mode == "pct_of_atr":
        base = row.get("atr", None)
    else:
        base = None

    if base is None or (isinstance(base, float) and not np.isfinite(base)):
        return 0.0
    return abs(pct) * float(base)


# -------------------------
# MTF driver utilities
# -------------------------
def _default_driver_cfg(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Graceful defaults; 'entries' 섹션도 자동 반영."""
    d = (settings or {}).get("mtf_driver", {}) or {}
    e = (settings or {}).get("entries", {}) or {}

    decision_tf = str(e.get("decision_tf", d.get("decision_tf", "15m")))
    align_mode  = str(e.get("align_mode", d.get("align_mode", "scan"))).lower()
    if align_mode not in ("scan", "strict"):
        align_mode = "scan"
    # strict → 바로 다음 1분봉만 허용(≈1분)
    # scan   → 확정 후 scan_window_mins 동안 허용
    win_mins = int(d.get("entry_window_minutes",
                         (1 if align_mode == "strict" else int(e.get("scan_window_mins", 8)))))

    return {
        "enable":                  bool(d.get("enable", True)),
        "decision_tf":             decision_tf,
        "confirm_on_close":        True,  # 15m 종가 확정 기준
        "entry_window_minutes":    max(1, win_mins),
        "rsi_min":                 float(d.get("rsi_min", settings.get("filters", {}).get("mtf_rsi_min", 55.0))),
        "use_macd":                bool(d.get("use_macd", True)),
        "macd_deadband_mode":      str(d.get("macd_deadband_mode", settings.get("filters", {}).get("macd_deadband_mode", "pct_of_atr"))),
        "macd_deadband":           float(d.get("macd_deadband", settings.get("filters", {}).get("macd_deadband", 0.0))),
        "macd_deadband_pct":       float(d.get("macd_deadband_pct", settings.get("filters", {}).get("macd_deadband_pct", 0.08))),
        "ema_bias":                bool(d.get("ema_bias", True)),
        "require_higher_agree":    list(d.get("require_higher_agree", ["1h"])),  # ["1h","4h"] 등
        "higher_rsi_min":          float(d.get("higher_rsi_min", 52.0)),
        "higher_ema_slope_min":    float(d.get("higher_ema_slope_min", 0.0002)),
    }


def _mk_driver_15m(df_15m: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    15m 종가 확정 바에서 방향(1/−1/0)을 산출.
    long:  rsi>=min & (close>=ema21 if bias) & (macd_hist > deadband if use_macd)
    short: rsi<=100-min & (close<=ema21 if bias) & (macd_hist < -deadband)
    """
    d = df_15m.copy()
    need = ["timestamp", "close"]
    for c in need:
        if c not in d.columns:
            raise KeyError(f"15m driver missing column: {c}")

    rsi_min = float(cfg["rsi_min"])
    use_macd = bool(cfg["use_macd"])
    mode = str(cfg["macd_deadband_mode"]).lower()
    dead_abs = float(cfg["macd_deadband"])
    dead_pct = float(cfg["macd_deadband_pct"])
    ema_bias = bool(cfg["ema_bias"])

    # deadband 준비
    if use_macd:
        if mode == "abs":
            thr_pos, thr_neg = abs(dead_abs), -abs(dead_abs)
        elif mode == "pct_of_atr":
            base = d["atr"] if "atr" in d.columns else d["close"]
            thr = base.astype(float) * abs(dead_pct)
            thr_pos, thr_neg = thr, -thr
        else:  # pct_of_close
            thr = d["close"].astype(float) * abs(dead_pct)
            thr_pos, thr_neg = thr, -thr
    else:
        thr_pos = 0.0
        thr_neg = 0.0

    rsi = d.get("rsi")
    ema = d.get("ema21")
    mh  = d.get("macd_hist")

    cond_long = (rsi >= rsi_min)
    cond_short= (rsi <= 100.0 - rsi_min)

    if ema_bias and "ema21" in d.columns:
        cond_long &= (d["close"] >= ema)
        cond_short&= (d["close"] <= ema)

    if use_macd and "macd_hist" in d.columns:
        cond_long &= (mh > thr_pos)
        cond_short&= (mh < thr_neg)

    side = np.where(cond_long, 1, np.where(cond_short, -1, 0)).astype(int)
    out = d[["timestamp"]].copy()
    out["driver_side"] = side
    return out


def _merge_driver_and_htf(df_1m: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    """
    15m driver + 1h/4h 지표를 1m 타임라인에 병합.
    """
    dcfg = _default_driver_cfg(settings)
    if not dcfg["enable"]:
        return df_1m

    # 15m/1h/4h 인디케이터 확보 (core.mtf.resample_with_indicators가 확장 컬럼 지원)
    ind_cols_15m = ("rsi","ema21","macd_hist","adx","ema21_slope","atr","close")
    ind_cols_1h4h = ("rsi","ema21","macd_hist","ema21_slope","close")

    d15 = resample_with_indicators(df_1m, dcfg["decision_tf"], settings, cols=ind_cols_15m)
    d1h = resample_with_indicators(df_1m, "1h", settings, cols=ind_cols_1h4h)
    d4h = resample_with_indicators(df_1m, "4h", settings, cols=ind_cols_1h4h)

    # 15m driver 로직
    driver15 = _mk_driver_15m(d15, dcfg)

    # Merge to 1m
    out = df_1m.copy()
    out = merge_mtf(out, d15, suffix="15m")
    out = merge_mtf(out, d1h, suffix="1h")
    out = merge_mtf(out, d4h, suffix="4h")

    driver15_ren = driver15.rename(columns={"driver_side": "driver_side_15m"})
    out = pd.merge_asof(out.sort_values("timestamp"),
                        driver15_ren.sort_values("timestamp"),
                        on="timestamp", direction="backward")

    # 최근 15m 종가로부터 경과 분
    t15 = d15[["timestamp"]].copy().rename(columns={"timestamp": "t15"})
    out = pd.merge_asof(out.sort_values("timestamp"),
                        t15.sort_values("t15"),
                        left_on="timestamp", right_on="t15", direction="backward")
    out["mins_since_t15"] = ((out["timestamp"] - out["t15"]).dt.total_seconds() / 60.0).clip(lower=0)
    out.drop(columns=["t15"], inplace=True)

    return out


def _driver_allows_entry(row: pd.Series, settings: Dict[str, Any], desired_side: str) -> Tuple[bool, str]:
    """
    Driver 게이트:
    - 15m driver_side 일치
    - entry window minutes 내
    - (선택) 1h/4h 동조
    """
    dcfg = _default_driver_cfg(settings)
    if not dcfg["enable"]:
        return True, ""

    drv = int(row.get("driver_side_15m", 0))
    if desired_side == "long" and drv != 1:
        return False, "driver_block"
    if desired_side == "short" and drv != -1:
        return False, "driver_block"

    win = int(max(1, dcfg["entry_window_minutes"]))
    mins = float(row.get("mins_since_t15", 1e9))
    if np.isfinite(mins) and mins > win:
        return False, "driver_window_block"

    req = [str(x).lower() for x in dcfg.get("require_higher_agree", [])]
    if req:
        rsi_min = float(dcfg.get("higher_rsi_min", 52.0))
        slope_min = float(dcfg.get("higher_ema_slope_min", 0.0002))

        def _agrees(tf: str) -> bool:
            rsi = row.get(f"rsi_{tf}")
            ema_slope = row.get(f"ema21_slope_{tf}")
            ok_rsi = True
            ok_slope = True
            if pd.notna(rsi):
                ok_rsi = (rsi >= rsi_min) if desired_side == "long" else (rsi <= 100.0 - rsi_min)
            if pd.notna(ema_slope):
                ok_slope = (ema_slope >= slope_min) if desired_side == "long" else (ema_slope <= -slope_min)
            return bool(ok_rsi and ok_slope)

        for tf in req:
            if tf not in ("1h","4h"):
                continue
            if not _agrees(tf):
                return False, f"htf_{tf}_block"

    return True, ""


# -------------------------
# Hard entry gate (legacy + tweaks)
# -------------------------
def _passes_entry_filters(row: pd.Series, settings: Dict[str, Any], side: str) -> Tuple[bool, str]:
    f = settings.get("filters", {}) or {}

    # 0) 세션 블락
    sess_cfg = settings.get("session", {}) or {}
    blocked = set(sess_cfg.get("block", []) or [])
    if blocked:
        sess = which_session(row["timestamp"].to_pydatetime())
        if sess in blocked:
            return False, "blocked_session"

    regime = str(row.get("regime", "")).lower()

    # 1) Regime allowlist
    allow = {str(x).lower() for x in (f.get("trade_in_regimes", []) or [])}
    if allow and regime not in allow:
        return False, "blocked_regime"

    # 2) Volume-spike gate
    vgate = f.get("volume_spike_gate", {})
    if bool(vgate.get("enable", False)):
        apply_in = {str(x).lower() for x in (vgate.get("apply_in", ["trend"]) or [])}
        if regime in apply_in:
            mult = float(vgate.get("mult", 1.5))
            vol = row.get("volume", None); vma = row.get("vol_ma", None)
            if vol is not None and vma is not None and vma > 0:
                if float(vol) < mult * float(vma):
                    return False, "vol_spike_block"

    # 3) EMA slope 방향 일치
    egate = f.get("ema_slope_gate", {})
    if bool(egate.get("enable", False)):
        apply_in = {str(x).lower() for x in (egate.get("apply_in", ["trend"]) or [])}
        if regime in apply_in:
            slope = row.get("ema_slope", None)
            if slope is not None and np.isfinite(float(slope)):
                min_s = float(egate.get("min", settings.get("regime", {}).get("ema_slope_min", 0.0006)))
                s = float(slope)
                if side == "long":
                    if s < min_s:
                        return False, "ema_slope_block"
                else:
                    if s > -min_s:
                        return False, "ema_slope_block"

    # 4) (옵션) Extreme RSI block — 과열 직후 금지
    xcfg = f.get("extreme_rsi_block", {}) or {}
    if bool(xcfg.get("enable", False)):
        # micro_triggers가 만들어주는 최근 극점 플래그 사용
        peak_recent   = row.get("rsi_peak_recent",  False)
        trough_recent = row.get("rsi_trough_recent",False)
        # 없다면 관대하게 통과 (zero-trade 방지)
        if side == "long" and bool(peak_recent):
            return False, "extreme_rsi_block"
        if side == "short" and bool(trough_recent):
            return False, "extreme_rsi_block"

    # 5) MTF RSI/EMA 게이트(레거시)
    if f.get("require_mtf", True):
        apply_in = {str(x).lower() for x in f.get("mtf_apply_in", ["trend"]) }
        if regime in apply_in:
            rsi_min = float(f.get("mtf_rsi_min", 52.0))
            mode = str(f.get("mtf_rsi_mode", "and")).lower()
            r5, r15 = row.get("rsi_5m"), row.get("rsi_15m")
            e5, e1 = row.get("ema21_5m"), row.get("ema21")
            close = row.get("close")

            if mode == "and":
                if side == "long":
                    if (r5 is not None and r5 < rsi_min) or (r15 is not None and r15 < rsi_min):
                        return False, "mtf_rsi_block"
                else:
                    if (r5 is not None and r5 > 100 - rsi_min) or (r15 is not None and r15 > 100 - rsi_min):
                        return False, "mtf_rsi_block"
            else:  # or
                flags = []
                if r5 is not None:
                    flags.append(r5 >= rsi_min if side == "long" else r5 <= 100 - rsi_min)
                if r15 is not None:
                    flags.append(r15 >= rsi_min if side == "long" else r15 <= 100 - rsi_min)
                if flags and not any(flags):
                    return False, "mtf_rsi_block"

            if f.get("mtf_use_ema_gate", True):
                if side == "long":
                    if (e5 is not None and close < e5) and (e1 is not None and close < e1):
                        return False, "mtf_ema_block"
                else:
                    if (e5 is not None and close > e5) and (e1 is not None and close > e1):
                        return False, "mtf_ema_block"

    # 6) MACD histogram sign + deadband
    if f.get("require_macd_hist_sign", True):
        macd_apply_in = {str(x).lower() for x in f.get("macd_apply_in", ["trend"]) }
        if regime in macd_apply_in:
            mh = row.get("macd_hist", None)
            if mh is not None:
                dead = _macd_deadband_threshold(row, f)
                if abs(float(mh)) > dead:
                    if side == "long" and mh < 0:  return False, "macd_sign_block"
                    if side == "short" and mh > 0: return False, "macd_sign_block"

    # 7) Range guard
    if regime == "range":
        bbw = row.get("bb_width", None)
        if bbw is not None:
            bbw_min = float(f.get("range_bbw_min", 0.0010))
            bbw_max = float(f.get("range_bbw_max", 0.0040))
            if bbw < bbw_min or bbw > bbw_max:
                return False, "range_bbw_block"
        close = float(row.get("close", 0.0)) or 1.0
        atrp  = float(row.get("atr", 0.0)) / close
        if atrp > float(f.get("range_atr_pct_max", 0.0040)):
            return False, "range_atr_block"

    return True, ""


# ============================================================
# Main backtest
# ============================================================
def run_backtest(data_path: str, settings: Dict[str, Any], score_override: float | None = None) -> Dict[str, Any]:
    """Deterministic backtest; artifacts 항상 저장."""
    # 기본 게이트
    settings.setdefault("filters", {
        "require_mtf": True,
        "mtf_rsi_min": 52.0,
        "mtf_use_ema_gate": True,
        "require_macd_hist_sign": True,
    })

    io_cfg = settings.get("io", {}) or {}
    write_debug_json = bool(io_cfg.get("write_debug_json", True))
    debug_example_limit = int(io_cfg.get("debug_limit_examples", 10))

    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return {}

    df = pd.read_parquet(data_path)
    if df.empty:
        logger.error("Empty data")
        return {}

    # timestamp
    if "timestamp" not in df.columns:
        logger.error("Input must contain 'timestamp' column")
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1) Indicators (1m)
    df = add_indicators(df, settings)

    # EMA slope (상대기울기) for gate/diag
    reg_cfg = settings.get("regime", {}) or {}
    lb = int(reg_cfg.get("ema_slope_lookback", 5))
    if "ema21" in df.columns:
        prior = df["ema21"].shift(lb)
        slope = (df["ema21"] - prior) / (prior.abs().replace(0, np.nan))
        df["ema_slope"] = slope.replace([np.inf, -np.inf], np.nan)

    # 2) MTF merges
    d5  = resample_with_indicators(df, "5min",  settings, cols=("rsi", "ema21"))
    d15 = resample_with_indicators(df, "15min", settings, cols=("rsi","ema21","macd_hist","adx","ema21_slope","atr","close"))
    df = merge_mtf(df, d5,  "5m")
    df = merge_mtf(df, d15, "15m")

    # 15m driver + 1h/4h
    df = _merge_driver_and_htf(df, settings)

    # 3) Regime
    ind_cfg = settings.get("indicators", {}) or {}
    df["regime"] = classify_regime(
        df,
        reg_cfg.get("trend_adx_min", 22),
        ind_cfg.get("bb_squeeze_th", 0.028),
        reg_cfg.get("range_bb_width_max", 0.00165),
        reg_cfg.get("ema_slope_lookback", 5),
        reg_cfg.get("ema_slope_min", 0.0007),
    )

    # 4) Features + detectors
    df = build_features(df, use_detectors=True, cfg=settings)

    # 5) Micro triggers (+ RSI-extrema 플래그)
    micro_cfg = settings.get("micro", {}) or {}
    micro = micro_triggers(
        df,
        lookback_break=int(micro_cfg.get("break_lookback", 8)),
        rsi_on=float(micro_cfg.get("rsi_on", 55.0)),
        mode=str(micro_cfg.get("mode", "pierce")),
        use_regime_gate=bool(micro_cfg.get("use_regime_gate", True)),
        regime_col=str(micro_cfg.get("regime_col", "regime")),
        use_mtf_align=bool(micro_cfg.get("use_mtf_align", False)),
        mtf_rsi_min=float(micro_cfg.get("mtf_rsi_min", 55.0)),
        mtf_mode=str(micro_cfg.get("mtf_mode", "and")),
        use_ema_align=bool(micro_cfg.get("use_ema_align", False)),
        ema_col=str(micro_cfg.get("ema_col", "ema21")),
        min_atr_pct=float(micro_cfg.get("min_atr_pct", 0.0)),
        max_atr_pct=float(micro_cfg.get("max_atr_pct", 0.0)),
        vol_spike_mult=float(micro_cfg.get("vol_spike_mult", 0.0)),
        vol_ma_len=int(micro_cfg.get("vol_ma_len", 50)),
        darvas_confirm=bool(micro_cfg.get("darvas_confirm", False)),
        cooloff_bars=int(micro_cfg.get("cooloff_bars", 0)),
        rsi_extrema=micro_cfg.get("rsi_extrema", {"enable": True, "peak_thr": 70.0, "trough_thr": 30.0, "window": 3, "require_sod": True})
    )
    df = pd.concat([df, micro], axis=1)

    # 6) Scorer + PositionManager
    w = settings.get("scoring_weights", {
        "volatility": 0.25, "momentum": 0.25, "volume": 0.20, "structure": 0.20, "fib": 0.10
    })
    scoring_cfg = settings.get("scoring", {}) or {}
    bonuses = scoring_cfg.get("detector_bonuses", settings.get("detector_bonuses", {}))
    scorer = SignalScorer(
        weights=w,
        squeeze_th=ind_cfg.get("bb_squeeze_th", 0.028),
        bonuses=bonuses,
        logistic_cap=bool(scoring_cfg.get("logistic_cap", True)),
        logistic_mid=float(scoring_cfg.get("logistic_mid", 60.0)),
        logistic_k=float(scoring_cfg.get("logistic_k", 12.0)),
    )

    rconf = settings.get("risk", {}) or {}
    tconf = settings.get("trailing", {}) or {}
    pconf = settings.get("position", {}) or {}

    fees = rconf.get("fees_bps_per_side", 6)
    assert isinstance(fees, (int, float)) and fees < 100, "fees_bps_per_side uses bps (6=0.06%)"

    pm = PositionManager(
        rr=tuple(rconf.get("rr_base", [1, 3])),
        atr_sl_mult=float(rconf.get("atr_sl_mult_trend", 1.6)),  # 레짐별로 엔트리 시점에 업데이트
        trail_conf=tconf,
        max_hold_minutes=int(rconf.get("max_hold_minutes", 360)),
        fees_bps_per_side=fees,
        add_on_max=int(pconf.get("add_on_max", 0)),
        add_on_trigger_R=float(pconf.get("add_on_trigger_R", 0.8)),
        add_on_size_pct=float(pconf.get("add_on_size_pct", 33)),
        ladder_scheme=pconf.get("ladder", None),
        be_at_R=float(pconf.get("be_at_R", 1.5)),
        partials=pconf.get("partials", []),
    )

    # 7) Thresholding
    df["score_raw"] = df.apply(lambda r: scorer.score_row(r, session_mult=1.0), axis=1)
    th_cfg = settings.get("thresholding", {}) or {}
    compare_mode = str(th_cfg.get("compare", "session_scaled")).lower()
    use_session_mult_for_score = (compare_mode == "session_scaled")

    if score_override is not None:
        df["thr"] = float(score_override)
    elif th_cfg.get("mode", "adaptive") == "adaptive":
        df["thr"] = adaptive_threshold(
            df["score_raw"],
            window_bars=int(th_cfg.get("window_bars", 4320)),
            pct=float(th_cfg.get("adaptive_percentile", 0.90)),
            floor=float(th_cfg.get("floor", 75.0)),
        )
    else:
        df["thr"] = float(th_cfg.get("fixed_score_th", 80.0))

    trades: List[dict] = []
    cooldown = int(settings.get("entries", {}).get("cooldown_bars", 120))
    cool_left = 0

    daily_limit_R = float(rconf.get("daily_loss_limit_R", np.inf))
    block_until_next_day = False

    # debug counters/samples
    dbg = {
        "cand": 0, "blocked_session": 0, "blocked_regime": 0, "no_side": 0,
        "micro_fail": 0, "mtf_rsi_block": 0, "mtf_ema_block": 0, "macd_sign_block": 0,
        "vol_spike_block": 0, "ema_slope_block": 0, "extreme_rsi_block": 0,
        "range_bbw_block": 0, "range_atr_block": 0,
        "driver_block": 0, "driver_window_block": 0, "htf_1h_block": 0, "htf_4h_block": 0,
        "gate_fail": 0, "entered": 0,
    }
    entered_samples: list[dict] = []
    skipped_samples: list[dict] = []

    def _realized_R_sum_for_day(day: pd.Timestamp) -> float:
        if not trades: return 0.0
        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        s = 0.0
        for t in trades:
            ttime = t.get("time") or t.get("exit") or t.get("entry")
            if not ttime: continue
            if pd.Timestamp(ttime).strftime("%Y-%m-%d") == day_str and (t.get("partial") or t.get("exit")):
                try: s += float(t.get("R", 0.0))
                except Exception: pass
        return s

    # ---------------- Loop ----------------
    for idx, row in df.iterrows():
        # essential fields
        if pd.isna(row.get("rsi")) or pd.isna(row.get("atr")) or pd.isna(row.get("bb_mid")) or pd.isna(row.get("bb_width")):
            continue

        now = row["timestamp"].to_pydatetime()
        price = float(row["close"]); rsi = float(row["rsi"]); atr = float(row["atr"])

        # daily loss block
        if pm.flat():
            curR = _realized_R_sum_for_day(pd.Timestamp(now))
            block_until_next_day = (curR <= -abs(daily_limit_R))

        # manage open pos
        if not pm.flat():
            pm.update_trailing(price, rsi, atr)
            for p in pm.try_maintenance(price, now):
                trades.append(p)

            soft = pm.maybe_soft_tp(price, now, rsi, row.get("macd_hist", 0.0), settings.get("soft_tp", {}))
            if soft:
                trades.append(soft)
                if pm.flat():
                    cool_left = cooldown
                    continue

            closed = pm.try_close(price, now, rsi, atr)
            if closed:
                trades.append(closed)
                if pm.flat():
                    cool_left = cooldown
                continue

        # cooldown
        if cool_left > 0:
            cool_left -= 1
            continue

        # entry
        if pm.flat() and not block_until_next_day:
            dbg["cand"] += 1

            score_raw = float(row.get("score_raw", 0.0))
            thr_val   = float(row.get("thr", 80.0))
            s_mult = float(score_mult(settings, now)) if use_session_mult_for_score else 1.0

            score_eff = score_raw * s_mult
            df.at[idx, "score_eff"] = score_eff
            df.at[idx, "score_mult"] = s_mult

            if score_eff >= thr_val:
                # (1) 전략 라우팅
                strat = route_by_regime(row["regime"])
                side  = "long" if strat.should_long(row) else ("short" if strat.should_short(row) else "")
                if not side:
                    side = _fallback_side(row)
                    if not side:
                        dbg["no_side"] += 1
                        if len(skipped_samples) < debug_example_limit:
                            skipped_samples.append({
                                "ts": str(row["timestamp"]), "why": "no_side",
                                "regime": row.get("regime"), "side": "",
                                "score_raw": score_raw, "thr": thr_val
                            })
                        continue

                # (2) 15m driver gate
                ok_drv, why_drv = _driver_allows_entry(row, settings, side)
                if not ok_drv:
                    dbg[why_drv] = dbg.get(why_drv, 0) + 1
                    dbg["gate_fail"] += 1
                    if len(skipped_samples) < debug_example_limit:
                        skipped_samples.append({
                            "ts": str(row["timestamp"]), "why": why_drv,
                            "regime": row.get("regime"), "side": side,
                            "score_raw": score_raw, "thr": thr_val
                        })
                    continue

                # (3) micro confirm
                reg_str = str(row.get("regime", "")).lower()
                if reg_str == "range":
                    hi, lo = row.get("bb_high"), row.get("bb_low")
                    ok_micro = False
                    if pd.notna(hi) and pd.notna(lo) and hi > lo and pd.notna(row.get("close")):
                        pos = (row["close"] - lo) / max(hi - lo, 1e-8)
                        rsi_val = row.get("rsi")
                        fcfg = settings.get("filters", {}) or {}
                        e_lo   = float(fcfg.get("range_edge_low", 0.25))
                        e_hi   = float(fcfg.get("range_edge_high", 0.75))
                        r_long_max  = float(fcfg.get("range_rsi_long_max", 43))
                        r_short_min = float(fcfg.get("range_rsi_short_min", 57))
                        if side == "long":
                            ok_micro = (pos <= e_lo) and pd.notna(rsi_val) and (rsi_val <= r_long_max)
                        else:
                            ok_micro = (pos >= e_hi) and pd.notna(rsi_val) and (rsi_val >= r_short_min)
                else:
                    ok_micro = bool(row.get("micro_long", False) if side == "long" else row.get("micro_short", False))

                if not ok_micro:
                    dbg["micro_fail"] += 1
                    if len(skipped_samples) < debug_example_limit:
                        skipped_samples.append({
                            "ts": str(row["timestamp"]), "why": "micro_fail",
                            "regime": row.get("regime"), "side": side,
                            "score_raw": score_raw, "thr": thr_val
                        })
                    continue

                # (4) 레거시 하드 게이트 + Extreme RSI block
                ok, reason = _passes_entry_filters(row, settings, side)
                if not ok:
                    dbg[reason] = dbg.get(reason, 0) + 1
                    dbg["gate_fail"] += 1
                    if len(skipped_samples) < debug_example_limit:
                        skipped_samples.append({
                            "ts": str(row["timestamp"]), "why": reason,
                            "regime": row.get("regime"), "side": side,
                            "score_raw": score_raw, "thr": thr_val
                        })
                    continue

                # === ENTER ===
                if row["regime"] == "trend":
                    pm.rr = tuple(rconf.get("rr_trend", rconf.get("rr_base", [1, 3])))
                    pm.atr_sl_mult = float(rconf.get("atr_sl_mult_trend", 1.6))
                    pm.trail_conf["atr_trail_mult"] = float(tconf.get("atr_trail_mult_trend", 1.6))
                else:
                    pm.rr = tuple(rconf.get("rr_range", rconf.get("rr_base", [1, 3])))
                    pm.atr_sl_mult = float(rconf.get("atr_sl_mult_range", 0.9))
                    pm.trail_conf["atr_trail_mult"] = float(tconf.get("atr_trail_mult_range", 1.2))

                # 세션 리스크 보정
                sess = which_session(now)
                sess_adj = settings.get("session", {}).get("risk_adj", {}).get(sess, 1.0)
                try: pm.atr_sl_mult *= float(sess_adj)
                except Exception: pass

                pm.open_position(
                    side, price, atr, now, size=1.0,
                    regime=row["regime"],
                    strategy=type(strat).__name__,
                    session=sess,
                )
                dbg["entered"] += 1
                if len(entered_samples) < debug_example_limit:
                    entered_samples.append({
                        "ts": str(row["timestamp"]), "side": side, "regime": row.get("regime"),
                        "score_raw": score_raw, "thr": thr_val, "score_eff": score_eff
                    })
                cool_left = cooldown

    # ---------------- loop end ----------------
    # flush last position
    if not pm.flat():
        last = df.iloc[-1]
        now  = last["timestamp"].to_pydatetime()
        price= float(last["close"]); rsi = float(last["rsi"]); atr = float(last["atr"])

        pm.update_trailing(price, rsi, atr)
        for p in pm.try_maintenance(price, now):
            trades.append(p)

        soft = pm.maybe_soft_tp(price, now, rsi, last.get("macd_hist", 0.0), settings.get("soft_tp", {}))
        if soft: trades.append(soft)

        closed = pm.try_close(price, now, rsi, atr)
        if closed: trades.append(closed)

    # reporting
    headline = perf_headline(trades)
    by = breakdowns(trades)

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = os.path.abspath(f"logs/last_report_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"headline": headline, "breakdowns": by}, f, ensure_ascii=False, indent=2)

    cols = ["time","entry","exit","side","R","reason","partial",
            "strategy","regime","session","mfe_R","mae_R","partial_pct"]
    df_trades = pd.DataFrame(trades)
    for c in cols:
        if c not in df_trades.columns: df_trades[c] = pd.NA
    csv_path = os.path.abspath(f"logs/trades_{ts}.csv")
    df_trades.to_csv(csv_path, index=False)

    logger.info(f"[ENGINE SAVE] trades CSV: {csv_path} rows={len(df_trades)}")
    logger.info(f"[ENGINE SAVE] report JSON: {json_path}")

    if write_debug_json:
        reason_dist = Counter(t.get("reason","") for t in trades if t.get("reason"))
        side_dist   = Counter(t.get("side","")   for t in trades if t.get("side"))
        regime_dist = Counter(t.get("regime","") for t in trades if t.get("regime"))

        cfg = {
            "filters":      settings.get("filters", {}),
            "thresholding": settings.get("thresholding", {}),
            "risk":         settings.get("risk", {}),
            "trailing":     settings.get("trailing", {}),
            "position":     settings.get("position", {}),
            "session":      settings.get("session", {}),
            "micro":        settings.get("micro", {}),
            "regime":       settings.get("regime", {}),
            "indicators":   settings.get("indicators", {}),
            "mtf_driver":   _default_driver_cfg(settings),
        }

        dbg_payload = {
            "summary": headline,
            "dbg": dbg,
            "entered_examples": entered_samples[:debug_example_limit],
            "skipped_examples": skipped_samples[:debug_example_limit],
            "dists": {"reason": dict(reason_dist), "regime": dict(regime_dist), "side": dict(side_dist)},
            "cfg_filters":      cfg["filters"],
            "cfg_thresholding": cfg["thresholding"],
            "cfg_risk":         cfg["risk"],
            "cfg_trailing":     cfg["trailing"],
            "cfg_position":     cfg["position"],
            "cfg_session":      cfg["session"],
            "cfg_micro":        cfg["micro"],
            "cfg_regime":       cfg["regime"],
            "cfg_indicators":   cfg["indicators"],
            "cfg_mtf_driver":   cfg["mtf_driver"],
            "compare_mode":     compare_mode,
            "paths": {"csv": csv_path, "json": json_path},
        }

        dbg_path = os.path.abspath(f"logs/last_debug_{ts}.json")
        with open(dbg_path, "w", encoding="utf-8") as f:
            json.dump(dbg_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"[ENGINE DEBUG] debug JSON: {dbg_path}")

    report = dict(headline)
    report["details_saved"] = True
    return report
