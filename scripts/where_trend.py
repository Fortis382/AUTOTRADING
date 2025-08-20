# scripts/where_trend.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# 프로젝트 소스 경로 추가 (…/src)
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 내부 모듈
from btcusdt_algo.core.indicators import add_indicators
from btcusdt_algo.core.regime import classify_regime

def load_settings(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def compute_ema_slope(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    if "ema21" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    prior = df["ema21"].shift(lookback)
    slope = (df["ema21"] - prior) / (prior.abs().replace(0, np.nan))
    return slope.replace([np.inf, -np.inf], np.nan)

def collapse_segments(ts: pd.Series, label: str = "trend") -> list[dict]:
    """
    ts: regime 시리즈 (값: 'trend'/'range')
    return: [{start, end, bars}, ...]   # label 구간들만
    """
    out = []
    cur = None
    for i, v in enumerate(ts.values):
        if v == label and cur is None:
            cur = {"start_idx": i}
        elif v != label and cur is not None:
            cur["end_idx"] = i - 1
            out.append(cur)
            cur = None
    if cur is not None:
        cur["end_idx"] = len(ts) - 1
        out.append(cur)
    return out

def main():
    ap = argparse.ArgumentParser(description="Find & export TREND regime ranges from 1m parquet")
    ap.add_argument("--data", required=True, help="path to 1m parquet (e.g. data/processed/BTCUSDT_1m.parquet)")
    ap.add_argument("--settings", required=True, help="settings.yaml path")
    ap.add_argument("--out", default="logs/trend_ranges.csv", help="output CSV path")
    ap.add_argument("--minbars", type=int, default=10, help="min bars per trend segment to keep")
    ap.add_argument("--head", type=int, default=10, help="print top-N longest segments")
    args = ap.parse_args()

    data_path = Path(args.data).resolve()
    cfg_path  = Path(args.settings).resolve()
    out_path  = Path(args.out).resolve()

    assert data_path.exists(), f"Data not found: {data_path}"
    assert cfg_path.exists(),  f"Settings not found: {cfg_path}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 엔진과 동일한 전처리
    df = pd.read_parquet(data_path)
    if "timestamp" not in df.columns:
        raise ValueError("Input must contain 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    settings = load_settings(cfg_path)
    ind_cfg  = settings.get("indicators", {}) or {}
    reg_cfg  = settings.get("regime", {}) or {}

    # 1) 인디케이터
    df = add_indicators(df, settings)

    # (선택) slope 컬럼도 만들어 둠 (게이트/요약용)
    lb = int(reg_cfg.get("ema_slope_lookback", 5))
    df["ema_slope"] = compute_ema_slope(df, lookback=lb)

    # 2) 레짐
    df["regime"] = classify_regime(
        df,
        reg_cfg.get("trend_adx_min", 22),
        ind_cfg.get("bb_squeeze_th", 0.028),
        reg_cfg.get("range_bb_width_max", 0.00165),
        reg_cfg.get("ema_slope_lookback", 5),
        reg_cfg.get("ema_slope_min", 0.0007),
    )

    # 3) 트렌드 구간 병합
    segs = collapse_segments(df["regime"], label="trend")

    rows = []
    for s in segs:
        i0, i1 = s["start_idx"], s["end_idx"]
        sub = df.iloc[i0:i1+1]
        if len(sub) < args.minbars:
            continue
        dur_min = (sub["timestamp"].iloc[-1] - sub["timestamp"].iloc[0]).total_seconds() / 60.0
        rows.append({
            "start": sub["timestamp"].iloc[0].isoformat(),
            "end":   sub["timestamp"].iloc[-1].isoformat(),
            "bars":  int(len(sub)),
            "minutes": round(dur_min, 2),
            "avg_bb_width": float(sub["bb_width"].mean(skipna=True)),
            "avg_adx": float(sub["adx"].mean(skipna=True)) if "adx" in sub.columns else np.nan,
            "avg_ema_slope": float(sub["ema_slope"].mean(skipna=True)),
            "close_chg_pct": float((sub["close"].iloc[-1]/sub["close"].iloc[0]-1.0)*100.0),
        })

    seg_df = pd.DataFrame(rows).sort_values(["bars","minutes"], ascending=[False, False]).reset_index(drop=True)
    seg_df.to_csv(out_path, index=False)

    # 4) 콘솔 요약
    total_bars = len(df)
    trend_bars = int((df["regime"] == "trend").sum())
    trend_pct  = trend_bars / max(1, total_bars) * 100.0

    print("\n=== TREND SUMMARY ===")
    print(f"total bars    : {total_bars}")
    print(f"trend bars    : {trend_bars} ({trend_pct:.2f}%)")
    print(f"n trend segs  : {len(seg_df)} (minbars>={args.minbars})")
    if not seg_df.empty:
        print("\nTop segments (by bars):")
        print(seg_df.head(args.head).to_string(index=False))
        print(f"\nSaved CSV -> {out_path}")

if __name__ == "__main__":
    main()
