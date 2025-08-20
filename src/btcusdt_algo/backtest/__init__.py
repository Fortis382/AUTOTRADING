from __future__ import annotations
import argparse, os, time
from pathlib import Path
from loguru import logger
import yaml

def _resolve_cfg(user_cfg: str|None) -> str:
    if user_cfg:
        p = Path(user_cfg)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return str(p)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "config" / "settings.yaml",  # <proj_root>/config/settings.yaml
        here.parents[2] / "config" / "settings.yaml",  # fallback
        Path.cwd() / "config" / "settings.yaml",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("settings.yaml not found. Tried: " + " | ".join(map(str, candidates)))

def _load_settings(cfg_path: str|None = None) -> dict:
    cfg_path = _resolve_cfg(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cli_download():
    # ← 여기서만 데이터 로더 임포트 (엔진/전략 안 불러옴)
    from btcusdt_algo.core.data_loader import download_days_segmented, download_klines_ccxt
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--step-days", type=int, default=10)
    p.add_argument("--out", default="data/processed/BTCUSDT_1m.parquet")
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.days <= a.step_days:
        logger.info(f"Downloading simple range: last {a.days} days")
        since_ms = int((time.time() - a.days*24*60*60)*1000)
        df = download_klines_ccxt(a.symbol, a.timeframe, since_ms)
    else:
        logger.info(f"Downloading segmented: {a.days} days in steps of {a.step_days} days")
        df = download_days_segmented(a.symbol, a.timeframe, a.days, a.step_days)

    if df.empty:
        logger.error("No data downloaded.")
        return
    df = df[["timestamp","open","high","low","close","volume"]]
    df.to_parquet(a.out, index=False)
    logger.info(f"Saved: {a.out} (rows={len(df)})")

def cli_backtest():
    # ← 백테스트 할 때만 엔진 임포트
    from .engine import run_backtest
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/BTCUSDT_1m.parquet")
    p.add_argument("--cfg", default=None, help="Path to settings.yaml (optional)")
    p.add_argument("--score-th", type=float, default=None, help="override threshold if set")
    a = p.parse_args()
    settings = _load_settings(a.cfg)
    report = run_backtest(a.data, settings, score_override=a.score_th)
    if report:
        print("==== Backtest Report ====")
        for k, v in report.items():
            print(f"{k:>15}: {v}")
        print("=========================")
