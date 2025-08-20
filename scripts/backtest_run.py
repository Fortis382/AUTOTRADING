# scripts/backtest_run.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml

# src 경로 추가
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btcusdt_algo.backtest.engine import run_backtest

def main():
    ap = argparse.ArgumentParser(description="Smoke backtest runner")
    ap.add_argument("--data", required=True, help="Parquet path with columns: timestamp, open, high, low, close, volume")
    ap.add_argument("--settings", required=True, help="settings.yaml path")
    ap.add_argument("--score-override", type=float, default=None, help="Override threshold (for quick tests)")
    args = ap.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    cfg_path = Path(args.settings).expanduser().resolve()
    assert data_path.exists(), f"Data not found: {data_path}"
    assert cfg_path.exists(), f"Settings not found: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}

    report = run_backtest(str(data_path), settings, score_override=args.score_override)

    print("\n=== HEADLINE ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n(참고) logs/ 디렉토리에 CSV/JSON/DEBUG JSON이 생성됩니다.")
    print(" - trades_*.csv : 체결 시퀀스(+partials)")
    print(" - last_report_*.json : headline + breakdowns")
    print(" - last_debug_*.json : 디버그 카운터/예시/설정 스냅샷")

if __name__ == "__main__":
    main()
