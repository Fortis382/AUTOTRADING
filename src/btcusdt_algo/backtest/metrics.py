# src/btcusdt_algo/backtest/metrics.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd

EPS = 1e-12

# ------------- helpers -------------
def _to_df(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        cols = ["time","entry","exit","side","R","reason","partial","strategy","regime","session","mfe_R","mae_R","partial_pct"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(trades).copy()

    # 타입/결측 위생
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    for c in ["R","mfe_R","mae_R","partial_pct","entry","exit"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["side","reason","strategy","regime","session"]:
        if c in df:
            df[c] = df[c].astype(str).fillna("")

    if "partial" in df:
        df["partial"] = df["partial"].astype(bool)

    return df


def _equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """트레이드(이벤트) 순서 기준 누적 R 및 드로다운"""
    if df.empty:
        return pd.DataFrame(columns=["time","R","cumR","highwater","drawdown","underwater"])
    df = df.sort_values("time").reset_index(drop=True)
    r = pd.to_numeric(df["R"], errors="coerce").fillna(0.0)
    df["R"] = r
    df["cumR"] = r.cumsum()
    df["highwater"] = df["cumR"].cummax()
    df["drawdown"] = df["cumR"] - df["highwater"]  # 음수(또는 0)
    df["underwater"] = df["drawdown"] < -EPS
    return df


def _max_drawdown(equity: pd.DataFrame) -> Tuple[float, int, int]:
    """최대 드로다운 깊이와 그 시점 인덱스(시작/끝은 계산 불가시 동일 인덱스 반환)"""
    if equity.empty:
        return (0.0, -1, -1)
    dd = equity["drawdown"]
    if dd.isna().all():
        return (0.0, -1, -1)
    i = int(dd.idxmin())
    return (float(dd.min()), i, i)


def _max_dd_length(equity: pd.DataFrame) -> int:
    """가장 긴 언더워터(최고점 아래) 구간의 길이(이벤트 개수)"""
    if equity.empty or "underwater" not in equity:
        return 0
    uw = equity["underwater"].astype(bool).to_numpy()
    best = cur = 0
    for x in uw:
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _streaks(signs: pd.Series) -> Dict[str,int]:
    """+/- 시퀀스에서 최장 연승/연패 길이(0은 무시)"""
    best_win = best_lose = cur_win = cur_lose = 0
    for s in signs:
        if s > 0:
            cur_win += 1
            best_win = max(best_win, cur_win)
            cur_lose = 0
        elif s < 0:
            cur_lose += 1
            best_lose = max(best_lose, cur_lose)
            cur_win = 0
        else:
            # zero → streak 끊지 않음
            pass
    return {"win_streak_max": best_win, "lose_streak_max": best_lose}


def _profit_factor(win_sum: float, loss_sum: float) -> float:
    loss_sum = abs(loss_sum)
    if loss_sum < EPS:
        return float("inf") if win_sum > 0 else 0.0
    return win_sum / loss_sum


def _expectancy(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    return float(r.mean())


def _sharpe_per_trade(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if len(r) < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1)
    return float(mu / (sd + EPS))


def _kelly_fraction(r: pd.Series) -> float:
    """
    이벤트 단위 Kelly 근사:
      p = win_rate, b = avg_win/abs(avg_loss)
      f* = p - (1-p)/b  (b>0일 때만)
    과대추정 방지: [0, 1]로 캡
    """
    r = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    wins = r[r > 0]
    losses = r[r < 0]
    p = len(wins) / len(r) if len(r) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0  # 음수
    if avg_loss >= -EPS or avg_win <= EPS:
        return 0.0
    b = avg_win / abs(avg_loss)
    f = p - (1 - p) / (b + EPS)
    return float(max(0.0, min(1.0, f)))


def _by_group(df: pd.DataFrame, key: str) -> List[Dict[str, Any]]:
    if key not in df:
        return []
    out = []
    for k, g in df.groupby(key):
        r = pd.to_numeric(g["R"], errors="coerce").fillna(0.0)
        n = len(g)
        win = (r > 0).sum()
        loss = (r < 0).sum()
        win_sum = float(r[r > 0].sum())
        loss_sum = float(r[r < 0].sum())
        out.append({
            key: str(k),
            "n": int(n),
            "R_sum": float(r.sum()),
            "E_R": float(r.mean()) if n else 0.0,
            "win_rate": float(win / n) if n else 0.0,
            "profit_factor": _profit_factor(win_sum, loss_sum),
            "avg_win_R": float(r[r > 0].mean()) if win else 0.0,
            "avg_loss_R": float(r[r < 0].mean()) if loss else 0.0,
        })
    # 큰 순으로 정렬
    out.sort(key=lambda x: (x["R_sum"], x["n"]), reverse=True)
    return out


def _cross_group(df: pd.DataFrame, key1: str, key2: str) -> List[Dict[str, Any]]:
    """key1×key2 교차표(간단 성과)"""
    if key1 not in df or key2 not in df:
        return []
    out = []
    for (k1, k2), g in df.groupby([key1, key2]):
        r = pd.to_numeric(g["R"], errors="coerce").fillna(0.0)
        n = len(g)
        win = (r > 0).sum()
        loss = (r < 0).sum()
        win_sum = float(r[r > 0].sum())
        loss_sum = float(r[r < 0].sum())
        out.append({
            key1: str(k1),
            key2: str(k2),
            "n": int(n),
            "R_sum": float(r.sum()),
            "E_R": float(r.mean()) if n else 0.0,
            "win_rate": float(win / n) if n else 0.0,
            "profit_factor": _profit_factor(win_sum, loss_sum),
        })
    out.sort(key=lambda x: (x["R_sum"], x["n"]), reverse=True)
    return out


# ------------- public API -------------
def perf_headline(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    이벤트(부분청산 포함) 기준 헤드라인 성과.
    JSON 직렬화 가능한 dict 반환.
    기존 키는 유지 + 유용한 추가 키를 포함.
    """
    df = _to_df(trades)
    if df.empty:
        return {
            "n_events": 0, "R_sum": 0.0, "E_R": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_dd_R": 0.0, "trade_sharpe": 0.0,
            "daily_sharpe": 0.0, "best_R": 0.0, "worst_R": 0.0,
            "win_streak_max": 0, "lose_streak_max": 0,
            # 추가 지표(빈 경우 0/None)
            "median_R": 0.0, "std_R": 0.0, "skew_R": 0.0,
            "p05_R": 0.0, "p95_R": 0.0,
            "avg_win_R": 0.0, "avg_loss_R": 0.0,
            "kelly_frac": 0.0,
            "max_dd_len_events": 0,
            "n_days": 0,
        }

    r = pd.to_numeric(df["R"], errors="coerce").fillna(0.0)
    n = len(df)
    win = (r > 0).sum()
    loss = (r < 0).sum()
    win_sum = float(r[r > 0].sum())
    loss_sum = float(r[r < 0].sum())

    eq = _equity_curve(df)
    max_dd, _, _ = _max_drawdown(eq)
    max_dd_len = _max_dd_length(eq)

    # 일간 집계로 데일리 샤프 참고값
    if "time" in df and df["time"].notna().any():
        tmp = df.dropna(subset=["time"]).copy()
        tmp["date"] = tmp["time"].dt.strftime("%Y-%m-%d")
        daily_r = pd.to_numeric(tmp.groupby("date")["R"].sum(), errors="coerce").fillna(0.0)
        daily_sharpe = _sharpe_per_trade(daily_r)
        n_days = int(daily_r.shape[0])
    else:
        daily_sharpe = 0.0
        n_days = 0

    streak = _streaks(np.sign(r))

    # 분포/건전성 부가 지표
    median_R = float(r.median())
    std_R = float(r.std(ddof=1)) if n >= 2 else 0.0
    skew_R = float(r.skew()) if n >= 3 else 0.0
    p05_R = float(r.quantile(0.05)) if n >= 1 else 0.0
    p95_R = float(r.quantile(0.95)) if n >= 1 else 0.0
    avg_win_R = float(r[r > 0].mean()) if win else 0.0
    avg_loss_R = float(r[r < 0].mean()) if loss else 0.0
    kelly_frac = _kelly_fraction(r)

    return {
        "n_events": int(n),
        "R_sum": float(r.sum()),
        "E_R": float(r.mean()),
        "win_rate": float(win / n),
        "profit_factor": _profit_factor(win_sum, loss_sum),
        "max_dd_R": float(max_dd),                   # 누적 R 기준 최대DD(음수)
        "trade_sharpe": _sharpe_per_trade(r),        # 이벤트 기준
        "daily_sharpe": float(daily_sharpe),         # 일자 합산 기준
        "best_R": float(r.max() if n else 0.0),
        "worst_R": float(r.min() if n else 0.0),
        "win_streak_max": streak["win_streak_max"],
        "lose_streak_max": streak["lose_streak_max"],
        # 추가 지표
        "median_R": median_R,
        "std_R": std_R,
        "skew_R": skew_R,
        "p05_R": p05_R,
        "p95_R": p95_R,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "kelly_frac": kelly_frac,                    # 0.0~1.0
        "max_dd_len_events": int(max_dd_len),        # 최장 언더워터 길이(이벤트)
        "n_days": n_days,
    }


def breakdowns(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    레짐/세션/사이드/전략/사유/시간대/일자 브레이크다운과
    MFE/MAE 요약 + 일자별 PnL 테이블(JSON 친화형) 반환
    + 요일/월/교차표/분포 분위수 추가
    """
    df = _to_df(trades)
    if df.empty:
        return {
            "by_regime": [], "by_session": [], "by_side": [], "by_strategy": [], "by_reason": [],
            "by_hour": [], "by_date": [], "by_dow": [], "by_month": [], "by_regime_session": [],
            "mfe_mae": {}, "equity_tail": [], "quantiles": {}
        }

    out = {
        "by_regime":   _by_group(df, "regime"),
        "by_session":  _by_group(df, "session"),
        "by_side":     _by_group(df, "side"),
        "by_strategy": _by_group(df, "strategy"),
        "by_reason":   _by_group(df, "reason"),
    }

    # 시간대(UTC) / 일자 / 요일 / 월 브레이크다운
    if "time" in df and df["time"].notna().any():
        tmp = df.dropna(subset=["time"]).copy()

        tmp["hour_utc"] = tmp["time"].dt.hour
        out["by_hour"] = _by_group(tmp.rename(columns={"hour_utc":"hour"}), "hour")

        tmp["date"] = tmp["time"].dt.strftime("%Y-%m-%d")
        daily = tmp.groupby("date")["R"].agg(["count","sum","mean"]).reset_index()
        out["by_date"] = [
            {"date": str(row["date"]), "n": int(row["count"]), "R_sum": float(row["sum"]), "E_R": float(row["mean"])}
            for _, row in daily.iterrows()
        ]

        tmp["dow"] = tmp["time"].dt.day_name().str[:3]  # Mon..Sun
        out["by_dow"] = _by_group(tmp, "dow")

        tmp["month"] = tmp["time"].dt.strftime("%Y-%m")
        out["by_month"] = _by_group(tmp, "month")
    else:
        out["by_hour"] = []
        out["by_date"] = []
        out["by_dow"] = []
        out["by_month"] = []

    # 교차표: 레짐 × 세션
    out["by_regime_session"] = _cross_group(df, "regime", "session")

    # MFE/MAE 요약(있을 때만)
    mfe = df["mfe_R"].dropna() if "mfe_R" in df else pd.Series(dtype=float)
    mae = df["mae_R"].dropna() if "mae_R" in df else pd.Series(dtype=float)
    out["mfe_mae"] = {
        "mfe_mean_R": float(mfe.mean()) if not mfe.empty else None,
        "mfe_p90_R": float(mfe.quantile(0.90)) if not mfe.empty else None,
        "mae_mean_R": float(mae.mean()) if not mae.empty else None,
        "mae_p90_R": float(mae.quantile(0.90)) if not mae.empty else None,
    }

    # 분포 분위수 (퀵 시각화용/튜닝 가이드)
    r = pd.to_numeric(df["R"], errors="coerce").fillna(0.0)
    out["quantiles"] = {
        "p01": float(r.quantile(0.01)), "p05": float(r.quantile(0.05)),
        "p25": float(r.quantile(0.25)), "p50": float(r.quantile(0.50)),
        "p75": float(r.quantile(0.75)), "p95": float(r.quantile(0.95)),
        "mean": float(r.mean()), "std": float(r.std(ddof=1) if len(r) >= 2 else 0.0),
    }

    # 에퀴티 테일(최근 50 이벤트) — 프론트에서 quick plot 용
    eq = _equity_curve(df)
    tail = eq.tail(50)
    out["equity_tail"] = [
        {"time": (t.isoformat() if pd.notna(t) else None),
         "cumR": float(c), "dd": float(d)} for t, c, d in zip(tail["time"], tail["cumR"], tail["drawdown"])
    ]

    return out
