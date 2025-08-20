# src/btcusdt_algo/core/session.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

"""
세션 유틸:
- which(dt, settings=None) -> "ASIA" | "EU" | "US" | <custom name>
- score_mult(settings, dt) -> float (세션별 스코어 배율)
- is_blocked(settings, dt) -> bool (세션/요일 차단)

설정 호환(둘 다 지원):
1) 평면형
session:
  score_mult: { ASIA: 1.0, EU: 1.0, US: 1.0 }
  block: ["EU"]

2) 중첩형
session:
  asia:   { score_mult: 1.0 }
  europe: { score_mult: 1.0 }
  us:     { score_mult: 1.0 }
  block: []

추가 지원:
- 타임존 지정 (기본: UTC)
  session:
    timezone: "Asia/Seoul"

- 커스텀 세션 창(분 단위 지원; 시간 정수/실수도 허용):
  session:
    windows:
      - { name: "ASIA", start: 0,   end: 8 }           # 시간(시간 정수/실수)
      - { name: "EU",   start: 8,   end: 13 }
      - { name: "US",   start: 13,  end: 21 }
      - { name: "ASIA", start: 21,  end: 24 }
    # 또는 분 단위:
      - { name: "US", start_minute: 780, end_minute: 1260 }  # 13:00~21:00

- 요일 차단:
  session:
    block_days: ["SAT", "SUN"]  # 대문자/소문자 무관, 앞 3글자만 봐도 됨
"""

# 표준 라이브러리 zoneinfo (Py3.9+) 사용. 없으면 UTC로 폴백.
try:
    from zoneinfo import ZoneInfo  # type: ignore
    _HAS_ZONEINFO = True
except Exception:
    ZoneInfo = None  # type: ignore
    _HAS_ZONEINFO = False

# 기본 세션 윈도우(분 단위, UTC 기준)
# 00:00~08:00 ASIA / 08:00~13:00 EU / 13:00~21:00 US / 21:00~24:00 ASIA
DEFAULT_WINDOWS_MIN: List[Tuple[str, int, int]] = [
    ("ASIA", 0,   480),
    ("EU",   480, 780),
    ("US",   780, 1260),
    ("ASIA", 1260, 1440),
]

_DOW3 = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

def _norm_name(s: str) -> str:
    return str(s or "").strip().upper()

def _to_minutes(val: Optional[float | int]) -> Optional[int]:
    """
    시간(시) → 분 변환. 실수면 8.5 → 510분.
    None/유효하지 않으면 None.
    """
    if val is None:
        return None
    try:
        f = float(val)
        if f < 0:
            return None
        # 시간(소수) → 분
        mins = int(round(f * 60))
        return mins
    except Exception:
        return None

def _pick_tz(settings: Dict[str, Any] | None):
    tzname = ((settings or {}).get("session", {}) or {}).get("timezone", "UTC")
    if not isinstance(tzname, str):
        tzname = "UTC"
    if _HAS_ZONEINFO:
        try:
            return ZoneInfo(tzname)  # type: ignore
        except Exception:
            return timezone.utc
    return timezone.utc

def _to_local(dt: datetime, settings: Dict[str, Any] | None) -> datetime:
    """
    설정된 세션 타임존으로 변환. naive면 UTC로 간주 후 변환.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    tz = _pick_tz(settings)
    return dt.astimezone(tz)

def _load_windows(settings: Dict[str, Any] | None) -> List[Tuple[str, int, int]]:
    """
    설정에서 세션 창을 읽어 분 단위 튜플(name, start_min, end_min) 리스트로 반환.
    - start/end (시간, 실수 허용) 또는 start_minute/end_minute (분) 지원.
    - 0 <= minute < 1440, start != end.
    - 자정 랩어라운드(start > end) 허용.
    - 유효하지 않으면 DEFAULT 사용.
    """
    sess_cfg = (settings or {}).get("session", {}) or {}
    wins = sess_cfg.get("windows")
    out: List[Tuple[str, int, int]] = []

    if isinstance(wins, list) and wins:
        for w in wins:
            try:
                name = _norm_name(w.get("name"))
                # 분 우선, 없으면 시간 사용
                sm = w.get("start_minute")
                em = w.get("end_minute")
                if sm is not None or em is not None:
                    smin = int(sm)
                    emin = int(em)
                else:
                    smin = _to_minutes(w.get("start"))
                    emin = _to_minutes(w.get("end"))
                    if smin is None or emin is None:
                        continue

                if 0 <= smin < 1440 and 0 < emin <= 1440 and smin != emin and name:
                    out.append((name, smin, emin))
            except Exception:
                continue

    return out or DEFAULT_WINDOWS_MIN

def which(dt: datetime, settings: Dict[str, Any] | None = None) -> str:
    """
    현재 시각이 어느 세션인지 반환.
    - 세션 창은 설정된 타임존 기준으로 평가.
    - 구간은 [start, end) 반개구간. 자정 랩어라운드(start>end) 처리.
    - 매칭되는 첫 번째 창을 반환 (리스트 순서대로).
    """
    dt_loc = _to_local(dt, settings)
    minute_of_day = dt_loc.hour * 60 + dt_loc.minute

    for name, smin, emin in _load_windows(settings):
        if smin < emin:
            if smin <= minute_of_day < emin:
                return name
        else:
            # 자정 랩어라운드
            if (minute_of_day >= smin) or (minute_of_day < emin):
                return name
    # 기본값
    return "ASIA"

def score_mult(settings: Dict[str, Any], dt: datetime) -> float:
    """
    세션별 스코어 배율을 반환. 두 가지 스키마 다 지원.
    1) session.score_mult = { "ASIA":1.0, "EU":1.0, "US":1.0 }
    2) session.asia.score_mult = 1.0 등 중첩형
    대소문자 무시.
    """
    sess = which(dt, settings)
    sess_cfg = (settings or {}).get("session", {}) or {}

    # (a) 평면형
    flat = sess_cfg.get("score_mult", {})
    if isinstance(flat, dict) and flat:
        for k, v in flat.items():
            if _norm_name(k) == _norm_name(sess):
                try:
                    return float(v)
                except Exception:
                    break

    # (b) 중첩형
    for k, v in sess_cfg.items():
        if _norm_name(k) == _norm_name(sess) and isinstance(v, dict):
            if "score_mult" in v:
                try:
                    return float(v.get("score_mult", 1.0))
                except Exception:
                    pass

    return 1.0

def _day3(dt: datetime, settings: Dict[str, Any] | None = None) -> str:
    """로컬 타임존 기준 요일 3글자 코드 반환 (MON..SUN)."""
    dloc = _to_local(dt, settings)
    # Monday=0 ... Sunday=6
    idx = dloc.weekday()
    return _DOW3[idx]

def is_blocked(settings: Dict[str, Any], dt: datetime) -> bool:
    """
    차단 여부:
    - session.block 에 현재 세션명이 있으면 True.
    - session.block_days 에 현재 요일(MON..SUN, 대/소문자 무관, 앞3글자) 이 있으면 True.
    """
    sess = which(dt, settings)
    sess_cfg = (settings or {}).get("session", {}) or {}

    # 세션 차단
    blocked = set(sess_cfg.get("block", []) or [])
    if any(_norm_name(x) == _norm_name(sess) for x in blocked):
        return True

    # 요일 차단
    bdays = sess_cfg.get("block_days", []) or []
    if isinstance(bdays, list) and bdays:
        d3 = _norm_name(_day3(dt, settings))[:3]
        for x in bdays:
            if _norm_name(str(x))[:3] == d3:
                return True

    return False
