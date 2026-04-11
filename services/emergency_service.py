"""Emergency gesture checks, cooldown, and persistence."""
from __future__ import annotations

import time
from typing import Dict, Optional, Set

from utils.app_logger import get_logger
from utils.db import get_db

_log = get_logger("signai.emergency")

_DEFAULT_EMERGENCY: Set[str] = {"HELP", "SOS", "DANGER"}


def load_emergency_gesture_names() -> Set[str]:
    names = set(_DEFAULT_EMERGENCY)
    try:
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(
                "SELECT gesture_name FROM gestures WHERE is_emergency = 1"
            )
            for row in cur.fetchall():
                if row.get("gesture_name"):
                    names.add(str(row["gesture_name"]).upper().strip())
        except Exception:
            cur.execute("SELECT gesture_name FROM gestures")
            for row in cur.fetchall():
                g = str(row["gesture_name"]).upper().strip()
                if g in _DEFAULT_EMERGENCY:
                    names.add(g)
        cur.close()
        conn.close()
    except Exception as e:
        _log.warning("emergency gesture load: %s", e)
    return names


def is_emergency_gesture(gesture: str, cache: Optional[Set[str]] = None) -> bool:
    g = (gesture or "").upper().strip()
    pool = cache if cache is not None else load_emergency_gesture_names()
    return g in pool


def cooldown_active(last_ts: Optional[float], cooldown_sec: float) -> bool:
    if last_ts is None or last_ts <= 0:
        return False
    return (time.time() - last_ts) < cooldown_sec


def log_emergency_event(
    user_id: Optional[int],
    gesture_name: str,
    confidence: float,
    extra: Optional[str] = None,
) -> None:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO emergency_logs (user_id, gesture_name, confidence_score, detail, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            """,
            (user_id, gesture_name[:80], float(confidence), (extra or "")[:500]),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        _log.error("emergency log failed: %s", e)
