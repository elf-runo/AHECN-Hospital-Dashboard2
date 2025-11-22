# components/utils.py
from datetime import datetime, timezone

def parse_dt(x):
    """Accept datetime or ISO string; return datetime (naive local tz)."""
    if isinstance(x, datetime):
        return x
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return datetime.now()
    return datetime.now()

def utcnow_iso():
    """UTC ISO timestamp for storage."""
    return datetime.now(timezone.utc).isoformat()
