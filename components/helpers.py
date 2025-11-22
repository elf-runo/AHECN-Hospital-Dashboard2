# components/helpers.py
"""
Clinical scoring helpers for AHECN triage engine.
Keep this module pure (no Streamlit UI) so it can be reused by EMT mobile,
batch scoring, and future APIs.
"""

from dataclasses import dataclass

def validate_vitals(hr, rr, sbp, temp, spo2):
    """Basic sanitization + defaults."""
    def _num(x, default):
        try:
            if x is None: 
                return default
            return float(x)
        except Exception:
            return default

    v = {
        "hr": _num(hr, 100),
        "rr": _num(rr, 20),
        "sbp": _num(sbp, 110),
        "temp": _num(temp, 37.0),
        "spo2": _num(spo2, 95),
    }
    return v


# -------------------------
# NEWS2 (Adult, Non-maternal)
# -------------------------
def calc_NEWS2(rr, spo2, sbp, hr, temp_c, avpu="A", on_oxygen=False, spo2_scale=1):
    """
    Returns: total_score, hits(list of tuples), review_flag(bool), urgent_flag(bool)
    This is a simplified but safe NEWS2 implementation for MVP demo.
    """

    hits = []
    total = 0

    # RR
    if rr <= 8: s = 3
    elif 9 <= rr <= 11: s = 1
    elif 12 <= rr <= 20: s = 0
    elif 21 <= rr <= 24: s = 2
    else: s = 3
    total += s; hits.append(("RR", rr, s))

    # SpO2 (Scale 1 / 2 simplified)
    if spo2_scale == 1:
        if spo2 <= 91: s = 3
        elif 92 <= spo2 <= 93: s = 2
        elif 94 <= spo2 <= 95: s = 1
        else: s = 0
    else:
        if spo2 <= 83: s = 3
        elif 84 <= spo2 <= 85: s = 2
        elif 86 <= spo2 <= 87: s = 1
        elif 88 <= spo2 <= 92: s = 0
        elif 93 <= spo2 <= 94: s = 1
        elif 95 <= spo2 <= 96: s = 2
        else: s = 3
    total += s; hits.append(("SpO2", spo2, s))

    # On oxygen adds 2
    if on_oxygen:
        total += 2
        hits.append(("O2", "Yes", 2))

    # SBP
    if sbp <= 90: s = 3
    elif 91 <= sbp <= 100: s = 2
    elif 101 <= sbp <= 110: s = 1
    elif 111 <= sbp <= 219: s = 0
    else: s = 3
    total += s; hits.append(("SBP", sbp, s))

    # HR
    if hr <= 40: s = 3
    elif 41 <= hr <= 50: s = 1
    elif 51 <= hr <= 90: s = 0
    elif 91 <= hr <= 110: s = 1
    elif 111 <= hr <= 130: s = 2
    else: s = 3
    total += s; hits.append(("HR", hr, s))

    # Temp
    if temp_c <= 35.0: s = 3
    elif 35.1 <= temp_c <= 36.0: s = 1
    elif 36.1 <= temp_c <= 38.0: s = 0
    elif 38.1 <= temp_c <= 39.0: s = 1
    else: s = 2
    total += s; hits.append(("Temp", temp_c, s))

    # AVPU
    avpu_map = {"A":0, "V":3, "P":3, "U":3}
    s = avpu_map.get(avpu, 0)
    total += s; hits.append(("AVPU", avpu, s))

    review_flag = total >= 5
    urgent_flag = total >= 7
    return total, hits, review_flag, urgent_flag


def safe_calc_NEWS2(rr, spo2, sbp, hr, temp_c, avpu="A", o2_device="Air", spo2_scale=1):
    """
    Wrapper used by triaging.py. Converts O2 device text to on_oxygen bool.
    """
    on_oxygen = (str(o2_device).lower() != "air")
    return calc_NEWS2(rr, spo2, sbp, hr, temp_c, avpu, on_oxygen, spo2_scale)


# -------------------------
# qSOFA (Sepsis comparator)
# -------------------------
def calc_qSOFA(rr, sbp, avpu="A"):
    hits = []
    score = 0
    if rr >= 22:
        score += 1; hits.append(("RR>=22", rr))
    if sbp <= 100:
        score += 1; hits.append(("SBP<=100", sbp))
    if avpu != "A":
        score += 1; hits.append(("Mental not alert", avpu))
    high = score >= 2
    return score, hits, high


# -------------------------
# MEOWS (Maternal)
# -------------------------
def calc_MEOWS(hr, rr, sbp, temp_c, spo2):
    red = []
    yellow = []

    if hr < 50 or hr > 120: red.append(("HR", hr))
    elif 100 <= hr <= 120: yellow.append(("HR", hr))

    if rr < 10 or rr > 30: red.append(("RR", rr))
    elif 21 <= rr <= 30: yellow.append(("RR", rr))

    if sbp < 90 or sbp > 160: red.append(("SBP", sbp))
    elif 90 <= sbp < 100 or 150 <= sbp <= 160: yellow.append(("SBP", sbp))

    if temp_c < 35 or temp_c >= 38: red.append(("Temp", temp_c))
    elif 37.5 <= temp_c < 38: yellow.append(("Temp", temp_c))

    if spo2 < 90: red.append(("SpO2", spo2))
    elif 90 <= spo2 < 95: yellow.append(("SpO2", spo2))

    return {"red": red, "yellow": yellow}


# -------------------------
# PEWS (Pediatrics)
# -------------------------
def calc_PEWS(age, rr, hr, behavior="Normal", spo2=98):
    """
    Very simplified PEWS for demo.
    Returns: score, meta, high(bool), watch(bool)
    """
    meta = {}
    score = 0

    # RR age bands (coarse)
    if age < 1:
        rr_high = rr > 50
    elif age < 5:
        rr_high = rr > 40
    else:
        rr_high = rr > 30
    if rr_high:
        score += 2; meta["RR"] = "high"

    # HR age bands (coarse)
    if age < 1:
        hr_high = hr > 180
    elif age < 5:
        hr_high = hr > 160
    else:
        hr_high = hr > 140
    if hr_high:
        score += 2; meta["HR"] = "high"

    # Behavior
    beh_map = {"Normal":0, "Irritable":1, "Lethargic":2}
    score += beh_map.get(behavior, 0)
    meta["Behavior"] = behavior

    # SpO2
    if spo2 < 92:
        score += 2; meta["SpO2"] = "low"
    elif spo2 < 95:
        score += 1; meta["SpO2"] = "borderline"

    high = score >= 6
    watch = score >= 4 and not high
    return score, meta, high, watch
