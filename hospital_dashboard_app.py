# hospital_dashboard_app.py
"""
AHECN Hospital Command Center — Streamlit MVP
--------------------------------------------
Self-contained Streamlit app implementing:
- Premium command-center UI
- Synthetic inter-hospital referral data with active inbound/outbound transfers
- Deterministic score-based triage:
    * MEOWS (primary) for all maternal cases
    * NEWS2 + qSOFA for adults
    * PEWS placeholder for paediatrics (<16)
- MEOWS / NEWS2 / qSOFA breakdown UI
- Optional ML model comparison (safe-guarded)
- New Outbound Referral flow with real-time triage preview,
  multi-reason referral inputs, protocol intervention preview,
  and smart facility matching + scoring

Notes:
- If deployment uses Python 3.13+, avoid pinning old pandas/sklearn in requirements.txt.
- my_model.pkl (optional) should output triage class labels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random
import math
import os
import csv
from pathlib import Path

# joblib is optional – used only if available
try:
    import joblib
except ImportError:
    joblib = None


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AHECN Hospital Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)


# =========================
# PREMIUM CSS
# =========================
st.markdown("""
<style>
:root {
    --primary: #1a237e;
    --secondary: #283593;
    --accent: #536dfe;
    --success: #00c853;
    --warning: #ffab00;
    --danger: #ff1744;
    --dark: #0d1b2a;
    --light: #f8f9fa;
    --gradient: linear-gradient(135deg, #1a237e 0%, #283593 100%);
}

.main-header {
    background: var(--gradient);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.premium-card {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid #e0e0e0;
}

.case-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 5px solid var(--accent);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    cursor: pointer;
    transition: all 0.3s ease;
}
.case-card:hover { transform: translateX(5px); box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
.case-card.critical { border-left-color: var(--danger); background: #fff5f5; }
.case-card.urgent { border-left-color: var(--warning); background: #fffbf0; }
.case-card.stable { border-left-color: var(--success); background: #f8fff8; }

.metric-highlight {
    background: var(--gradient);
    color: white;
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.timeline-event {
    border-left: 3px solid var(--accent);
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    background: #f8f9ff;
}

.intervention-badge {
    display: inline-block;
    background: #e3f2fd;
    color: #1976d2;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin: 0.2rem;
    border: 1px solid #bbdefb;
}
</style>
""", unsafe_allow_html=True)


# =========================
# CONSTANTS & CATALOGS
# =========================
DASHBOARD_HOSPITAL = "Tertiary Central Hospital"
DASHBOARD_LAT = 25.578
DASHBOARD_LON = 91.893

# Seed ICDs per category + auto-generated demo ICDs for pitch density
BASE_ICD_BY_TYPE = {
    "Maternal": [
        ("O72.0","Third-stage haemorrhage"),
        ("O72.1","Immediate postpartum haemorrhage"),
        ("O14.1","Severe pre-eclampsia"),
        ("O15.0","Eclampsia in pregnancy"),
        ("O34.2","Previous cesarean scar complication"),
    ],
    "Trauma": [
        ("S06.0","Concussion"),
        ("S06.5","Traumatic subdural haemorrhage"),
        ("S72.3","Fracture of femur"),
        ("S27.0","Traumatic pneumothorax"),
        ("S36.1","Laceration of liver"),
    ],
    "Stroke": [
        ("I63.9","Cerebral infarction unspecified"),
        ("I61.9","Intracerebral haemorrhage unspecified"),
        ("G45.9","Transient ischemic attack"),
    ],
    "Cardiac": [
        ("I21.9","Acute myocardial infarction unspecified"),
        ("I50.9","Heart failure unspecified"),
        ("I20.0","Unstable angina"),
    ],
    "Sepsis": [
        ("A41.9","Sepsis unspecified organism"),
        ("R65.2","Severe sepsis"),
        ("J18.9","Pneumonia unspecified organism"),
    ],
    "Other": [
        ("J96.0","Acute respiratory failure"),
        ("K92.2","GI bleed unspecified"),
        ("E11.6","Diabetes with other complications"),
    ],
}

def build_demo_icd_catalog(per_type=50):
    catalog = []
    for ctype, seeds in BASE_ICD_BY_TYPE.items():
        for code, label in seeds:
            catalog.append({"icd_code": code, "label": label, "case_type": ctype, "age_min": 0, "age_max": 120})
        for i in range(per_type - len(seeds)):
            code = f"{ctype[:1]}{100+i}"
            label = f"{ctype} provisional diagnosis {i+1}"
            catalog.append({"icd_code": code, "label": label, "case_type": ctype, "age_min": 0, "age_max": 120})
    return catalog

ICD_CATALOG = build_demo_icd_catalog(per_type=50)

INTERVENTION_PROTOCOLS = {
    "Maternal": [
        "IV fluids", "Uterotonics", "Tranexamic acid (TXA)",
        "Oxygen", "BP monitoring", "Fundal massage",
        "Prepare OT/LSCS", "Crossmatch blood", "Magnesium sulphate if indicated"
    ],
    "Trauma": [
        "Airway management", "2 large-bore IV access", "Bleeding control",
        "C-spine immobilization", "Pain management", "FAST/CT prep",
        "Chest tube if indicated", "Emergency surgery prep"
    ],
    "Stroke": [
        "BP control", "Glucose check", "Neurological assessment",
        "CT brain", "Thrombolysis prep", "Neuro consult", "Frequent monitoring"
    ],
    "Cardiac": [
        "Aspirin", "Oxygen", "IV access",
        "ECG monitoring", "Nitroglycerin if indicated",
        "Cath lab activation", "Heparin per protocol"
    ],
    "Sepsis": [
        "Broad-spectrum antibiotics", "IV fluids bolus",
        "Blood cultures", "Lactate monitoring",
        "Vasopressors if shock", "Oxygen", "Urine output monitoring"
    ],
    "Other": [
        "Oxygen", "IV access", "Symptom control",
        "Monitoring", "Specialist consult", "Imaging/labs as indicated"
    ]
}

# Map key ICDs to tighter intervention sets
ICD_INTERVENTION_MAP = {
    "O72.0": ["IV fluids", "Uterotonics", "TXA", "Crossmatch blood", "Prepare OT", "Oxygen"],
    "O72.1": ["IV fluids", "Uterotonics", "TXA", "2 large-bore IV lines", "Monitor uterine tone", "Oxygen"],
    "O14.1": ["Magnesium sulphate", "IV access", "BP monitoring", "Labs (LFT/RFT/platelets)", "Prepare delivery", "Oxygen"],
    "S06.0": ["C-spine immobilization", "Airway management", "Neuro checks", "IV access", "CT head prep"],
    "S06.5": ["Airway management", "IV access", "Hyperosmolar therapy prep", "CT head", "Neurosurgery alert"],
    "I63.9": ["BP control", "Glucose check", "CT brain", "Thrombolysis prep", "Neuro consult"],
    "I21.9": ["Aspirin", "ECG monitoring", "Nitroglycerin", "Oxygen", "Cath lab activation"],
    "A41.9": ["Broad-spectrum antibiotics", "IV fluids", "Blood cultures", "Lactate monitoring", "Oxygen"],
    "J96.0": ["Oxygen", "Nebulization", "IV access", "Ventilation prep", "ABG sample"]
}

REFERRAL_REASONS = {
    "ICU_BED_UNAVAILABLE": [
        "No ICU beds free at referring facility",
        "Ventilator bed unavailable",
        "High-dependency unit full"
    ],
    "SPECIALTY_REQUIRED": {
        "Maternal": ["ObGyn emergency surgery / LSCS", "Maternal ICU monitoring"],
        "Trauma": ["Neurosurgery / trauma surgeon", "Ortho trauma fixation"],
        "Stroke": ["Stroke unit / thrombolysis capability"],
        "Cardiac": ["Cath lab / cardiologist", "Cardiac ICU"],
        "Sepsis": ["Critical care intensivist", "Organ support"],
        "Other": ["Specialist consult needed"]
    },
    "EQUIPMENT_REQUIRED": {
        "Maternal": ["Blood bank / massive transfusion", "USG/CT obstetric imaging"],
        "Trauma": ["CT/MRI trauma imaging", "Operating theatre"],
        "Stroke": ["CT/MRI + thrombolysis setting"],
        "Cardiac": ["Cath lab / PCI suite"],
        "Sepsis": ["Ventilator / dialysis support"],
        "Other": ["Advanced respiratory or imaging support"]
    }
}

EMT_CREW = [
    {"id": "EMT_001", "name": "John Carter", "level": "ALS", "vehicle": "Ambulance 1", "status": "active"},
    {"id": "EMT_002", "name": "Sarah Connor", "level": "BLS", "vehicle": "Ambulance 2", "status": "active"},
    {"id": "EMT_003", "name": "Mike Rodriguez", "level": "ALS", "vehicle": "Ambulance 3", "status": "available"},
    {"id": "EMT_004", "name": "Lisa Park", "level": "Critical Care", "vehicle": "Mobile ICU", "status": "available"},
]


# =========================
# HELPERS
# =========================
def get_unique_key(component_type, case_type, case_data, extra_suffix=""):
    case_id = case_data.get("case_id", "unknown")
    ts = case_data.get("timestamp")
    if hasattr(ts, "strftime"):
        timestamp = ts.strftime("%Y%m%d%H%M%S")
    else:
        timestamp = "notime"
    return f"{component_type}_{case_type}_{case_id}_{timestamp}_{extra_suffix}"


def generate_facility_capabilities(facilities):
    caps = {}
    for f in facilities:
        name = f["name"]
        ftype = f["type"]
        icu_beds_total = int(f["beds"] * 0.12)
        icu_beds_available = random.randint(0, max(1, icu_beds_total))
        bed_occupancy_pct = random.randint(60, 98)

        if ftype == "Tertiary":
            specs = {"ObGyn", "Neuro", "Cardiology", "Trauma", "ICU", "Neonatal"}
            eqp = {"CT", "MRI", "Ventilator", "CathLab", "BloodBank", "USG"}
        elif ftype == "Trauma":
            specs = {"Trauma", "Ortho", "Neuro", "ICU"}
            eqp = {"CT", "Ventilator", "OR", "BloodBank"}
        elif ftype == "Specialty":
            specs = {"Cardiology", "Neuro", "ICU"}
            eqp = {"CathLab", "CT", "Ventilator"}
        else:
            specs = {"GeneralMed", "ObGyn", "ICU"}
            eqp = {"USG", "Ventilator", "BloodBank"}

        caps[name] = {
            "icu_beds_total": icu_beds_total,
            "icu_beds_available": icu_beds_available,
            "bed_occupancy_pct": bed_occupancy_pct,
            "specialties": specs,
            "equipment": eqp,
            "type": ftype,
            "lat": f["lat"],
            "lon": f["lon"],
            "beds": f["beds"],
        }
    return caps


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2*r*math.asin(math.sqrt(a))


def score_facility_for_case(case_type, triage_color, reason_category, facility_cap):
    dist_km = haversine_km(DASHBOARD_LAT, DASHBOARD_LON, facility_cap["lat"], facility_cap["lon"])
    distance_score = max(0, 30 * (1 - min(dist_km, 100) / 100))

    specs = facility_cap["specialties"]
    eqp = facility_cap["equipment"]
    capability_score = 0

    if reason_category == "ICU_BED_UNAVAILABLE":
        capability_score += 30 if facility_cap["icu_beds_available"] > 0 else 5
    elif reason_category == "SPECIALTY_REQUIRED":
        spec_map = {
            "ObGyn": {"Maternal"},
            "Trauma": {"Trauma"},
            "Neuro": {"Stroke", "Trauma"},
            "Cardiology": {"Cardiac"},
            "ICU": {"Sepsis", "Other", "Trauma", "Cardiac", "Stroke", "Maternal"},
        }
        if any(case_type in v for k, v in spec_map.items() if k in specs):
            capability_score += 35
        else:
            capability_score += 10
    elif reason_category == "EQUIPMENT_REQUIRED":
        eq_map = {
            "CT": {"Trauma", "Stroke"},
            "MRI": {"Stroke"},
            "CathLab": {"Cardiac"},
            "Ventilator": {"Sepsis", "Other", "Trauma"},
            "USG": {"Maternal"},
            "BloodBank": {"Maternal", "Trauma"},
            "OR": {"Trauma", "Maternal"},
        }
        if any(case_type in v for k, v in eq_map.items() if k in eqp):
            capability_score += 35
        else:
            capability_score += 10

    if "ICU" in specs:
        capability_score += 5
    capability_score = min(capability_score, 40)

    ftype = facility_cap["type"]
    if triage_color == "RED":
        triage_fit = 20 if ftype in ["Tertiary", "Trauma", "Specialty"] else 8
    elif triage_color == "YELLOW":
        triage_fit = 16 if ftype in ["Tertiary", "District", "Specialty"] else 10
    else:
        triage_fit = 12 if ftype in ["District", "Tertiary"] else 8

    occ = facility_cap["bed_occupancy_pct"]
    load_score = max(0, 10 * (1 - (occ - 60) / 40))
    load_score = min(load_score, 10)

    total = distance_score + capability_score + triage_fit + load_score

    return round(total, 1), {
        "distance_km": round(dist_km, 1),
        "distance_score": round(distance_score, 1),
        "capability_score": round(capability_score, 1),
        "triage_fit": triage_fit,
        "load_score": round(load_score, 1),
        "icu_available": facility_cap["icu_beds_available"],
        "occupancy_pct": occ,
        "facility_type": ftype
    }


def _make_transit_updates(case_time, vitals, transport_time):
    """Generate EMT vitals + note updates during transit."""
    vitals = vitals or {}
    updates = []
    n_updates = random.randint(2, 4)

    base_hr = vitals.get("hr", 90)
    base_sbp = vitals.get("sbp", 110)
    base_rr = vitals.get("rr", 20)
    base_spo2 = vitals.get("spo2", 95)
    base_temp = vitals.get("temp", 37.0)
    base_avpu = vitals.get("avpu", "A")

    for u in range(n_updates):
        upd_time = case_time + timedelta(minutes=(u+1) * transport_time/(n_updates+1))
        upd_vitals = {
            "hr": max(40, base_hr + random.randint(-8, 8)),
            "sbp": max(60, base_sbp + random.randint(-10, 10)),
            "rr": max(8, base_rr + random.randint(-3, 3)),
            "spo2": min(100, max(80, base_spo2 + random.randint(-2, 2))),
            "temp": round(base_temp + random.uniform(-0.2, 0.2), 1),
            "avpu": base_avpu
        }
        updates.append({
            "timestamp": upd_time,
            "vitals": upd_vitals,
            "emt_note": random.choice([
                "Airway patent, oxygen continued",
                "IV line secured, monitoring ongoing",
                "BP stabilized after fluids",
                "Pain controlled, immobilization maintained",
                "GCS stable; neuro checks repeated",
                "Nebulization given; SpO₂ improved"
            ])
        })
    return updates


# =========================
# MODEL LOADING (optional)
# =========================
@st.cache_resource
def load_triage_model():
    if joblib is None:
        return None
    model_path = Path(__file__).resolve().parent / "my_model.pkl"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def get_triage_model():
    return load_triage_model()


# =========================
# FEEDBACK LOGGING
# =========================
FEEDBACK_LOG_PATH = "ai_feedback_log.csv"
FEEDBACK_FIELDS = ["timestamp_utc","case_id","patient_age","sbp","spo2","hr","ai_suggestion","feedback"]

def log_ai_feedback(case, features, ai_suggestion, feedback):
    if not isinstance(features, (list, tuple)) or len(features) < 4:
        return
    file_exists = os.path.exists(FEEDBACK_LOG_PATH)
    with open(FEEDBACK_LOG_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.utcnow().isoformat(),
            "case_id": case.get("case_id","-"),
            "patient_age": features[0],
            "sbp": features[1],
            "spo2": features[2],
            "hr": features[3],
            "ai_suggestion": ai_suggestion,
            "feedback": feedback
        })


# =========================
# TRIAGE SCORING
# =========================
def calculate_qsofa(rr, sbp, avpu):
    score = 0
    if rr >= 22: score += 1
    if sbp <= 100: score += 1
    if (avpu or "A").upper() != "A": score += 1
    return score


def calculate_news2(rr, spo2, sbp, hr, temp_c, avpu, on_oxygen=False, spo2_scale2=False):
    parts = {}

    # RR
    if rr <= 8: parts["rr"] = 3
    elif rr <= 11: parts["rr"] = 1
    elif rr <= 20: parts["rr"] = 0
    elif rr <= 24: parts["rr"] = 2
    else: parts["rr"] = 3

    # SpO2
    if spo2_scale2:
        if spo2 >= 97: parts["spo2"]=3
        elif spo2 >= 95: parts["spo2"]=2
        elif spo2 >= 93: parts["spo2"]=1
        elif spo2 >= 88: parts["spo2"]=0
        elif spo2 >= 86: parts["spo2"]=1
        elif spo2 >= 84: parts["spo2"]=2
        else: parts["spo2"]=3
    else:
        if spo2 >= 96: parts["spo2"]=0
        elif spo2 >= 94: parts["spo2"]=1
        elif spo2 >= 92: parts["spo2"]=2
        else: parts["spo2"]=3

    # Oxygen
    parts["oxygen"] = 2 if on_oxygen else 0

    # SBP
    if sbp <= 90: parts["sbp"]=3
    elif sbp <= 100: parts["sbp"]=2
    elif sbp <= 110: parts["sbp"]=1
    elif sbp <= 219: parts["sbp"]=0
    else: parts["sbp"]=3

    # HR
    if hr <= 40: parts["hr"]=3
    elif hr <= 50: parts["hr"]=1
    elif hr <= 90: parts["hr"]=0
    elif hr <= 110: parts["hr"]=1
    elif hr <= 130: parts["hr"]=2
    else: parts["hr"]=3

    # Temp
    if temp_c <= 35.0: parts["temp"]=3
    elif temp_c <= 36.0: parts["temp"]=1
    elif temp_c <= 38.0: parts["temp"]=0
    elif temp_c <= 39.0: parts["temp"]=1
    else: parts["temp"]=2

    # Consciousness
    parts["conc"] = 3 if (avpu or "A").upper() != "A" else 0

    total = sum(parts.values())
    return total, parts


def calculate_pews_placeholder(age, rr, spo2, sbp, hr, avpu):
    score = 0
    if rr > 30: score += 1
    if spo2 < 92: score += 1
    if hr > 140 or hr < 60: score += 1
    if sbp < 90: score += 1
    if (avpu or "A").upper() != "A": score += 1
    return score


def calculate_meows(rr, spo2, sbp, hr, temp_c, avpu):
    """
    Simplified MEOWS trigger chart logic for maternal cases.
    Returns:
      triage_color, triggers_dict, amber_count
    """
    avpu = (avpu or "A").upper()
    triggers = {}

    # HR
    if hr < 40 or hr > 120:
        triggers["HR"] = "RED"
    elif 40 <= hr < 50 or 100 <= hr <= 120:
        triggers["HR"] = "AMBER"
    else:
        triggers["HR"] = "GREEN"

    # SBP
    if sbp < 90 or sbp > 160:
        triggers["SBP"] = "RED"
    elif 90 <= sbp < 100 or 150 <= sbp <= 160:
        triggers["SBP"] = "AMBER"
    else:
        triggers["SBP"] = "GREEN"

    # RR
    if rr < 10 or rr > 30:
        triggers["RR"] = "RED"
    elif 10 <= rr < 12 or 21 <= rr <= 30:
        triggers["RR"] = "AMBER"
    else:
        triggers["RR"] = "GREEN"

    # SpO2
    if spo2 < 92:
        triggers["SpO2"] = "RED"
    elif 92 <= spo2 <= 94:
        triggers["SpO2"] = "AMBER"
    else:
        triggers["SpO2"] = "GREEN"

    # Temp
    if temp_c < 35.0 or temp_c >= 38.5:
        triggers["Temp"] = "RED"
    elif 35.0 <= temp_c < 36.0 or 38.0 <= temp_c < 38.5:
        triggers["Temp"] = "AMBER"
    else:
        triggers["Temp"] = "GREEN"

    # AVPU
    if avpu in ["P", "U"]:
        triggers["AVPU"] = "RED"
    elif avpu == "V":
        triggers["AVPU"] = "AMBER"
    else:
        triggers["AVPU"] = "GREEN"

    red_count = sum(1 for v in triggers.values() if v == "RED")
    amber_count = sum(1 for v in triggers.values() if v == "AMBER")

    if red_count >= 1:
        triage = "RED"
    elif amber_count >= 2:
        triage = "YELLOW"
    else:
        triage = "GREEN"

    return triage, triggers, amber_count


def score_based_triage(case, on_oxygen=False, spo2_scale2=False):
    """
    Deterministic triage using:
      - MEOWS for all Maternal cases (primary)
      - PEWS placeholder for pediatric (<16)
      - NEWS2 + qSOFA for adults (>=16)

    Returns:
      triage_color (RED/YELLOW/GREEN),
      explanation_dict
    """
    vitals = case.get("vitals", {}) or {}

    age = float(case.get("patient_age", 0) or 0)
    rr = float(vitals.get("rr", 0) or 0)
    spo2 = float(vitals.get("spo2", 0) or 0)
    sbp = float(vitals.get("sbp", 0) or 0)
    hr = float(vitals.get("hr", 0) or 0)
    temp_c = float(vitals.get("temp", 36.5) or 36.5)
    avpu = (vitals.get("avpu", "A") or "A").upper()

    # -------------------------
    # Maternal pathway
    # -------------------------
    if case.get("case_type") == "Maternal":
        meows_triage, meows_triggers, meows_total = calculate_meows(
            rr=rr, spo2=spo2, sbp=sbp, hr=hr, temp_c=temp_c, avpu=avpu
        )

        # Secondary comparator only (not used to drive maternal triage)
        news2_total, news2_parts = calculate_news2(
            rr=rr, spo2=spo2, sbp=sbp, hr=hr, temp_c=temp_c, avpu=avpu,
            on_oxygen=on_oxygen, spo2_scale2=spo2_scale2
        )
        qsofa = calculate_qsofa(rr=rr, sbp=sbp, avpu=avpu)

        return meows_triage, {
            "system": "MEOWS",
            "secondary_system": "NEWS2/qSOFA (comparison only)",
            "meows_total": meows_total,
            "meows_triggers": meows_triggers,
            "secondary_news2_total": news2_total,
            "secondary_news2_parts": news2_parts,
            "secondary_qsofa": qsofa,
            "notes": "Maternal triage uses MEOWS as primary. NEWS2/qSOFA shown only for comparison."
        }

    # -------------------------
    # Pediatric pathway
    # -------------------------
    if age < 16:
        pews = calculate_pews_placeholder(age, rr, spo2, sbp, hr, avpu)
        if pews >= 4:
            triage = "RED"
        elif pews >= 2:
            triage = "YELLOW"
        else:
            triage = "GREEN"
        return triage, {
            "system": "PEWS (placeholder)",
            "score": pews,
            "components": {},
            "notes": "Replace placeholder with hospital PEWS chart later."
        }

    # -------------------------
    # Adult pathway (NEWS2 + qSOFA)
    # -------------------------
    news2_total, news2_parts = calculate_news2(
        rr=rr, spo2=spo2, sbp=sbp, hr=hr, temp_c=temp_c, avpu=avpu,
        on_oxygen=on_oxygen, spo2_scale2=spo2_scale2
    )
    qsofa = calculate_qsofa(rr=rr, sbp=sbp, avpu=avpu)

    if news2_total >= 7:
        triage = "RED"
    elif news2_total >= 5:
        triage = "YELLOW"
    else:
        triage = "GREEN"

    if qsofa >= 2:
        if triage == "GREEN":
            triage = "YELLOW"
        elif triage == "YELLOW":
            triage = "RED"

    return triage, {
        "system": "NEWS2 + qSOFA",
        "news2_total": news2_total,
        "news2_parts": news2_parts,
        "qsofa": qsofa
    }


# =========================
# SYNTHETIC DATA GENERATION
# =========================
def generate_premium_synthetic_data(days_back=30):
    facilities = [
        {"name": "Tertiary Central Hospital", "type": "Tertiary", "lat": 25.578, "lon": 91.893, "beds": 500},
        {"name": "District North General", "type": "District", "lat": 25.591, "lon": 91.878, "beds": 200},
        {"name": "Specialty South Medical", "type": "Specialty", "lat": 25.565, "lon": 91.901, "beds": 150},
        {"name": "Trauma East Center", "type": "Trauma", "lat": 25.572, "lon": 91.885, "beds": 300},
    ]
    facility_caps = generate_facility_capabilities(facilities)

    case_types = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]
    referred_cases, received_cases = [], []
    base_time = datetime.now() - timedelta(days=days_back)

    for day in range(days_back):
        daily_cases = random.randint(3, 8)
        for case_num in range(daily_cases):
            case_time = base_time + timedelta(
                days=day,
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            case_type = random.choice(case_types)
            age = random.randint(18, 80) if case_type != "Maternal" else random.randint(18, 40)

            # Vitals by type
            if case_type == "Maternal":
                vitals = {
                    "hr": random.randint(90, 140),
                    "sbp": random.randint(80, 160),
                    "rr": random.randint(18, 30),
                    "temp": round(random.uniform(36.5, 38.5), 1),
                    "spo2": random.randint(92, 99),
                    "avpu": "A"
                }
            elif case_type == "Trauma":
                vitals = {
                    "hr": random.randint(70, 150),
                    "sbp": random.randint(70, 180),
                    "rr": random.randint(16, 35),
                    "temp": round(random.uniform(36.0, 38.0), 1),
                    "spo2": random.randint(88, 98),
                    "avpu": random.choices(["A", "V", "P"], weights=[0.7, 0.2, 0.1])[0]
                }
            else:
                vitals = {
                    "hr": random.randint(80, 140),
                    "sbp": random.randint(90, 150),
                    "rr": random.randint(18, 32),
                    "temp": round(random.uniform(36.5, 39.0), 1),
                    "spo2": random.randint(86, 95),
                    "avpu": random.choices(["A", "V"], weights=[0.8, 0.2])[0]
                }

            # ICD pick
            matching_icd = [
                icd for icd in ICD_CATALOG
                if icd["case_type"] == case_type and icd["age_min"] <= age <= icd["age_max"]
            ]
            icd = random.choice(matching_icd) if matching_icd else random.choice(
                [i for i in ICD_CATALOG if i["case_type"] == case_type]
            )

            # Referral reason
            reason_category = random.choices(
                ["ICU_BED_UNAVAILABLE","SPECIALTY_REQUIRED","EQUIPMENT_REQUIRED"],
                weights=[0.35,0.45,0.20]
            )[0]
            if reason_category == "ICU_BED_UNAVAILABLE":
                reason_detail = random.choice(REFERRAL_REASONS["ICU_BED_UNAVAILABLE"])
            elif reason_category == "SPECIALTY_REQUIRED":
                reason_detail = random.choice(REFERRAL_REASONS["SPECIALTY_REQUIRED"][case_type])
            else:
                reason_detail = random.choice(REFERRAL_REASONS["EQUIPMENT_REQUIRED"][case_type])

            # Interventions
            base_intv = ICD_INTERVENTION_MAP.get(icd["icd_code"], INTERVENTION_PROTOCOLS[case_type])
            interventions = random.sample(base_intv, min(len(base_intv), random.randint(2, 4)))

            emt_crew = random.choice(EMT_CREW)
            transport_time = random.randint(20, 90)

            # Generate EMT monitoring history for all cases, then optionally drop for some completed
            transit_updates = _make_transit_updates(case_time, vitals, transport_time)

            # Active vs completed transfer
            is_active_transfer = random.random() < 0.18  # slightly higher for demo
            eta_minutes = transport_time if is_active_transfer else 0
            if not is_active_transfer:
                # keep logs for most completed cases to enrich demo depth
                if random.random() > 0.7:
                    transit_updates = []

            # OUTBOUND (from dashboard hospital to others)
            referring_facility_out = DASHBOARD_HOSPITAL
            receiving_facility_out = random.choice(
                [f["name"] for f in facilities if f["name"] != DASHBOARD_HOSPITAL]
            )

            # INBOUND (to dashboard hospital from variable centres)
            referring_facility_in = random.choice(
                ["PHC Mawlai", "CHC Smit", "CHC Pynursla", "Rural Health Center"]
            )
            receiving_facility_in = DASHBOARD_HOSPITAL

            case_id_ref = f"REF_{case_time.strftime('%Y%m%d')}_{case_num:03d}"
            case_id_rec = f"REC_{case_time.strftime('%Y%m%d')}_{case_num:03d}"

            referred_case = {
                "case_id": case_id_ref,
                "timestamp": case_time,
                "referring_facility": referring_facility_out,
                "receiving_facility": receiving_facility_out,
                "patient_name": random.choice(["A. Marbaniang", "S. Khonglam", "R. Syiem", "M. Lyngdoh", "Unknown"]),
                "patient_age": age,
                "patient_sex": "F" if case_type == "Maternal" else random.choice(["M", "F"]),
                "case_type": case_type,
                "icd_code": icd["icd_code"],
                "icd_label": icd["label"],
                "triage_color": random.choices(["RED", "YELLOW", "GREEN"], weights=[0.3, 0.5, 0.2])[0],
                "vitals": vitals,
                "interventions_referring": interventions,
                "status": random.choices(["Accepted", "Rejected", "Pending"], weights=[0.8, 0.1, 0.1])[0],
                "referral_reason_category": reason_category,
                "referral_reason_detail": reason_detail,
                "is_active_transfer": is_active_transfer,
                "eta_minutes": eta_minutes,
                "transit_updates": transit_updates,
                "clinical_notes": f"Patient presented with {case_type.lower()} symptoms requiring specialist care"
            }

            received_case = {
                **referred_case,
                "case_id": case_id_rec,
                "transport_time_minutes": transport_time,
                "emt_crew": emt_crew,
                "vehicle_id": emt_crew["vehicle"],
                "interventions_receiving": interventions + random.sample(
                    INTERVENTION_PROTOCOLS[case_type],
                    random.randint(1, 3)
                ),
                "final_outcome": random.choices(
                    ["Excellent", "Good", "Fair", "Poor"],
                    weights=[0.4, 0.3, 0.2, 0.1]
                )[0],
                "length_of_stay_hours": random.randint(24, 240),
                "referring_facility": referring_facility_in,
                "receiving_facility": receiving_facility_in,
                "discharge_date": case_time + timedelta(hours=random.randint(24, 240)),
                # inbound transfers more likely active to show 3–5 on any view
                "is_active_transfer": (random.random() < 0.25),
            }
            if received_case["is_active_transfer"]:
                received_case["eta_minutes"] = transport_time
                if not received_case.get("transit_updates"):
                    received_case["transit_updates"] = _make_transit_updates(
                        received_case["timestamp"],
                        received_case["vitals"],
                        transport_time
                    )

            referred_cases.append(referred_case)
            received_cases.append(received_case)

    # --- Ensure minimum active inbound transfers in last 48h ---
    recent_received_idx = [
        i for i, c in enumerate(received_cases)
        if (datetime.now() - c["timestamp"]) <= timedelta(hours=48)
    ]
    target_active = 4
    current_active = sum(
        1 for i in recent_received_idx if received_cases[i].get("is_active_transfer")
    )
    need = max(0, target_active - current_active)
    if need > 0 and recent_received_idx:
        promote = random.sample(recent_received_idx, min(need, len(recent_received_idx)))
        for i in promote:
            received_cases[i]["is_active_transfer"] = True
            tt = received_cases[i].get("transport_time_minutes", random.randint(20, 90))
            received_cases[i]["eta_minutes"] = tt
            if not received_cases[i].get("transit_updates"):
                received_cases[i]["transit_updates"] = _make_transit_updates(
                    received_cases[i]["timestamp"],
                    received_cases[i]["vitals"],
                    tt
                )

    ref_df = pd.DataFrame(referred_cases)
    rec_df = pd.DataFrame(received_cases)
    ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"])
    rec_df["timestamp"] = pd.to_datetime(rec_df["timestamp"])

    return {
        "referred_cases": ref_df,
        "received_cases": rec_df,
        "facilities": facilities,
        "facility_caps": facility_caps,
        "emt_crews": EMT_CREW
    }


# =========================
# SESSION STATE
# =========================
def initialize_session_state():
    if "premium_data" not in st.session_state:
        st.session_state.premium_data = generate_premium_synthetic_data(days_back=60)
    if "date_filters" not in st.session_state:
        st.session_state.date_filters = {
            "start_date": datetime.now().date() - timedelta(days=7),
            "end_date": datetime.now().date()
        }


# =========================
# UI BLOCKS
# =========================
def render_premium_header():
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.5rem;">🏥 AHECN Hospital Command Center</h1>
        <p style="margin:0; font-size:1.1rem; opacity:0.9;">
            Dashboard Hospital: <b>{DASHBOARD_HOSPITAL}</b> • Advanced Emergency Care Coordination Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.premium_data
    total_cases = len(data["referred_cases"])
    critical_cases = len(data["referred_cases"][data["referred_cases"]["triage_color"]=="RED"])
    urgent_cases = len(data["referred_cases"][data["referred_cases"]["triage_color"]=="YELLOW"])
    acceptance_rate = (len(data["referred_cases"][data["referred_cases"]["status"]=="Accepted"])/total_cases*100) if total_cases else 0
    avg_transport = data["received_cases"]["transport_time_minutes"].mean()

    active_outbound = len(data["referred_cases"][data["referred_cases"]["is_active_transfer"]==True])
    active_inbound = len(data["received_cases"][data["received_cases"]["is_active_transfer"]==True])

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1: st.markdown(f"<div class='metric-highlight'><div style='font-size:1.8rem;font-weight:700'>{total_cases}</div>Total Outbound</div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='metric-highlight'><div style='font-size:1.8rem;font-weight:700'>{critical_cases}</div>Outbound RED</div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='metric-highlight'><div style='font-size:1.8rem;font-weight:700'>{urgent_cases}</div>Outbound YELLOW</div>", unsafe_allow_html=True)
    with col4: st.markdown(f"<div class='metric-highlight'><div style='font-size:1.8rem;font-weight:700'>{acceptance_rate:.1f}%</div>Acceptance</div>", unsafe_allow_html=True)
    with col5: st.markdown(f"<div class='metric-highlight'><div style='font-size:1.8rem;font-weight:700'>{avg_transport:.0f}m</div>Avg Transport</div>", unsafe_allow_html=True)
    with col6: st.markdown(f"<div class='metric-highlight'><div style='font-size:1.8rem;font-weight:700'>{active_outbound}</div>Active Outbound</div>", unsafe_allow_html=True)
    with col7: st.markdown(f"<div class='metric-highlight'><div style='font-size:1.8rem;font-weight:700'>{active_inbound}</div>Active Inbound</div>", unsafe_allow_html=True)


def render_case_calendar():
    st.markdown("### 📅 Case Calendar & Timeline")
    col1, col2 = st.columns([2, 1])
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(st.session_state.date_filters["start_date"], st.session_state.date_filters["end_date"]),
            key="global_date_range_selector"
        )
        if isinstance(date_range, tuple) and len(date_range)==2:
            st.session_state.date_filters["start_date"] = date_range[0]
            st.session_state.date_filters["end_date"] = date_range[1]

        df = st.session_state.premium_data["referred_cases"]
        mask = (df["timestamp"].dt.date >= st.session_state.date_filters["start_date"]) & \
               (df["timestamp"].dt.date <= st.session_state.date_filters["end_date"])
        filtered = df[mask]

        if not filtered.empty:
            timeline = filtered.copy()
            timeline["date"] = timeline["timestamp"].dt.date
            timeline = timeline.groupby("date").size().reset_index(name="count")
            fig = px.line(timeline, x="date", y="count", title="Daily Outbound Case Volume Trend", markers=True)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cases found in selected date range")
    with col2:
        st.markdown("#### Quick Filters")
        st.multiselect("Case Types", options=["Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"],
                       default=["Maternal","Trauma","Cardiac"], key="case_type_filter")
        st.multiselect("Triage Levels", options=["RED","YELLOW","GREEN"],
                       default=["RED","YELLOW"], key="triage_filter")
        st.multiselect("Receiving Facilities",
                       options=[f["name"] for f in st.session_state.premium_data["facilities"]],
                       default=[f["name"] for f in st.session_state.premium_data["facilities"]],
                       key="facility_filter")


def render_interactive_case_list(case_type="referred"):
    key_prefix = f"{case_type}_cases"
    st.markdown(f"### 📋 {'Outbound' if case_type=='referred' else 'Inbound'} Referrals")
    cases_df = st.session_state.premium_data[f"{case_type}_cases"]

    # Active section preface for outbound
    if case_type == "referred":
        active = cases_df[cases_df["is_active_transfer"]==True].sort_values("eta_minutes").head(5)
        if not active.empty:
            st.markdown("#### 🚑 Active Outbound Transfers")
            for _, c in active.iterrows():
                st.info(f"{c['case_id']} → {c['receiving_facility']} | ETA {c.get('eta_minutes','-')} min | {c['triage_color']}")
            st.markdown("---")

    mask = (cases_df["timestamp"].dt.date >= st.session_state.date_filters["start_date"]) & \
           (cases_df["timestamp"].dt.date <= st.session_state.date_filters["end_date"])
    filtered_cases = cases_df[mask]

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        search_term = st.text_input("🔍 Search cases...", placeholder="case ID / facility / diagnosis", key=f"{key_prefix}_search")
    with col2:
        sort_by = st.selectbox("Sort by", ["Timestamp (Newest)","Timestamp (Oldest)","Triage","Case Type"], key=f"{key_prefix}_sort")
    with col3:
        items_per_page = st.selectbox("Items per page", [10,25,50], key=f"{key_prefix}_pagination")

    if search_term:
        filtered_cases = filtered_cases[
            filtered_cases["case_id"].str.contains(search_term, case=False) |
            filtered_cases["referring_facility"].str.contains(search_term, case=False) |
            filtered_cases["icd_label"].str.contains(search_term, case=False) |
            filtered_cases["receiving_facility"].str.contains(search_term, case=False)
        ]

    # Sort
    if sort_by == "Timestamp (Newest)":
        filtered_cases = filtered_cases.sort_values("timestamp", ascending=False)
    elif sort_by == "Timestamp (Oldest)":
        filtered_cases = filtered_cases.sort_values("timestamp", ascending=True)
    elif sort_by == "Triage":
        triage_order = {"RED":1,"YELLOW":2,"GREEN":3}
        filtered_cases = filtered_cases.sort_values("triage_color", key=lambda x: x.map(triage_order))
    else:
        filtered_cases = filtered_cases.sort_values("case_type", ascending=True)

    if filtered_cases.empty:
        st.info("No cases found matching current filters")
        return

    total_pages = int(len(filtered_cases)/items_per_page) + (1 if len(filtered_cases)%items_per_page else 0)
    total_pages = max(total_pages, 1)
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=f"{key_prefix}_page")

    start_idx = (page_number-1) * items_per_page
    end_idx = start_idx + items_per_page
    page_df = filtered_cases.iloc[start_idx:end_idx]

    st.write(f"Showing {len(page_df)} of {len(filtered_cases)} cases")
    for display_idx, (_, case) in enumerate(page_df.iterrows()):
        render_case_card(case, case_type, display_idx)


def render_case_card(case, case_type, index):
    triage_map = {"RED":"critical","YELLOW":"urgent","GREEN":"stable"}
    case_class = f"case-card {triage_map.get(case['triage_color'], '')}"

    st.markdown(f"""
    <div class="{case_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div><strong>{case['case_id']}</strong> • {case['icd_label']}</div>
            <div style="background:{'#ff4444' if case['triage_color']=='RED' else '#ffaa00' if case['triage_color']=='YELLOW' else '#00c853'};
                        color:white;padding:0.2rem 0.8rem;border-radius:15px;font-size:0.8rem;">
                {case['triage_color']}
            </div>
        </div>
        <div style="color:#666;font-size:0.9rem;margin-top:0.5rem;">
            {case['referring_facility']} → {case['receiving_facility']} •
            {case['timestamp'].strftime('%Y-%m-%d %H:%M')} •
            {case['patient_age']}{case['patient_sex']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"View full details for {case['case_id']}"):
        render_case_details(case, case_type)


def _render_referral_reasons(case):
    """Handle both old single-reason and new multi-reason structures."""
    cat = case.get("referral_reason_category", "-")
    det = case.get("referral_reason_detail", "-")
    other = case.get("referral_reason_other_notes", "")

    if isinstance(cat, list):
        st.write(f"**Category:** {', '.join(cat) if cat else '-'}")
        if isinstance(det, dict):
            for k, v in det.items():
                if v:
                    st.write(f"• {k}: {', '.join(v)}")
        else:
            st.write(f"**Detail:** {det}")
    else:
        st.write(f"**Category:** {cat}")
        st.write(f"**Detail:** {det}")

    if other:
        st.caption(f"Other notes: {other}")


def render_case_details(case, case_type):
    col1, col2 = st.columns([1,1])

    vitals = case.get("vitals", {}) or {}
    age = float(case.get("patient_age", 0) or 0)
    sbp = float(vitals.get("sbp", 0) or 0)
    spo2 = float(vitals.get("spo2", 0) or 0)
    hr = float(vitals.get("hr", 0) or 0)
    rr = float(vitals.get("rr", 0) or 0)
    temp_c = float(vitals.get("temp", 36.5) or 36.5)
    avpu = vitals.get("avpu","A") or "A"

    with col1:
        st.markdown("#### Patient Information")
        st.write(f"**Case ID:** {case.get('case_id','-')}")
        if case.get("patient_name"):
            st.write(f"**Name:** {case.get('patient_name')}")
        st.write(f"**Age/Sex:** {case.get('patient_age','-')}{case.get('patient_sex','-')}")
        st.write(f"**Diagnosis:** {case.get('icd_label','-')} ({case.get('icd_code','-')})")
        st.write(f"**Triage (original):** {case.get('triage_color','-')}")
        st.write(f"**Referring Facility:** {case.get('referring_facility','-')}")
        st.write(f"**Receiving Facility:** {case.get('receiving_facility','-')}")

        st.markdown("#### Vital Signs")
        st.write(f"**HR:** {hr:.0f} bpm | **SBP:** {sbp:.0f} mmHg")
        st.write(f"**RR:** {rr:.0f} /min | **SpO₂:** {spo2:.0f}%")
        st.write(f"**Temp:** {temp_c:.1f}°C | **AVPU:** {avpu}")

        st.markdown("#### Referral Reason")
        _render_referral_reasons(case)

        if case.get("is_active_transfer"):
            st.warning(f"🚑 Active Transfer • ETA: {case.get('eta_minutes','-')} mins")
            st.markdown("#### In-Transit Monitoring")
            for upd in case.get("transit_updates", []):
                st.write(
                    f"**{upd['timestamp'].strftime('%H:%M')}**  "
                    f"HR {upd['vitals']['hr']} | SBP {upd['vitals']['sbp']} | "
                    f"RR {upd['vitals']['rr']} | SpO₂ {upd['vitals']['spo2']}%"
                )
                st.caption(f"EMT note: {upd.get('emt_note','-')}")
        else:
            st.markdown("#### Transfer Summary (Completed)")
            if case.get("transit_updates"):
                last = case["transit_updates"][-1]
                st.write(
                    "Last in-transit vitals:",
                    f"HR {last['vitals']['hr']}, SBP {last['vitals']['sbp']}, SpO₂ {last['vitals']['spo2']}%"
                )
                st.caption(f"Last EMT note: {last.get('emt_note','-')}")
            else:
                st.caption("No EMT log recorded for this completed transfer (synthetic).")

    with col2:
        st.markdown("#### Timeline & Interventions")
        st.markdown("**Referring Interventions:**")
        for itv in case.get("interventions_referring", []):
            st.markdown(f'<div class="intervention-badge">{itv}</div>', unsafe_allow_html=True)

        if case_type=="received":
            st.markdown("**Receiving Interventions:**")
            for itv in case.get("interventions_receiving", []):
                st.markdown(f'<div class="intervention-badge">{itv}</div>', unsafe_allow_html=True)

            st.markdown("#### Transport Details")
            st.write(f"**Transport Time:** {case.get('transport_time_minutes','-')} minutes")
            crew = case.get('emt_crew', {}) or {}
            st.write(f"**EMT Crew:** {crew.get('name','-')} ({crew.get('level','-')})")
            st.write(f"**Vehicle:** {case.get('vehicle_id','-')}")
            st.write(f"**Outcome:** {case.get('final_outcome','-')}")
            st.write(f"**Length of Stay:** {case.get('length_of_stay_hours','-')} hours")

    # =========================
    # AI / SCORE-BASED TRIAGE PANEL
    # =========================
    st.markdown("#### 🧠 Clinical Triage Recommendation (Score-Based)")
    st.caption("MEOWS for maternal, NEWS2 + qSOFA for adults, PEWS placeholder for paediatrics.")

    oxy_key = get_unique_key("oxy", case_type, case)
    spo2s2_key = get_unique_key("spo2s2", case_type, case)

    colA, colB = st.columns(2)
    with colA:
        on_oxygen = st.checkbox("Patient on supplemental oxygen?", value=False, key=oxy_key)
    with colB:
        spo2_scale2 = st.checkbox("Use SpO₂ Scale 2 (COPD/hypercapnic)?", value=False, key=spo2s2_key)

    triage_color, details = score_based_triage(case, on_oxygen=on_oxygen, spo2_scale2=spo2_scale2)

    last_pred_key = get_unique_key("ai_last_pred", case_type, case)
    last_feat_key = get_unique_key("ai_last_features", case_type, case)
    st.session_state[last_pred_key] = triage_color
    st.session_state[last_feat_key] = [age, sbp, spo2, hr]

    color_map = {"RED":"#ff4444","YELLOW":"#ffaa00","GREEN":"#00c853"}
    triage_hex = color_map.get(triage_color, "#999999")

    st.markdown(f"""
    <div style="margin:0.5rem 0 1rem 0;padding:1rem 1.2rem;border-radius:12px;
                background:{triage_hex}15;border-left:6px solid {triage_hex};">
        <h4 style="margin:0;color:{triage_hex};">Recommended Triage: {triage_color}</h4>
        <div style="color:#333;margin-top:0.5rem;">
            <b>System:</b> {details.get("system")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    system = details.get("system", "")

    if system.startswith("NEWS2"):
        st.write(
            f"**NEWS2 Total:** {details['news2_total']}  (RR:{details['news2_parts']['rr']}, "
            f"SpO₂:{details['news2_parts']['spo2']}, O₂:{details['news2_parts']['oxygen']}, "
            f"SBP:{details['news2_parts']['sbp']}, HR:{details['news2_parts']['hr']}, "
            f"Temp:{details['news2_parts']['temp']}, AVPU:{details['news2_parts']['conc']})"
        )
        st.write(f"**qSOFA:** {details['qsofa']} (≥2 upgrades risk)")
    elif system.startswith("MEOWS"):
        st.markdown("**MEOWS Breakdown (Primary Maternal Score)**")
        triggers = details.get("meows_triggers", {}) or {}
        meows_rows = [{"Parameter": k, "Trigger": v} for k, v in triggers.items()]
        st.table(pd.DataFrame(meows_rows))
        st.write(f"**MEOWS Amber count:** {details.get('meows_total', '-')}")
        st.caption(details.get("notes", ""))

        with st.expander("See Secondary Adult Comparator (NEWS2 + qSOFA)"):
            st.write(f"**NEWS2 Total:** {details.get('secondary_news2_total', '-')}")
            parts = details.get("secondary_news2_parts", {}) or {}
            if parts:
                st.write(
                    f"(RR:{parts.get('rr')}, SpO₂:{parts.get('spo2')}, O₂:{parts.get('oxygen')}, "
                    f"SBP:{parts.get('sbp')}, HR:{parts.get('hr')}, Temp:{parts.get('temp')}, "
                    f"AVPU:{parts.get('conc')})"
                )
            st.write(f"**qSOFA:** {details.get('secondary_qsofa', '-')}")
            st.caption("Comparator only; not used to drive maternal triage.")
    else:
        st.write(f"**PEWS Score (placeholder):** {details.get('score', '-')}")
        st.caption(details.get("notes", ""))

    # Optional ML model comparison
    model = get_triage_model()
    if model is not None:
        st.markdown("#### 🤖 ML Model Suggestion (Optional)")
        try:
            features = np.array([[age, sbp, spo2, hr]], dtype=float)
            ml_pred = str(model.predict(features)[0])
            st.write(f"**Model output:** {ml_pred}")
            st.caption("Demo only. Production model must be clinically validated.")
        except Exception as e:
            st.caption(f"ML model available but could not run here: {e}")

    # Feedback UI
    st.subheader("Was this recommendation helpful?")
    fb_radio_key = get_unique_key("ai_fb_radio", case_type, case)
    feedback = st.radio("Feedback", ("Yes","No","Unsure"), key=fb_radio_key, horizontal=True)

    fb_submit_key = get_unique_key("ai_fb_submit", case_type, case)
    if st.button("Submit Feedback", key=fb_submit_key):
        log_ai_feedback(case, st.session_state[last_feat_key], st.session_state[last_pred_key], feedback)
        st.success("Thanks — feedback logged.")


def render_advanced_analytics():
    st.markdown("### 📊 Advanced Analytics Dashboard")
    data = st.session_state.premium_data
    df = data["referred_cases"]
    mask = (df["timestamp"].dt.date >= st.session_state.date_filters["start_date"]) & \
           (df["timestamp"].dt.date <= st.session_state.date_filters["end_date"])
    filtered_cases = df[mask]

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Outbound", len(filtered_cases))
    with col2:
        total = len(filtered_cases)
        ar = (len(filtered_cases[filtered_cases["status"]=="Accepted"])/total*100) if total else 0
        st.metric("Acceptance Rate", f"{ar:.1f}%")
    with col3:
        avg_transport = data["received_cases"]["transport_time_minutes"].mean()
        st.metric("Avg Transport Time", f"{avg_transport:.1f} min")
    with col4: st.metric("Outbound RED", len(filtered_cases[filtered_cases["triage_color"]=="RED"]))

    if filtered_cases.empty:
        st.info("No data available for analytics with current filters")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Case Type Analysis")
        case_counts = filtered_cases["case_type"].value_counts()
        st.plotly_chart(px.pie(values=case_counts.values, names=case_counts.index, title="Outbound by Type"),
                        use_container_width=True)
    with c2:
        st.markdown("#### Triage Distribution")
        triage_counts = filtered_cases["triage_color"].value_counts()
        fig = px.bar(x=triage_counts.index, y=triage_counts.values, color=triage_counts.index,
                     color_discrete_map={'RED':'#ff4444','YELLOW':'#ffaa00','GREEN':'#00c853'},
                     title="Outbound by Triage")
        st.plotly_chart(fig, use_container_width=True)


def render_emt_tracking():
    st.markdown("### 🚑 Real-time EMT Tracking")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("#### Active Transports (Outbound)")
        active_transports = st.session_state.premium_data["referred_cases"]
        active_transports = active_transports[active_transports["is_active_transfer"]==True].head(5)

        if active_transports.empty:
            st.info("No active outbound transports right now (synthetic).")
        for _, transport in active_transports.iterrows():
            progress = min(95, random.randint(30,90))
            st.markdown(f"""
            <div class="premium-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <strong>{transport['case_id']}</strong><br>
                        <small>{transport['receiving_facility']} • {transport['triage_color']}</small>
                    </div>
                    <div style="background:#e3f2fd;padding:0.5rem 1rem;border-radius:10px;">
                        {progress}% Complete
                    </div>
                </div>
                <div style="margin-top:1rem;">
                    <div style="background:#f0f0f0;border-radius:10px;height:10px;">
                        <div style="background:#2196f3;width:{progress}%;height:100%;border-radius:10px;"></div>
                    </div>
                </div>
                <div style="margin-top:0.5rem;font-size:0.9rem;color:#666;">
                    ETA: {transport.get('eta_minutes', '-') } min • {transport['referring_facility']} → {transport['receiving_facility']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("#### Crew Status")
        for idx, crew in enumerate(st.session_state.premium_data["emt_crews"]):
            status_color = "🟢" if crew["status"]=="active" else "🟡" if crew["status"]=="available" else "🔴"
            st.write(f"{status_color} **{crew['name']}**")
            st.write(f"   {crew['level']} • {crew['vehicle']}")
            if idx < len(st.session_state.premium_data["emt_crews"]) - 1:
                st.write("---")


def render_quick_actions():
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh All Data", use_container_width=True, key="refresh_data_btn_main"):
            st.session_state.premium_data = generate_premium_synthetic_data(days_back=60)
            st.success("Data refreshed successfully!")
    with col2:
        if st.button("🚨 Emergency Mode", use_container_width=True, type="secondary", key="emergency_mode_btn_main"):
            st.warning("Emergency mode activated - prioritizing critical cases")
    with col3:
        if st.button("🖨️ Export Data", use_container_width=True, key="export_data_btn_main"):
            st.info("Preparing data export... (demo)")


def render_dashboard_overview():
    col1, col2 = st.columns([2,1])
    with col1:
        render_case_calendar()
        st.markdown("### 📈 Recent Outbound Activity")
        recent_cases = st.session_state.premium_data["referred_cases"].tail(10)
        for _, case in recent_cases.iterrows():
            st.markdown(f"""
            <div class="timeline-event">
                <strong>{case['case_id']}</strong> - {case['icd_label']}<br>
                <small>{case['timestamp'].strftime('%Y-%m-%d %H:%M')} • {case['triage_color']} • {case['receiving_facility']}</small>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("### 🖥️ System Status")
        status_items = [
            {"name":"Database","status":"Optimal","color":"success"},
            {"name":"API Services","status":"Stable","color":"success"},
            {"name":"EMT Tracking","status":"Active","color":"success"},
            {"name":"Analytics","status":"Processing","color":"warning"},
        ]
        for item in status_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0;">
                <span>{item['name']}</span>
                <span style="background:{'#00c853' if item['color']=='success' else '#ffab00'};
                            color:white;padding:0.2rem 0.8rem;border-radius:15px;font-size:0.8rem;">
                    {item['status']}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔥 Referral Pressure")
        reasons = st.session_state.premium_data["referred_cases"].copy()
        def _main_reason(x):
            if isinstance(x, str):
                return x
            if isinstance(x, list) and x:
                return x[0]
            return "NA"
        reasons["reason_main"] = reasons["referral_reason_category"].apply(_main_reason)
        rc = reasons["reason_main"].value_counts()
        st.bar_chart(rc)
        st.caption("Driver split: beds vs specialty vs equipment (synthetic).")


def render_premium_sidebar():
    st.sidebar.markdown("### 🔔 Live Alerts")
    critical = st.session_state.premium_data["referred_cases"]
    critical = critical[critical["triage_color"]=="RED"].tail(3)
    for _, case in critical.iterrows():
        st.sidebar.error(f"**{case['case_id']}**\n\n{case['icd_label']}\n\n*{case['timestamp'].strftime('%H:%M')} • {case['receiving_facility']}*")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Performance")
    metrics_df = st.session_state.premium_data["referred_cases"]
    acceptance_rate = (len(metrics_df[metrics_df["status"]=="Accepted"])/len(metrics_df)*100) if len(metrics_df) else 0
    st.sidebar.metric("Response Rate", f"{acceptance_rate:.1f}%")
    st.sidebar.metric("Avg Decision Time", "8.2 min")
    st.sidebar.metric("System Uptime", "99.9%")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Tools")
    if st.sidebar.button("🔄 Force Refresh", key="sidebar_refresh_btn"):
        st.session_state.premium_data = generate_premium_synthetic_data(days_back=60)
        st.rerun()


def render_new_referral_form():
    st.markdown("### ➕ Create New Outbound Referral")
    st.caption("Create a referral from this dashboard hospital and auto-match the best receiving facility.")

    with st.form("new_referral_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            patient_name = st.text_input("Patient Name (demo)", key="new_ref_name")
            patient_age = st.number_input("Patient Age", 0, 120, 40, key="new_ref_age")
            patient_sex = st.selectbox("Sex", ["M","F"], key="new_ref_sex")
            case_type = st.selectbox("Case Type", ["Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"], key="new_ref_case_type")
            icd_options = [f"{i['icd_code']} — {i['label']}" for i in ICD_CATALOG if i["case_type"]==case_type]
            icd_choice = st.selectbox("Provisional Diagnosis (ICD)", icd_options, key="new_ref_icd")

            icd_code_preview = icd_choice.split("—")[0].strip()
            base_intv_preview = ICD_INTERVENTION_MAP.get(icd_code_preview, INTERVENTION_PROTOCOLS[case_type])
            st.markdown("**Suggested Referring Interventions (auto-protocol)**")
            st.write(", ".join(base_intv_preview[:6]))

        with col2:
            hr = st.number_input("Heart Rate (bpm)", 20, 220, 90, key="new_ref_hr")
            sbp = st.number_input("SBP (mmHg)", 50, 260, 110, key="new_ref_sbp")
            rr = st.number_input("Resp Rate (/min)", 5, 60, 22, key="new_ref_rr")
            spo2 = st.number_input("SpO₂ (%)", 50, 100, 95, key="new_ref_spo2")
            temp_c = st.number_input("Temp (°C)", 34.0, 42.0, 37.2, step=0.1, key="new_ref_temp")
            avpu = st.selectbox("AVPU", ["A","V","P","U"], key="new_ref_avpu")

        clinical_notes = st.text_area("Clinical Notes / Why referral?", key="new_ref_notes")

        st.markdown("#### Reason(s) for Referral")
        reason_checks = st.multiselect(
            "Tick all applicable reasons",
            ["ICU_BED_UNAVAILABLE", "SPECIALTY_REQUIRED", "EQUIPMENT_REQUIRED"],
            default=["SPECIALTY_REQUIRED"],
            key="new_ref_reason_multi"
        )

        reason_detail_map = {}
        if "ICU_BED_UNAVAILABLE" in reason_checks:
            reason_detail_map["ICU_BED_UNAVAILABLE"] = st.multiselect(
                "ICU / bed related",
                REFERRAL_REASONS["ICU_BED_UNAVAILABLE"],
                key="new_ref_reason_icu"
            )
        if "SPECIALTY_REQUIRED" in reason_checks:
            reason_detail_map["SPECIALTY_REQUIRED"] = st.multiselect(
                "Specialty required",
                REFERRAL_REASONS["SPECIALTY_REQUIRED"][case_type],
                key="new_ref_reason_spec"
            )
        if "EQUIPMENT_REQUIRED" in reason_checks:
            reason_detail_map["EQUIPMENT_REQUIRED"] = st.multiselect(
                "Equipment required",
                REFERRAL_REASONS["EQUIPMENT_REQUIRED"][case_type],
                key="new_ref_reason_eqp"
            )
        other_reason_notes = st.text_area("Other referral notes (free text)", key="new_ref_reason_other")

        # Real-time triage preview
        tmp_case = {
            "case_type": case_type,
            "patient_age": patient_age,
            "vitals": {"hr":hr,"sbp":sbp,"rr":rr,"spo2":spo2,"temp":temp_c,"avpu":avpu}
        }
        cA, cB = st.columns(2)
        with cA: on_oxygen = st.checkbox("On oxygen?", value=False, key="new_ref_oxygen")
        with cB: spo2_scale2 = st.checkbox("SpO₂ Scale 2 (COPD)?", value=False, key="new_ref_spo2s2")

        triage_for_match, triage_details = score_based_triage(
            tmp_case, on_oxygen=on_oxygen, spo2_scale2=spo2_scale2
        )
        st.info(f"Triage preview: **{triage_for_match}** via {triage_details.get('system')}")

        # Facility matching uses dominant reason (first tick) for scoring
        dominant_reason = reason_checks[0] if reason_checks else "SPECIALTY_REQUIRED"

        facility_caps = st.session_state.premium_data.get("facility_caps", {})
        scored = []
        for fname, fcap in facility_caps.items():
            if fname == DASHBOARD_HOSPITAL:
                continue
            score, breakdown = score_facility_for_case(case_type, triage_for_match, dominant_reason, fcap)
            eta = int(breakdown["distance_km"] * 1.2 + 15)  # road factor
            scored.append({
                "Facility": fname,
                "Score": score,
                "Distance (km)": breakdown["distance_km"],
                "ETA (min)": eta,
                "ICU Beds Free": breakdown["icu_available"],
                "Occupancy %": breakdown["occupancy_pct"],
                "Type": breakdown["facility_type"],
                "Has Required Capability": "✅" if breakdown["capability_score"] >= 25 else "⚠️",
                "Why matched": f"Cap:{breakdown['capability_score']} + Dist:{breakdown['distance_score']}"
            })
        scored_df = pd.DataFrame(scored).sort_values("Score", ascending=False)

        st.markdown("#### Smart Receiving Facility Match (Auto-ranked)")
        st.table(scored_df.reset_index(drop=True))

        top_facilities = scored_df["Facility"].tolist()
        receiving_facility = st.selectbox(
            "Receiving Facility (auto-ranked)",
            options=top_facilities,
            index=0 if top_facilities else None,
            key="new_ref_recv_fac"
        )

        submitted = st.form_submit_button("🚀 Create Referral", type="primary")

    if submitted:
        case_time = datetime.now()
        case_num = random.randint(0,999)
        case_id_ref = f"REF_{case_time.strftime('%Y%m%d')}_{case_num:03d}"

        icd_code = icd_choice.split("—")[0].strip()
        icd_label = icd_choice.split("—")[1].strip()

        base_intv = ICD_INTERVENTION_MAP.get(icd_code, INTERVENTION_PROTOCOLS[case_type])
        interventions = random.sample(base_intv, min(len(base_intv), 3))

        eta_pick = 60
        if not scored_df.empty and receiving_facility in scored_df["Facility"].values:
            eta_pick = int(scored_df.loc[scored_df["Facility"]==receiving_facility, "ETA (min)"].iloc[0])

        referred_case = {
            "case_id": case_id_ref,
            "timestamp": case_time,
            "referring_facility": DASHBOARD_HOSPITAL,
            "receiving_facility": receiving_facility,
            "patient_name": patient_name or "Unknown",
            "patient_age": patient_age,
            "patient_sex": patient_sex,
            "case_type": case_type,
            "icd_code": icd_code,
            "icd_label": icd_label,
            "triage_color": triage_for_match,
            "vitals": {"hr":hr,"sbp":sbp,"rr":rr,"spo2":spo2,"temp":temp_c,"avpu":avpu},
            "interventions_referring": interventions,
            "status": "Pending",
            "referral_reason_category": reason_checks,
            "referral_reason_detail": reason_detail_map,
            "referral_reason_other_notes": other_reason_notes,
            "is_active_transfer": True,
            "eta_minutes": eta_pick,
            "transit_updates": [],
            "clinical_notes": clinical_notes or "New referral created"
        }

        ref_df = st.session_state.premium_data["referred_cases"]
        ref_df = pd.concat([ref_df, pd.DataFrame([referred_case])], ignore_index=True)
        ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"])
        st.session_state.premium_data["referred_cases"] = ref_df

        st.success(f"Referral created: {case_id_ref} → {receiving_facility}")


# =========================
# MAIN
# =========================
def main():
    initialize_session_state()
    render_premium_header()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard",
        "📤 Refer / Outbound Referrals",
        "🏥 Inbound Referrals",
        "📊 Analytics",
        "🚑 EMT Tracking",
        "⚡ Quick Actions",
    ])

    with tab1:
        render_dashboard_overview()
    with tab2:
        render_new_referral_form()
        st.markdown("---")
        render_interactive_case_list("referred")
    with tab3:
        render_interactive_case_list("received")
    with tab4:
        render_advanced_analytics()
    with tab5:
        render_emt_tracking()
    with tab6:
        render_quick_actions()

    render_premium_sidebar()


if __name__ == "__main__":
    main()
