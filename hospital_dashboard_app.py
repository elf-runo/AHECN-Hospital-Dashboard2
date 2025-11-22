# hospital_dashboard_app.py
"""
AHECN Hospital Command Center — Updated Demo MVP (Investor / AI-Conclave Grade)
-------------------------------------------------------------------------------
Single-file Streamlit app with internal mini-services, SQLite auditability,
config-driven triage protocols, multi-constraint facility matching, capacity market,
pre-arrival deterioration digital twin, SLA escalation, offline store-and-forward,
and safe-guarded ML comparator.

Key upgrades vs previous MVP:
A) Clinical safety & governance:
   - PEWS placeholder replaced with configurable age-banded PEWS v1
   - MEOWS versioned + maternal context modifiers
   - Immutable audit log (hash-chained)
B) Product realism:
   - Multi-reason + hard/soft constraint facility matching
   - Facility registry + live availability stub (no random capabilities in matching)
   - Terrain/TOD-aware ETA
C) Scalability posture:
   - Internal mini-services + SQLite persistence
   - RBAC demo login
   - FHIR-lite export placeholders
D) Data/ML readiness:
   - Full feature logging + outcome adjudication
   - ML comparator recommender-only
E) Adoption/Ops:
   - Referral state machine + SLA escalation
   - Offline queue
   - Human handshake comms placeholders
Moat ideas:
   1) District Capacity Market dashboard feeding routing bonus
   2) Pre-arrival Digital Twin (deterioration risk forecast)

Notes:
- This is still a demo MVP. Clinical protocols MUST be validated/approved
  per hospital/state prior to real deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random
import math
import os
import json
import sqlite3
import hashlib
import time
import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
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


# joblib optional – used only if available
try:
    import joblib
except ImportError:
    joblib = None


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AHECN Hospital Command Center (Updated MVP)",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# =============================================================================
# PREMIUM CSS (kept compatible with original styling)
# =============================================================================
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

.badge-red {background:#ffebee;color:#c62828;padding:0.2rem 0.5rem;border-radius:8px;font-weight:600;}
.badge-yellow {background:#fff8e1;color:#f57f17;padding:0.2rem 0.5rem;border-radius:8px;font-weight:600;}
.badge-green {background:#e8f5e9;color:#2e7d32;padding:0.2rem 0.5rem;border-radius:8px;font-weight:600;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PATHS & STORAGE
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DB_PATH = str(APP_DIR / "ahecn_mvp.db")
FACILITY_REGISTRY_PATH = str(APP_DIR / "facility_registry.json")
PROTOCOLS_PATH = str(APP_DIR / "protocols.json")
FEEDBACK_LOG_PATH = str(APP_DIR / "ai_feedback_log.csv")  # optional legacy CSV


# =============================================================================
# CONSTANTS & CATALOGS
# =============================================================================
DASHBOARD_HOSPITAL = "Tertiary Central Hospital"
DASHBOARD_LAT = 25.578
DASHBOARD_LON = 91.893

CASE_TYPES = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]
TRIAGE_COLORS = ["RED", "YELLOW", "GREEN"]

REFERRAL_STATE_FLOW = [
    "REQUESTED",
    "ACKNOWLEDGED",
    "ACCEPTED",
    "REJECTED",
    "DISPATCHED",
    "ENROUTE",
    "ARRIVED",
    "CLOSED",
    "QUEUED_OFFLINE"
]


# ICD seeds per category + auto-generated demo ICDs
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
        "Maternal": ["Blood bank / massive transfusion", "USG/CT obstetric imaging", "Operating theatre"],
        "Trauma": ["CT/MRI trauma imaging", "Operating theatre", "Ventilator support"],
        "Stroke": ["CT/MRI + thrombolysis setting"],
        "Cardiac": ["Cath lab / PCI suite", "Ventilator support"],
        "Sepsis": ["Ventilator / dialysis support"],
        "Other": ["Advanced respiratory or imaging support"]
    }
}

# Normalize capabilities we track
CAP_KEYS = [
    "ICU_BED",
    "VENTILATOR",
    "CT",
    "MRI",
    "CATHLAB",
    "BLOODBANK",
    "OR",
    "NEURO_SPECIALIST",
    "CARDIO_SPECIALIST",
    "OBGYN_SPECIALIST"
]

# Map detailed reasons -> required capabilities
REASON_TO_CAPS = {
    # Specialty required
    "ObGyn emergency surgery / LSCS": ["OBGYN_SPECIALIST", "OR", "BLOODBANK"],
    "Maternal ICU monitoring": ["ICU_BED", "OBGYN_SPECIALIST", "VENTILATOR"],
    "Neurosurgery / trauma surgeon": ["NEURO_SPECIALIST", "OR", "CT", "ICU_BED"],
    "Ortho trauma fixation": ["OR", "CT", "ICU_BED"],
    "Stroke unit / thrombolysis capability": ["NEURO_SPECIALIST", "CT", "ICU_BED"],
    "Cath lab / cardiologist": ["CATHLAB", "CARDIO_SPECIALIST", "ICU_BED"],
    "Cardiac ICU": ["ICU_BED", "CARDIO_SPECIALIST", "VENTILATOR"],
    "Critical care intensivist": ["ICU_BED", "VENTILATOR"],
    "Organ support": ["ICU_BED", "VENTILATOR"],
    "Specialist consult needed": ["ICU_BED"],

    # Equipment required
    "Blood bank / massive transfusion": ["BLOODBANK", "ICU_BED"],
    "USG/CT obstetric imaging": ["CT"],
    "CT/MRI trauma imaging": ["CT", "MRI"],
    "CT/MRI + thrombolysis setting": ["CT", "MRI", "NEURO_SPECIALIST"],
    "Cath lab / PCI suite": ["CATHLAB"],
    "Ventilator support": ["VENTILATOR", "ICU_BED"],
    "Ventilator / dialysis support": ["VENTILATOR", "ICU_BED"],
    "Advanced respiratory or imaging support": ["VENTILATOR", "CT"],

    # ICU bed unavailable
    "No ICU beds free at referring facility": ["ICU_BED"],
    "Ventilator bed unavailable": ["ICU_BED", "VENTILATOR"],
    "High-dependency unit full": ["ICU_BED"],
}

HARD_CAPS = set([
    "ICU_BED", "VENTILATOR", "CT", "MRI", "CATHLAB",
    "BLOODBANK", "OR", "NEURO_SPECIALIST", "CARDIO_SPECIALIST", "OBGYN_SPECIALIST"
])

EMT_CREW = [
    {"id": "EMT_001", "name": "John Carter", "level": "ALS", "vehicle": "Ambulance 1", "status": "active"},
    {"id": "EMT_002", "name": "Sarah Connor", "level": "BLS", "vehicle": "Ambulance 2", "status": "active"},
    {"id": "EMT_003", "name": "Mike Rodriguez", "level": "ALS", "vehicle": "Ambulance 3", "status": "available"},
    {"id": "EMT_004", "name": "Lisa Park", "level": "Critical Care", "vehicle": "Mobile ICU", "status": "available"},
]

# ETA speed profiles (km/h)
SPEED_PROFILES = {
    "flat_day": 45,
    "hilly_day": 28,
    "flat_night": 35,
    "hilly_night": 22,
}


# =============================================================================
# MINI-SERVICES
# =============================================================================
class AuditService:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_tables(self):
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS audit_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, user TEXT, role TEXT,
                case_id TEXT, event_type TEXT,
                payload TEXT, prev_hash TEXT, hash TEXT
            )""")

    def log(self, user, role, case_id, event_type, payload):
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1")
            prev = cur.fetchone()
            prev_hash = prev[0] if prev else "GENESIS"
            h = hashlib.sha256((prev_hash + payload_str + str(time.time())).encode()).hexdigest()
            cur.execute(
                "INSERT INTO audit_log(ts,user,role,case_id,event_type,payload,prev_hash,hash) VALUES(?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), user, role, case_id, event_type, payload_str, prev_hash, h)
            )
            conn.commit()

    def recent(self, limit=200):
        with self._conn() as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM audit_log ORDER BY id DESC LIMIT {int(limit)}", conn
            )
        return df


class DataService:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_tables(self):
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS referrals(
                case_id TEXT PRIMARY KEY,
                payload TEXT,
                created_at TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS outcomes(
                case_id TEXT PRIMARY KEY,
                payload TEXT,
                adjudicated_at TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS offline_queue(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT,
                queued_at TEXT,
                synced INTEGER DEFAULT 0
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS facilities(
                facility_id TEXT PRIMARY KEY,
                payload TEXT,
                updated_at TEXT
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS ml_features(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT,
                payload TEXT,
                captured_at TEXT
            )""")

    # Facilities
    def upsert_facilities(self, facilities):
        with self._conn() as conn:
            for f in facilities:
                conn.execute(
                    "INSERT OR REPLACE INTO facilities(facility_id,payload,updated_at) VALUES(?,?,?)",
                    (f["facility_id"], json.dumps(f, default=str), datetime.now(timezone.utc).isoformat())
                )
            conn.commit()

    def load_facilities(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT payload FROM facilities").fetchall()
        if not rows:
            return []
        return [json.loads(r[0]) for r in rows]

    # Referrals
    def save_referral(self, case):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO referrals(case_id,payload,created_at) VALUES(?,?,?)",
                (case["case_id"], json.dumps(case, default=str), datetime.now(timezone.utc).isoformat())
            )
            conn.commit()

    def load_referrals(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT payload FROM referrals").fetchall()

        cases = []
        for r in rows:
            c = json.loads(r[0])

            # normalize top-level datetimes
            for k in ["timestamp", "requested_at", "status_updated_at", "sla_escalated_at"]:
                if k in c and c[k] is not None:
                    c[k] = parse_dt(c[k])

            # normalize transit update timestamps
            tu = c.get("transit_updates", []) or []
            for u in tu:
                if "timestamp" in u:
                    u["timestamp"] = parse_dt(u["timestamp"])
            c["transit_updates"] = tu

            cases.append(c)

        return cases
       
    # Outcomes
    def save_outcome(self, case_id, outcome_payload):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO outcomes(case_id,payload,adjudicated_at) VALUES(?,?,?)",
                (case_id, json.dumps(outcome_payload, default=str), datetime.now(timezone.utc).isoformat())
            )
            conn.commit()

    def load_outcomes(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT case_id, payload FROM outcomes").fetchall()
        data = {}
        for case_id, payload in rows:
            data[case_id] = json.loads(payload)
        return data

    # Offline
    def queue_offline(self, case):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO offline_queue(payload,queued_at,synced) VALUES(?,?,0)",
                (json.dumps(case, default=str), datetime.now(timezone.utc).isoformat())
            )
            conn.commit()

    def load_offline_unsynced(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT id,payload FROM offline_queue WHERE synced=0").fetchall()
        return [{"id": r[0], "case": json.loads(r[1])} for r in rows]

    def mark_offline_synced(self, qid):
        with self._conn() as conn:
            conn.execute("UPDATE offline_queue SET synced=1 WHERE id=?", (qid,))
            conn.commit()

    # ML features
    def log_ml_features(self, case_id, payload):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO ml_features(case_id,payload,captured_at) VALUES(?,?,?)",
                (case_id, json.dumps(payload, default=str), datetime.now(timezone.utc).isoformat())
            )
            conn.commit()

    def load_ml_features(self, limit=1000):
        with self._conn() as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM ml_features ORDER BY id DESC LIMIT {int(limit)}", conn
            )
        return df


class TriageService:
    def __init__(self, protocols):
        self.protocols = protocols

    # --- qSOFA / NEWS2 as per existing MVP ---
    @staticmethod
    def calculate_qsofa(rr, sbp, avpu):
        score = 0
        if rr >= 22: score += 1
        if sbp <= 100: score += 1
        if (avpu or "A").upper() != "A": score += 1
        return score

    @staticmethod
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

    # --- PEWS v1 (config-driven, age banded) ---
    def calculate_pews_v1(self, age, rr, spo2, sbp, hr, avpu):
        proto = self.protocols["pews_v1"]
        band = next(b for b in proto["age_bands"] if b["min"] <= age <= b["max"])
        score = 0

        # RR scoring (rule-of-thumb bands)
        low_rr, high_rr = band["rr"]
        if rr < low_rr-5 or rr > high_rr+10: score += 3
        elif rr < low_rr or rr > high_rr+5: score += 2
        elif rr < low_rr+2 or rr > high_rr: score += 1

        # HR scoring
        low_hr, high_hr = band["hr"]
        if hr < low_hr-10 or hr > high_hr+20: score += 3
        elif hr < low_hr or hr > high_hr+10: score += 2
        elif hr < low_hr+5 or hr > high_hr: score += 1

        # SBP scoring
        if sbp < band["sbp_min"]: score += proto["score_rules"]["sbp"]["low"]

        # SpO2 scoring
        if spo2 < 88: score += proto["score_rules"]["spo2"]["<88"]
        elif spo2 < 92: score += proto["score_rules"]["spo2"]["<92"]

        # AVPU scoring
        score += proto["score_rules"]["avpu"].get(avpu, 0)

        return score, band

    # --- MEOWS v1 + maternal context modifiers ---
    def calculate_meows_v1(self, rr, spo2, sbp, hr, temp_c, avpu, maternal_context="Antenatal"):
        proto = self.protocols["meows_v1"]
        avpu = (avpu or "A").upper()
        triggers = {}

        # HR thresholds
        hr_rule = proto["rules"]["hr"]
        if hr < hr_rule["red_low"] or hr > hr_rule["red_high"]:
            triggers["HR"] = "RED"
        elif hr < hr_rule["amber_low"] or hr > hr_rule["amber_high"]:
            triggers["HR"] = "AMBER"
        else:
            triggers["HR"] = "GREEN"

        # SBP thresholds
        sbp_rule = proto["rules"]["sbp"]
        if sbp < sbp_rule["red_low"] or sbp > sbp_rule["red_high"]:
            triggers["SBP"] = "RED"
        elif sbp < sbp_rule["amber_low"] or sbp > sbp_rule["amber_high"]:
            triggers["SBP"] = "AMBER"
        else:
            triggers["SBP"] = "GREEN"

        # RR thresholds
        rr_rule = proto["rules"]["rr"]
        if rr < rr_rule["red_low"] or rr > rr_rule["red_high"]:
            triggers["RR"] = "RED"
        elif rr < rr_rule["amber_low"] or rr > rr_rule["amber_high"]:
            triggers["RR"] = "AMBER"
        else:
            triggers["RR"] = "GREEN"

        # SpO2
        spo_rule = proto["rules"]["spo2"]
        if spo2 < spo_rule["red"]:
            triggers["SpO2"] = "RED"
        elif spo2 <= spo_rule["amber_high"]:
            triggers["SpO2"] = "AMBER"
        else:
            triggers["SpO2"] = "GREEN"

        # Temp
        t_rule = proto["rules"]["temp"]
        if temp_c < t_rule["red_low"] or temp_c >= t_rule["red_high"]:
            triggers["Temp"] = "RED"
        elif temp_c < t_rule["amber_low"] or temp_c >= t_rule["amber_high"]:
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

        # Context modifiers
        ctx_mod = proto["context_modifiers"].get(maternal_context, {})
        amber_needed = ctx_mod.get("amber_needed", 2)
        force_red_if_any_red = ctx_mod.get("force_red_if_any_red", True)

        if force_red_if_any_red and red_count >= 1:
            triage = "RED"
        elif amber_count >= amber_needed:
            triage = "YELLOW"
        else:
            triage = "GREEN"

        return triage, triggers, amber_count

    def score_based_triage(self, case, on_oxygen=False, spo2_scale2=False, maternal_context="Antenatal"):
        vitals = case.get("vitals", {}) or {}
        age = float(case.get("patient_age", 0) or 0)
        rr = float(vitals.get("rr", 0) or 0)
        spo2 = float(vitals.get("spo2", 0) or 0)
        sbp = float(vitals.get("sbp", 0) or 0)
        hr = float(vitals.get("hr", 0) or 0)
        temp_c = float(vitals.get("temp", 36.5) or 36.5)
        avpu = (vitals.get("avpu", "A") or "A").upper()

        # Maternal pathway
        if case.get("case_type") == "Maternal":
            triage, triggers, amber_count = self.calculate_meows_v1(
                rr, spo2, sbp, hr, temp_c, avpu, maternal_context=maternal_context
            )

            # Secondary comparator only
            news2_total, news2_parts = self.calculate_news2(
                rr, spo2, sbp, hr, temp_c, avpu,
                on_oxygen=on_oxygen, spo2_scale2=spo2_scale2
            )
            qsofa = self.calculate_qsofa(rr, sbp, avpu)

            return triage, {
                "system": "MEOWS v1",
                "maternal_context": maternal_context,
                "meows_triggers": triggers,
                "meows_total": amber_count,
                "secondary_system": "NEWS2/qSOFA (comparison only)",
                "secondary_news2_total": news2_total,
                "secondary_news2_parts": news2_parts,
                "secondary_qsofa": qsofa
            }

        # Pediatric pathway (PEWS v1)
        if age < 16:
            pews_score, band = self.calculate_pews_v1(age, rr, spo2, sbp, hr, avpu)
            cut = self.protocols["pews_v1"]["triage_cutoffs"]
            if pews_score >= cut["RED"]:
                triage = "RED"
            elif pews_score >= cut["YELLOW"]:
                triage = "YELLOW"
            else:
                triage = "GREEN"
            return triage, {
                "system": "PEWS v1",
                "pews_total": pews_score,
                "age_band": band
            }

        # Adult pathway (NEWS2 + qSOFA)
        news2_total, news2_parts = self.calculate_news2(
            rr, spo2, sbp, hr, temp_c, avpu,
            on_oxygen=on_oxygen, spo2_scale2=spo2_scale2
        )
        qsofa = self.calculate_qsofa(rr, sbp, avpu)

        if news2_total >= 7:
            triage = "RED"
        elif news2_total >= 5:
            triage = "YELLOW"
        else:
            triage = "GREEN"

        if qsofa >= 2:
            triage = "YELLOW" if triage == "GREEN" else "RED"

        return triage, {
            "system": "NEWS2 + qSOFA",
            "news2_total": news2_total,
            "news2_parts": news2_parts,
            "qsofa": qsofa
        }


class FacilityMatchService:
    def __init__(self, registry):
        self.registry = registry

    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
        return 2*r*math.asin(math.sqrt(a))

    @staticmethod
    def estimate_eta(distance_km, terrain="hilly", tod="day"):
        speed = SPEED_PROFILES.get(f"{terrain}_{tod}", 28)
        return int((distance_km / max(speed, 5)) * 60 + 8)

    def _required_caps_from_reasons(self, reason_detail_map, other_notes=""):
        req = []
        for cat, details in reason_detail_map.items():
            for d in details:
                req.extend(REASON_TO_CAPS.get(d, []))
        # free-text notes can be ignored in demo
        return list(set(req))

    def filter_facilities(self, required_caps):
        hard = [c for c in required_caps if c in HARD_CAPS]
        def ok(f):
            caps = f["capabilities"]
            # must have ICU bed free always
            if f["beds"]["icu_free"] <= 0:
                return False
            return all(caps.get(h, 0) >= 1 for h in hard)
        return ok

    def score_facility(self, case_type, triage, required_caps, facility):
        caps = facility["capabilities"]
        beds = facility["beds"]

        dist_km = self.haversine_km(
            DASHBOARD_LAT, DASHBOARD_LON,
            facility["lat"], facility["lon"]
        )

        # Capability fit
        cap_score = 0
        for c in required_caps:
            cap_score += 10 if caps.get(c, 0) > 0 else 0

        # Distance fit
        distance_score = max(0, 40 - dist_km)

        # Load penalty
        occ = facility["occupancy_pct"]
        load_penalty = occ * 0.2

        # Triage fit bonus
        ftype = facility["type"]
        triage_bonus = 0
        if triage == "RED" and ftype in ["Tertiary", "Trauma", "Specialty"]:
            triage_bonus = 10

        # Capacity market willingness bonus (moat idea #1)
        market_bonus = 15 if facility.get("accepting_red_now") and triage == "RED" else 0

        total = cap_score + distance_score + triage_bonus + market_bonus - load_penalty

        terrain = facility.get("terrain", "hilly")
        tod = "night" if datetime.now().hour < 6 or datetime.now().hour >= 19 else "day"
        eta_min = self.estimate_eta(dist_km, terrain=terrain, tod=tod)

        breakdown = dict(
            distance_km=round(dist_km, 1),
            cap_score=cap_score,
            distance_score=round(distance_score, 1),
            load_penalty=round(load_penalty, 1),
            triage_bonus=triage_bonus,
            market_bonus=market_bonus,
            icu_free=beds["icu_free"],
            occupancy_pct=occ,
            facility_type=ftype,
            eta_min=eta_min
        )
        return round(total, 1), breakdown

    def match(self, case_type, triage, reason_detail_map, other_reason_notes=""):
        required_caps = self._required_caps_from_reasons(reason_detail_map, other_reason_notes)
        ok_fn = self.filter_facilities(required_caps)

        scored = []
        for f in self.registry:
            if f["name"] == DASHBOARD_HOSPITAL:
                continue
            if not ok_fn(f):
                continue
            score, breakdown = self.score_facility(case_type, triage, required_caps, f)
            scored.append({
                "Facility": f["name"],
                "Score": score,
                "Distance (km)": breakdown["distance_km"],
                "ETA (min)": breakdown["eta_min"],
                "ICU Beds Free": breakdown["icu_free"],
                "Occupancy %": breakdown["occupancy_pct"],
                "Type": breakdown["facility_type"],
                "Market Bonus": breakdown["market_bonus"],
                "Capability Fit Score": breakdown["cap_score"],
                "Why matched": f"Cap:{breakdown['cap_score']} + Dist:{breakdown['distance_score']} + Market:{breakdown['market_bonus']}"
            })

        df = pd.DataFrame(scored).sort_values("Score", ascending=False) if scored else pd.DataFrame(
            columns=["Facility","Score","Distance (km)","ETA (min)","ICU Beds Free","Occupancy %","Type"]
        )
        return df, required_caps


class ReferralService:
    def __init__(self, audit: AuditService, data: DataService):
        self.audit = audit
        self.data = data

    @staticmethod
    def update_sla(case):
        if case.get("status") != "REQUESTED":
            return case

        req_at = parse_dt(case.get("requested_at"))
        mins = (datetime.now() - req_at).total_seconds() / 60

        sla = case.get("sla_minutes", 15)
        if mins > sla:
            case["escalation_level"] = case.get("escalation_level", 0) + 1
            case["sla_minutes"] = sla + 10
            case["sla_escalated_at"] = datetime.now()
        return case

    def transition(self, case, new_status, user, role, note=""):
        old = case.get("status")
        case["status"] = new_status
        case["status_updated_at"] = datetime.now()
        self.audit.log(user, role, case["case_id"], "STATUS_TRANSITION", {
            "from": old, "to": new_status, "note": note
        })
        self.data.save_referral(case)
        return case


# =============================================================================
# MODEL LOADING (optional)
# =============================================================================
@st.cache_resource
def load_triage_model():
    if joblib is None:
        return None
    model_path = APP_DIR / "my_model.pkl"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


# =============================================================================
# BOOTSTRAP DEFAULTS
# =============================================================================
DEFAULT_PROTOCOLS = {
    "pews_v1": {
        "age_bands": [
            {"min": 0, "max": 1, "rr": [30, 50], "hr": [110, 160], "sbp_min": 70},
            {"min": 1, "max": 4, "rr": [20, 40], "hr": [95, 140], "sbp_min": 80},
            {"min": 5, "max": 12, "rr": [18, 30], "hr": [80, 120], "sbp_min": 90},
            {"min": 13, "max": 15, "rr": [12, 20], "hr": [60, 100], "sbp_min": 95},
        ],
        "score_rules": {
            "sbp": {"low": 2},
            "spo2": {"<92": 2, "<88": 3},
            "avpu": {"V": 1, "P": 2, "U": 3}
        },
        "triage_cutoffs": {"RED": 4, "YELLOW": 2}
    },
    "meows_v1": {
        "rules": {
            "hr": {"red_low": 40, "red_high": 120, "amber_low": 50, "amber_high": 110},
            "sbp": {"red_low": 90, "red_high": 160, "amber_low": 100, "amber_high": 150},
            "rr": {"red_low": 10, "red_high": 30, "amber_low": 12, "amber_high": 24},
            "spo2": {"red": 92, "amber_high": 94},
            "temp": {"red_low": 35.0, "red_high": 38.5, "amber_low": 36.0, "amber_high": 38.0},
        },
        "context_modifiers": {
            "Antenatal": {"amber_needed": 2, "force_red_if_any_red": True},
            "Intrapartum": {"amber_needed": 2, "force_red_if_any_red": True},
            "Postpartum <24h": {"amber_needed": 1, "force_red_if_any_red": True},
            "Postpartum >24h": {"amber_needed": 2, "force_red_if_any_red": True},
            "Hemorrhage risk": {"amber_needed": 1, "force_red_if_any_red": True},
            "Eclampsia risk": {"amber_needed": 1, "force_red_if_any_red": True},
        }
    }
}

DEFAULT_FACILITY_REGISTRY = [
    {
        "facility_id": "TCH01",
        "name": "Tertiary Central Hospital",
        "type": "Tertiary",
        "lat": 25.578, "lon": 91.893,
        "terrain": "hilly",
        "capabilities": {
            "ICU_BED": 30, "VENTILATOR": 18, "CT": 1, "MRI": 1, "CATHLAB": 1,
            "BLOODBANK": 1, "OR": 4, "NEURO_SPECIALIST": 1, "CARDIO_SPECIALIST": 1, "OBGYN_SPECIALIST": 1
        },
        "beds": {"icu_total": 30, "icu_free": 6, "hdu_free": 8},
        "on_call": {"neuro": True, "cardiac": True, "obgyn": True},
        "occupancy_pct": 82,
        "accepting_red_now": True,
        "last_updated": datetime.now(timezone.utc).isoformat()
    },
    {
        "facility_id": "DNG01",
        "name": "District North General",
        "type": "District",
        "lat": 25.591, "lon": 91.878,
        "terrain": "hilly",
        "capabilities": {
            "ICU_BED": 10, "VENTILATOR": 4, "CT": 1, "MRI": 0, "CATHLAB": 0,
            "BLOODBANK": 1, "OR": 2, "NEURO_SPECIALIST": 0, "CARDIO_SPECIALIST": 0, "OBGYN_SPECIALIST": 1
        },
        "beds": {"icu_total": 10, "icu_free": 2, "hdu_free": 4},
        "on_call": {"neuro": False, "cardiac": False, "obgyn": True},
        "occupancy_pct": 88,
        "accepting_red_now": False,
        "last_updated": datetime.now(timezone.utc).isoformat()
    },
    {
        "facility_id": "SSM01",
        "name": "Specialty South Medical",
        "type": "Specialty",
        "lat": 25.565, "lon": 91.901,
        "terrain": "hilly",
        "capabilities": {
            "ICU_BED": 12, "VENTILATOR": 6, "CT": 1, "MRI": 0, "CATHLAB": 1,
            "BLOODBANK": 1, "OR": 2, "NEURO_SPECIALIST": 1, "CARDIO_SPECIALIST": 1, "OBGYN_SPECIALIST": 0
        },
        "beds": {"icu_total": 12, "icu_free": 3, "hdu_free": 3},
        "on_call": {"neuro": True, "cardiac": True, "obgyn": False},
        "occupancy_pct": 76,
        "accepting_red_now": True,
        "last_updated": datetime.now(timezone.utc).isoformat()
    },
    {
        "facility_id": "TEC01",
        "name": "Trauma East Center",
        "type": "Trauma",
        "lat": 25.572, "lon": 91.885,
        "terrain": "hilly",
        "capabilities": {
            "ICU_BED": 16, "VENTILATOR": 8, "CT": 1, "MRI": 0, "CATHLAB": 0,
            "BLOODBANK": 1, "OR": 3, "NEURO_SPECIALIST": 1, "CARDIO_SPECIALIST": 0, "OBGYN_SPECIALIST": 0
        },
        "beds": {"icu_total": 16, "icu_free": 4, "hdu_free": 5},
        "on_call": {"neuro": True, "cardiac": False, "obgyn": False},
        "occupancy_pct": 70,
        "accepting_red_now": True,
        "last_updated": datetime.now(timezone.utc).isoformat()
    },
    {
        "facility_id": "CHC01",
        "name": "CHC Smit",
        "type": "District",
        "lat": 25.613, "lon": 91.915,
        "terrain": "hilly",
        "capabilities": {
            "ICU_BED": 4, "VENTILATOR": 1, "CT": 0, "MRI": 0, "CATHLAB": 0,
            "BLOODBANK": 0, "OR": 1, "NEURO_SPECIALIST": 0, "CARDIO_SPECIALIST": 0, "OBGYN_SPECIALIST": 1
        },
        "beds": {"icu_total": 4, "icu_free": 1, "hdu_free": 1},
        "on_call": {"neuro": False, "cardiac": False, "obgyn": True},
        "occupancy_pct": 92,
        "accepting_red_now": False,
        "last_updated": datetime.now(timezone.utc).isoformat()
    },
]


def init_storage_and_seed():
    data = DataService(DB_PATH)
    # Protocol seed
    if not os.path.exists(PROTOCOLS_PATH):
        with open(PROTOCOLS_PATH, "w") as f:
            json.dump(DEFAULT_PROTOCOLS, f, indent=2)

    # Facility registry seed
    facilities = data.load_facilities()
    if not facilities:
        data.upsert_facilities(DEFAULT_FACILITY_REGISTRY)

    # also create a JSON registry for easy external edits
    if not os.path.exists(FACILITY_REGISTRY_PATH):
        with open(FACILITY_REGISTRY_PATH, "w") as f:
            json.dump(DEFAULT_FACILITY_REGISTRY, f, indent=2, default=str)

    return data


def load_protocols():
    try:
        with open(PROTOCOLS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PROTOCOLS


def load_registry(data_service):
    facilities = data_service.load_facilities()
    if facilities:
        return facilities
    return DEFAULT_FACILITY_REGISTRY


def refresh_live_availability(registry):
    """Controlled mock live updates to demonstrate dynamic capacity market."""
    now = time.time()
    last = st.session_state.get("last_live_refresh", 0)
    if now - last < 30:
        return registry  # throttle
    st.session_state["last_live_refresh"] = now

    updated = []
    for f in registry:
        g = dict(f)
        beds = dict(g["beds"])
        caps = dict(g["capabilities"])

        # Randomly adjust ICU free within bounds
        icu_total = beds["icu_total"]
        beds["icu_free"] = int(min(icu_total, max(0, beds["icu_free"] + random.randint(-2, 2))))
        beds["hdu_free"] = int(min(icu_total, max(0, beds.get("hdu_free", 0) + random.randint(-1, 1))))

        # Occupancy drift
        g["occupancy_pct"] = int(min(98, max(60, g["occupancy_pct"] + random.randint(-3, 3))))

        # Willingness to accept REDs depends on free ICU + occupancy
        g["accepting_red_now"] = beds["icu_free"] >= 2 and g["occupancy_pct"] <= 88

        g["beds"] = beds
        g["capabilities"] = caps
        g["last_updated"] = datetime.now(timezone.utc).isoformat()
        updated.append(g)
    return updated


def deterioration_risk(transit_updates):
    """Simple trend-based digital twin risk."""
    if not transit_updates or len(transit_updates) < 2:
        return "LOW", {}
    try:
        v0 = transit_updates[0]["vitals"]
        v1 = transit_updates[-1]["vitals"]
        spo2_drop = v1["spo2"] - v0["spo2"]
        sbp_drop = v1["sbp"] - v0["sbp"]
        hr_rise = v1["hr"] - v0["hr"]
        if spo2_drop < -4 or sbp_drop < -15 or hr_rise > 15:
            return "HIGH", {"spo2_drop": spo2_drop, "sbp_drop": sbp_drop, "hr_rise": hr_rise}
        if spo2_drop < -2 or sbp_drop < -8 or hr_rise > 8:
            return "MODERATE", {"spo2_drop": spo2_drop, "sbp_drop": sbp_drop, "hr_rise": hr_rise}
        return "LOW", {"spo2_drop": spo2_drop, "sbp_drop": sbp_drop, "hr_rise": hr_rise}
    except Exception:
        return "LOW", {}


def make_transit_updates(case_time, vitals, transport_time):
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
            "spo2": min(100, max(80, base_spo2 + random.randint(-3, 3))),
            "temp": round(base_temp + random.uniform(-0.3, 0.3), 1),
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


def generate_synthetic_cases(registry, days_back=30):
    """Synthetic inbound/outbound cases using registry (no random capability in matching)."""
    referred_cases, received_cases = [], []
    base_time = datetime.now() - timedelta(days=days_back)

    for day in range(days_back):
        daily_cases = random.randint(3, 7)
        for case_num in range(daily_cases):
            case_time = base_time + timedelta(
                days=day,
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            case_type = random.choice(CASE_TYPES)
            age = random.randint(18, 80) if case_type != "Maternal" else random.randint(18, 40)
            if random.random() < 0.12:
                age = random.randint(0, 15)  # pediatric mix

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

            matching_icd = [
                icd for icd in ICD_CATALOG
                if icd["case_type"] == case_type and icd["age_min"] <= age <= icd["age_max"]
            ]
            icd = random.choice(matching_icd) if matching_icd else random.choice(
                [i for i in ICD_CATALOG if i["case_type"] == case_type]
            )

            # Referral reasons
            reason_category = random.choice(["ICU_BED_UNAVAILABLE","SPECIALTY_REQUIRED","EQUIPMENT_REQUIRED"])
            if reason_category == "ICU_BED_UNAVAILABLE":
                reason_detail_map = {"ICU_BED_UNAVAILABLE":[random.choice(REFERRAL_REASONS["ICU_BED_UNAVAILABLE"])]}
            elif reason_category == "SPECIALTY_REQUIRED":
                reason_detail_map = {"SPECIALTY_REQUIRED":[random.choice(REFERRAL_REASONS["SPECIALTY_REQUIRED"][case_type])]}
            else:
                reason_detail_map = {"EQUIPMENT_REQUIRED":[random.choice(REFERRAL_REASONS["EQUIPMENT_REQUIRED"][case_type])]}

            base_intv = ICD_INTERVENTION_MAP.get(icd["icd_code"], INTERVENTION_PROTOCOLS[case_type])
            interventions = random.sample(base_intv, min(len(base_intv), random.randint(2, 4)))

            emt_crew = random.choice(EMT_CREW)
            transport_time = random.randint(20, 90)

            transit_updates = make_transit_updates(case_time, vitals, transport_time)

            is_active_transfer = random.random() < 0.18
            eta_minutes = transport_time if is_active_transfer else 0
            if not is_active_transfer and random.random() > 0.7:
                transit_updates = []

            # OUTBOUND
            referring_facility_out = DASHBOARD_HOSPITAL
            receiving_facility_out = random.choice([f["name"] for f in registry if f["name"] != DASHBOARD_HOSPITAL])

            # INBOUND
            referring_facility_in = random.choice(["PHC Mawlai", "CHC Smit", "CHC Pynursla", "Rural Health Center"])
            receiving_facility_in = DASHBOARD_HOSPITAL

            case_id_ref = f"REF_{case_time.strftime('%Y%m%d')}_{case_num:03d}"
            case_id_rec = f"REC_{case_time.strftime('%Y%m%d')}_{case_num:03d}"

            base_case = {
                "timestamp": case_time,
                "patient_name": random.choice(["A. Marbaniang", "S. Khonglam", "R. Syiem", "M. Lyngdoh", "Unknown"]),
                "patient_age": age,
                "patient_sex": "F" if case_type == "Maternal" else random.choice(["M","F"]),
                "case_type": case_type,
                "icd_code": icd["icd_code"],
                "icd_label": icd["label"],
                "vitals": vitals,
                "interventions": interventions,
                "reason_detail_map": reason_detail_map,
                "clinical_notes": "Synthetic demo case",
                "emt_crew": emt_crew,
                "vehicle_id": emt_crew["vehicle"],
                "transport_time_minutes": transport_time,
                "eta_minutes": eta_minutes,
                "transit_updates": transit_updates,
                "requested_at": case_time,
                "sla_minutes": 15,
                "escalation_level": 0,
                "status": "ENROUTE" if is_active_transfer else "CLOSED",
            }

            referred_case = dict(base_case)
            referred_case.update({
                "case_id": case_id_ref,
                "referring_facility": referring_facility_out,
                "receiving_facility": receiving_facility_out
            })

            received_case = dict(base_case)
            received_case.update({
                "case_id": case_id_rec,
                "referring_facility": referring_facility_in,
                "receiving_facility": receiving_facility_in
            })

            referred_cases.append(referred_case)
            received_cases.append(received_case)

    return referred_cases, received_cases


def to_fhir_observation(case):
    v = case.get("vitals", {})
    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {"text": "Vitals set"},
        "subject": {"display": case.get("patient_name","Unknown")},
        "component": [
            {"code":{"text":"RR"},"valueQuantity":{"value":v.get("rr"),"unit":"breaths/min"}},
            {"code":{"text":"SpO2"},"valueQuantity":{"value":v.get("spo2"),"unit":"%" }},
            {"code":{"text":"SBP"},"valueQuantity":{"value":v.get("sbp"),"unit":"mmHg"}},
            {"code":{"text":"HR"},"valueQuantity":{"value":v.get("hr"),"unit":"bpm"}},
            {"code":{"text":"Temp"},"valueQuantity":{"value":v.get("temp"),"unit":"C"}},
            {"code":{"text":"AVPU"},"valueString":v.get("avpu")}
        ]
    }


def safe_badge(color):
    if color == "RED":
        return "<span class='badge-red'>RED</span>"
    if color == "YELLOW":
        return "<span class='badge-yellow'>YELLOW</span>"
    return "<span class='badge-green'>GREEN</span>"


# =============================================================================
# LOGIN & RBAC
# =============================================================================
USERS = {
    "cc_admin": {"pwd": "demo", "role": "COMMAND_CENTER"},
    "spoke_doc": {"pwd": "demo", "role": "REFERRER"},
    "emt_01": {"pwd": "demo", "role": "EMT"},
    "district_admin": {"pwd": "demo", "role": "DISTRICT_ADMIN"},
}

def login_sidebar():
    st.sidebar.markdown("### 🔐 Login (Demo RBAC)")
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if u in USERS and USERS[u]["pwd"] == p:
            st.session_state.user = u
            st.session_state.role = USERS[u]["role"]
            st.sidebar.success(f"Logged in as {u} ({st.session_state.role})")
        else:
            st.sidebar.error("Invalid credentials")

if "user" not in st.session_state or "role" not in st.session_state:
    login_sidebar()
    st.stop()


# =============================================================================
# INIT APP DATA
# =============================================================================
data_service = init_storage_and_seed()
audit_service = AuditService(DB_PATH)

protocols = load_protocols()
registry = load_registry(data_service)

# Sidebar toggles
st.sidebar.markdown("---")
offline_mode = st.sidebar.toggle("Offline mode (store & forward demo)", False)
live_availability = st.sidebar.toggle("Mock live availability refresh", True)
show_debug = st.sidebar.toggle("Show debug panels", False)

if live_availability:
    registry = refresh_live_availability(registry)
    data_service.upsert_facilities(registry)

triage_service = TriageService(protocols)
match_service = FacilityMatchService(registry)
referral_service = ReferralService(audit_service, data_service)

# Load persisted referrals/outcomes, then merge into session state
persisted_referrals = data_service.load_referrals()
outcomes_map = data_service.load_outcomes()

if "premium_data" not in st.session_state:
    synthetic_out, synthetic_in = generate_synthetic_cases(registry, days_back=30)

    # merge persisted referrals into outbound list
    outbound = synthetic_out + persisted_referrals

    st.session_state.premium_data = {
        "outbound": outbound,
        "inbound": synthetic_in,
    }

# Apply outcomes to cases
for bucket in ["outbound", "inbound"]:
    for c in st.session_state.premium_data[bucket]:
        if c.get("case_id") in outcomes_map:
            c["outcome_adjudication"] = outcomes_map[c["case_id"]]

# SLA update pass
for bucket in ["outbound", "inbound"]:
    st.session_state.premium_data[bucket] = [
        referral_service.update_sla(c) for c in st.session_state.premium_data[bucket]
    ]

# Sync offline queue if online
if not offline_mode:
    unsynced = data_service.load_offline_unsynced()
    for q in unsynced:
        case = q["case"]
        case["status"] = "REQUESTED"
        data_service.save_referral(case)
        data_service.mark_offline_synced(q["id"])
        audit_service.log(st.session_state.user, st.session_state.role, case["case_id"], "OFFLINE_SYNCED", {"queued_id": q["id"]})
        st.session_state.premium_data["outbound"].append(case)


# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
  <h1 style="margin-bottom:0.2rem;">AHECN Hospital Command Center</h1>
  <div style="opacity:0.9;font-size:1.05rem;">
    Advanced Emergency Care Coordination Platform — Updated Demo MVP
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================
tab_cc, tab_new, tab_market, tab_outcomes, tab_audit = st.tabs([
    "🏥 Command Center",
    "🚀 New Outbound Referral",
    "📈 Capacity Market",
    "✅ Outcome Adjudication",
    "🧾 Audit & ML Flywheel"
])


# =============================================================================
# TAB 1: COMMAND CENTER
# =============================================================================
with tab_cc:
    left, right = st.columns([0.38, 0.62])

    outbound = st.session_state.premium_data["outbound"]
    inbound = st.session_state.premium_data["inbound"]

    # Filters
    with left:
        st.markdown("### Active Network Cases")
        view_filter = st.selectbox("View", ["Outbound (Sent)", "Inbound (Received)"])
        status_filter = st.multiselect(
            "Status filter",
            options=REFERRAL_STATE_FLOW,
            default=["REQUESTED","ACKNOWLEDGED","ACCEPTED","DISPATCHED","ENROUTE"]
        )

        cases = outbound if view_filter.startswith("Outbound") else inbound
        cases = [c for c in cases if c.get("status") in status_filter]

        if not cases:
            st.info("No cases match filter.")
        else:
            for i, c in enumerate(sorted(cases, key=lambda x: x["timestamp"], reverse=True)[:30]):
                triage_preview, _ = triage_service.score_based_triage(c)
                cls = "critical" if triage_preview == "RED" else "urgent" if triage_preview == "YELLOW" else "stable"
                if st.button(
                    f"{c['case_id']} • {c.get('case_type')} • {triage_preview} • {c.get('status')}",
                    key=f"case_btn_{view_filter}_{i}"
                ):
                    st.session_state.selected_case = c["case_id"]
                    st.session_state.selected_bucket = "outbound" if view_filter.startswith("Outbound") else "inbound"

    with right:
        sel_id = st.session_state.get("selected_case")
        sel_bucket = st.session_state.get("selected_bucket", "outbound")

        sel_case = None
        for c in st.session_state.premium_data[sel_bucket]:
            if c.get("case_id") == sel_id:
                sel_case = c
                break

        if not sel_case:
            st.info("Select a case from the left panel.")
        else:
            c = sel_case
            st.markdown(f"## Case Detail — {c['case_id']}")
            triage_color = c.get("triage_color")

            topA, topB, topC = st.columns(3)
            with topA:
                st.metric("Patient", c.get("patient_name","Unknown"))
                st.write(f"Age/Sex: {c.get('patient_age')} / {c.get('patient_sex')}")
            with topB:
                st.metric("Case Type", c.get("case_type"))
                st.write(f"ICD: {c.get('icd_code')} — {c.get('icd_label')}")
            with topC:
                st.markdown(f"Triage (current): {safe_badge(triage_color) if triage_color else '-'}", unsafe_allow_html=True)
                st.write(f"Status: **{c.get('status')}**")

            st.markdown("---")

            # Timeline
            st.markdown("### Timeline")
            st.write(f"Requested at: {c.get('requested_at')}")
            if c.get("sla_escalated_at"):
                st.warning(f"SLA Escalated (L{c.get('escalation_level')}): {c.get('sla_escalated_at')}")

            # Vitals
            st.markdown("### Vitals")
            v = c.get("vitals", {})
            vit_df = pd.DataFrame([v])
            st.table(vit_df)

            # Transport + Digital Twin
            st.markdown("### Transport & Pre-arrival Digital Twin")
            updates = c.get("transit_updates", []) or []
            risk, deltas = deterioration_risk(updates)
            risk_color = {"LOW":"#2e7d32","MODERATE":"#f57f17","HIGH":"#c62828"}[risk]
            st.markdown(
                f"<div class='premium-card' style='border-left:6px solid {risk_color};'>"
                f"<b>Deterioration Risk Forecast:</b> <span style='color:{risk_color};font-weight:700;'>{risk}</span><br/>"
                f"Trend deltas: {deltas if deltas else 'n/a'}"
                f"</div>",
                unsafe_allow_html=True
            )
            if updates:
                upd_rows = []
                for u in updates:
                    uv = u["vitals"]
                    upd_rows.append({
                        "Time": u["timestamp"],
                        "HR": uv["hr"], "SBP": uv["sbp"], "RR": uv["rr"], "SpO2": uv["spo2"], "Temp": uv["temp"],
                        "Note": u["emt_note"]
                    })
                upd_df = pd.DataFrame(upd_rows)
                st.dataframe(upd_df, use_container_width=True)
                # Trend chart
                try:
                    chart_df = upd_df[["Time","SpO2","SBP","HR"]].copy()
                    chart_df["Time"] = pd.to_datetime(chart_df["Time"])
                    fig = px.line(chart_df, x="Time", y=["SpO2","SBP","HR"], markers=True, title="Transit Vitals Trend")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
            else:
                st.caption("No transit vitals available for this case.")

            st.markdown("---")

            # Clinical triage recommendation
            st.markdown("### 🧠 Clinical Triage Recommendation")
            oxy_col, s2_col, ctx_col = st.columns([0.25,0.25,0.5])
            with oxy_col:
                on_oxygen = st.checkbox("On oxygen?", value=False, key=f"oxy_{c['case_id']}")
            with s2_col:
                spo2_scale2 = st.checkbox("SpO₂ Scale 2?", value=False, key=f"s2_{c['case_id']}")
            maternal_context = "Antenatal"
            if c.get("case_type") == "Maternal":
                with ctx_col:
                    maternal_context = st.selectbox(
                        "Maternal context",
                        ["Antenatal","Intrapartum","Postpartum <24h","Postpartum >24h","Hemorrhage risk","Eclampsia risk"],
                        key=f"ctx_{c['case_id']}"
                    )

            triage_color, details = triage_service.score_based_triage(
                c, on_oxygen=on_oxygen, spo2_scale2=spo2_scale2, maternal_context=maternal_context
            )
            c["triage_color"] = triage_color  # update in-memory

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
                    f"**NEWS2 Total:** {details['news2_total']}  "
                    f"(RR:{details['news2_parts']['rr']}, SpO₂:{details['news2_parts']['spo2']}, "
                    f"O₂:{details['news2_parts']['oxygen']}, SBP:{details['news2_parts']['sbp']}, "
                    f"HR:{details['news2_parts']['hr']}, Temp:{details['news2_parts']['temp']}, "
                    f"AVPU:{details['news2_parts']['conc']})"
                )
                st.write(f"**qSOFA:** {details['qsofa']} (≥2 upgrades risk)")
            elif system.startswith("MEOWS"):
                st.markdown("**MEOWS v1 Triggers**")
                st.write(details.get("meows_triggers", {}))
                st.caption(f"Context: {details.get('maternal_context')}")
            elif system.startswith("PEWS"):
                st.markdown("**PEWS v1**")
                st.write(f"Score: {details.get('pews_total')} | Band: {details.get('age_band')}")

            # ML comparator (recommender only)
            st.markdown("#### ML Comparator (Recommender-only)")
            model = load_triage_model()
            age = float(c.get("patient_age",0) or 0)
            features_payload = {
                "age": age,
                "sex": c.get("patient_sex"),
                "vitals": c.get("vitals"),
                "on_oxygen": on_oxygen,
                "spo2_scale2": spo2_scale2,
                "meows_total": details.get("meows_total"),
                "news2_total": details.get("news2_total"),
                "qsofa": details.get("qsofa"),
                "maternal_context": maternal_context if c.get("case_type")=="Maternal" else None,
                "reasons": c.get("reason_detail_map", {})
            }
            data_service.log_ml_features(c["case_id"], features_payload)

            if model is None:
                st.info("No ML model file found (my_model.pkl). Showing rules-only triage.")
            else:
                # simple feature vector (demo)
                v = c.get("vitals", {})
                vec = np.array([[
                    age, v.get("sbp",0), v.get("spo2",0), v.get("hr",0),
                    v.get("rr",0), v.get("temp",36.5), 1 if (v.get("avpu","A")!="A") else 0,
                    1 if on_oxygen else 0, 1 if spo2_scale2 else 0
                ]])
                try:
                    ml_pred = model.predict(vec)[0]
                    st.write(f"ML suggested triage: **{ml_pred}** (does NOT override rules)")
                    agree = "✅ Agree" if ml_pred == triage_color else "⚠️ Disagree"
                    st.write(f"Rules vs ML: {agree}")

                    fb_col1, fb_col2, fb_col3 = st.columns(3)
                    with fb_col1:
                        if st.button("Clinician agrees"):
                            log_payload = dict(features_payload)
                            log_payload["rules_triage"] = triage_color
                            log_payload["ml_triage"] = ml_pred
                            log_payload["feedback"] = "agree"
                            data_service.log_ml_features(c["case_id"], log_payload)
                            audit_service.log(st.session_state.user, st.session_state.role, c["case_id"], "ML_FEEDBACK", log_payload)
                            st.success("Feedback logged.")
                    with fb_col2:
                        if st.button("Clinician disagrees"):
                            log_payload = dict(features_payload)
                            log_payload["rules_triage"] = triage_color
                            log_payload["ml_triage"] = ml_pred
                            log_payload["feedback"] = "disagree"
                            data_service.log_ml_features(c["case_id"], log_payload)
                            audit_service.log(st.session_state.user, st.session_state.role, c["case_id"], "ML_FEEDBACK", log_payload)
                            st.success("Feedback logged.")
                    with fb_col3:
                        if st.button("Needs review"):
                            log_payload = dict(features_payload)
                            log_payload["rules_triage"] = triage_color
                            log_payload["ml_triage"] = ml_pred
                            log_payload["feedback"] = "review"
                            data_service.log_ml_features(c["case_id"], log_payload)
                            audit_service.log(st.session_state.user, st.session_state.role, c["case_id"], "ML_FEEDBACK", log_payload)
                            st.success("Feedback logged.")
                except Exception:
                    st.warning("ML prediction failed; check model interface.")

            # Optional override with rationale
            st.markdown("---")
            st.markdown("### Override (with audit rationale)")
            if st.session_state.role in ["COMMAND_CENTER","REFERRER"]:
                override_col1, override_col2 = st.columns([0.4,0.6])
                with override_col1:
                    override_to = st.selectbox("Override to", TRIAGE_COLORS, index=TRIAGE_COLORS.index(triage_color))
                with override_col2:
                    override_reason = st.selectbox(
                        "Override reason", 
                        ["Clinical judgement","Protocol exception","Equipment limitation","Contextual risk","Other"]
                    )
                override_note = st.text_area("Override note (mandatory)")
                if st.button("Apply override"):
                    if not override_note.strip():
                        st.error("Override note is mandatory.")
                    else:
                        prev = triage_color
                        c["triage_color"] = override_to
                        audit_service.log(st.session_state.user, st.session_state.role, c["case_id"], "TRIAGE_OVERRIDE", {
                            "from": prev, "to": override_to,
                            "reason": override_reason, "note": override_note,
                            "rules_details": details
                        })
                        data_service.save_referral(c)
                        st.success("Override applied and audited.")
            else:
                st.caption("You do not have permission to override triage.")

            # Status transitions
            st.markdown("---")
            st.markdown("### Referral Workflow")
            st.write(f"Current status: **{c.get('status')}**")
            next_actions = []
            status = c.get("status")

            if status == "REQUESTED" and st.session_state.role in ["COMMAND_CENTER","REFERRER"]:
                next_actions = [("ACKNOWLEDGE","ACKNOWLEDGED"), ("REJECT","REJECTED")]
            elif status == "ACKNOWLEDGED" and st.session_state.role in ["COMMAND_CENTER"]:
                next_actions = [("ACCEPT","ACCEPTED"), ("REJECT","REJECTED")]
            elif status == "ACCEPTED" and st.session_state.role in ["COMMAND_CENTER","EMT"]:
                next_actions = [("DISPATCH","DISPATCHED")]
            elif status == "DISPATCHED" and st.session_state.role in ["EMT","COMMAND_CENTER"]:
                next_actions = [("ENROUTE","ENROUTE")]
            elif status == "ENROUTE" and st.session_state.role in ["EMT","COMMAND_CENTER"]:
                next_actions = [("ARRIVED","ARRIVED")]
            elif status == "ARRIVED" and st.session_state.role in ["COMMAND_CENTER"]:
                next_actions = [("CLOSE","CLOSED")]

            if next_actions:
                act_cols = st.columns(len(next_actions))
                for i, (label, new_status) in enumerate(next_actions):
                    with act_cols[i]:
                        if st.button(label):
                            c = referral_service.transition(
                                c, new_status, st.session_state.user, st.session_state.role
                            )
                            st.success(f"Status moved to {new_status}.")

            # Export
            st.markdown("---")
            st.markdown("### Interoperability Export (FHIR-lite)")
            if st.button("Generate FHIR Observation JSON"):
                st.json(to_fhir_observation(c))

            if show_debug:
                st.markdown("### Debug case object")
                st.json(c)


# =============================================================================
# TAB 2: NEW OUTBOUND REFERRAL
# =============================================================================
with tab_new:
    st.markdown("## 🚀 Create New Outbound Referral")

    if st.session_state.role not in ["REFERRER","COMMAND_CENTER"]:
        st.warning("You don't have permission to create referrals.")
    else:
        with st.form("new_ref_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_name = st.text_input("Patient Name")
                patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
                patient_sex = st.selectbox("Sex", ["M","F"])
            with c2:
                case_type = st.selectbox("Case Type", CASE_TYPES)
                icd_options = [
                    f"{i['icd_code']} — {i['label']}"
                    for i in ICD_CATALOG if i["case_type"] == case_type
                ]
                icd_choice = st.selectbox("Provisional ICD", icd_options)
            with c3:
                maternal_context = "Antenatal"
                if case_type == "Maternal":
                    maternal_context = st.selectbox(
                        "Maternal context",
                        ["Antenatal","Intrapartum","Postpartum <24h","Postpartum >24h","Hemorrhage risk","Eclampsia risk"]
                    )

            st.markdown("### Vitals")
            v1, v2, v3, v4, v5, v6 = st.columns(6)
            with v1: hr = st.number_input("HR", min_value=30, max_value=220, value=100)
            with v2: sbp = st.number_input("SBP", min_value=50, max_value=250, value=120)
            with v3: rr = st.number_input("RR", min_value=6, max_value=60, value=20)
            with v4: spo2 = st.number_input("SpO₂", min_value=70, max_value=100, value=96)
            with v5: temp_c = st.number_input("Temp (°C)", min_value=32.0, max_value=42.0, value=37.0, step=0.1)
            with v6: avpu = st.selectbox("AVPU", ["A","V","P","U"])

            clinical_notes = st.text_area("Clinical Notes / Why referral?")

            st.markdown("### Reason(s) for Referral")
            reason_checks = st.multiselect(
                "Tick all applicable categories",
                ["ICU_BED_UNAVAILABLE", "SPECIALTY_REQUIRED", "EQUIPMENT_REQUIRED"],
                default=["SPECIALTY_REQUIRED"]
            )

            reason_detail_map = {}
            if "ICU_BED_UNAVAILABLE" in reason_checks:
                reason_detail_map["ICU_BED_UNAVAILABLE"] = st.multiselect(
                    "ICU / bed related",
                    REFERRAL_REASONS["ICU_BED_UNAVAILABLE"]
                )
            if "SPECIALTY_REQUIRED" in reason_checks:
                reason_detail_map["SPECIALTY_REQUIRED"] = st.multiselect(
                    "Specialty required",
                    REFERRAL_REASONS["SPECIALTY_REQUIRED"][case_type]
                )
            if "EQUIPMENT_REQUIRED" in reason_checks:
                reason_detail_map["EQUIPMENT_REQUIRED"] = st.multiselect(
                    "Equipment required",
                    REFERRAL_REASONS["EQUIPMENT_REQUIRED"][case_type]
                )
            other_reason_notes = st.text_area("Other referral notes (free text)")

            # Real-time triage preview
            tmp_case = {
                "case_type": case_type,
                "patient_age": patient_age,
                "patient_sex": patient_sex,
                "vitals": {"hr":hr,"sbp":sbp,"rr":rr,"spo2":spo2,"temp":temp_c,"avpu":avpu}
            }
            on_oxygen = st.checkbox("On oxygen?", value=False)
            spo2_scale2 = st.checkbox("SpO₂ Scale 2 (COPD)?", value=False)

            triage_for_match, triage_details = triage_service.score_based_triage(
                tmp_case, on_oxygen=on_oxygen, spo2_scale2=spo2_scale2, maternal_context=maternal_context
            )
            st.info(f"Triage preview: **{triage_for_match}** via {triage_details.get('system')}")

            # Multi-constraint facility matching
            scored_df, required_caps = match_service.match(
                case_type, triage_for_match, reason_detail_map, other_reason_notes
            )
            st.markdown("### Smart Receiving Facility Match (Auto-ranked)")
            st.caption(f"Required capabilities (derived): {required_caps if required_caps else 'none'}")
            st.dataframe(scored_df.reset_index(drop=True), use_container_width=True)

            top_facilities = scored_df["Facility"].tolist()
            receiving_facility = st.selectbox(
                "Receiving Facility (auto-ranked)",
                options=top_facilities if top_facilities else [f["name"] for f in registry if f["name"]!=DASHBOARD_HOSPITAL],
                index=0
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
                "maternal_context": maternal_context if case_type=="Maternal" else None,
                "icd_code": icd_code,
                "icd_label": icd_label,
                "triage_color": triage_for_match,
                "vitals": {"hr":hr,"sbp":sbp,"rr":rr,"spo2":spo2,"temp":temp_c,"avpu":avpu},
                "clinical_notes": clinical_notes,
                "reason_detail_map": reason_detail_map,
                "other_reason_notes": other_reason_notes,
                "interventions": interventions,
                "eta_minutes": eta_pick,
                "requested_at": case_time,
                "sla_minutes": 15,
                "escalation_level": 0,
                "status": "QUEUED_OFFLINE" if offline_mode else "REQUESTED",
                "transit_updates": []
            }

            if offline_mode:
                data_service.queue_offline(referred_case)
                audit_service.log(st.session_state.user, st.session_state.role, case_id_ref, "REFERRAL_QUEUED_OFFLINE", referred_case)
                st.success(f"Referral queued offline: {case_id_ref}")
            else:
                data_service.save_referral(referred_case)
                audit_service.log(st.session_state.user, st.session_state.role, case_id_ref, "REFERRAL_CREATED", referred_case)
                st.session_state.premium_data["outbound"].append(referred_case)
                st.success(f"Referral created: {case_id_ref}")


# =============================================================================
# TAB 3: CAPACITY MARKET
# =============================================================================
with tab_market:
    st.markdown("## 📈 District Capacity Market (Real-time Stub)")

    reg_rows = []
    for f in registry:
        beds = f["beds"]
        caps = f["capabilities"]
        reg_rows.append({
            "Facility": f["name"],
            "Type": f["type"],
            "ICU Free / Total": f"{beds['icu_free']} / {beds['icu_total']}",
            "HDU Free": beds.get("hdu_free",0),
            "Occupancy %": f["occupancy_pct"],
            "Accepting RED now": "✅" if f.get("accepting_red_now") else "❌",
            "Neuro On-call": "✅" if f["on_call"].get("neuro") else "❌",
            "Cardiac On-call": "✅" if f["on_call"].get("cardiac") else "❌",
            "ObGyn On-call": "✅" if f["on_call"].get("obgyn") else "❌",
            "Ventilators": caps.get("VENTILATOR",0)
        })
    reg_df = pd.DataFrame(reg_rows)
    st.dataframe(reg_df, use_container_width=True)

    # Heatmap
    st.markdown("### Capacity Heatmap")
    hm_df = reg_df.copy()
    hm_df["ICU_Free"] = hm_df["ICU Free / Total"].apply(lambda x: int(str(x).split("/")[0].strip()))
    hm_df["Occupancy"] = hm_df["Occupancy %"].astype(int)
    hm_df = hm_df[["Facility","ICU_Free","Occupancy"]].set_index("Facility")
    fig = px.imshow(hm_df.T, aspect="auto", title="ICU Free & Occupancy (Market View)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Routing engine automatically adds a Market Bonus for RED cases to facilities accepting REDs now.")


# =============================================================================
# TAB 4: OUTCOME ADJUDICATION
# =============================================================================
with tab_outcomes:
    st.markdown("## ✅ Clinical Outcome Adjudication (Ground Truth)")

    # collect RED/YELLOW without outcomes
    pending = []
    for bucket in ["outbound","inbound"]:
        for c in st.session_state.premium_data[bucket]:
            if c.get("triage_color") in ["RED","YELLOW"] and c.get("case_id") not in outcomes_map:
                pending.append(c)

    if not pending:
        st.info("No pending outcomes for adjudication.")
    else:
        case_select = st.selectbox("Select case", [p["case_id"] for p in pending])
        case_obj = next(p for p in pending if p["case_id"] == case_select)

        st.write(f"Case type: **{case_obj.get('case_type')}** | Triage: **{case_obj.get('triage_color')}**")
        with st.form("outcome_form"):
            final_triage = st.selectbox("Final triage at receiving facility", TRIAGE_COLORS)
            deteriorated_in_transit = st.selectbox("Deteriorated in transit?", ["No","Yes"])
            icu_admission = st.selectbox("ICU admission?", ["No","Yes"])
            mortality = st.selectbox("Mortality outcome", ["Alive","Deceased"])
            los_hours = st.number_input("Length of stay (hours)", min_value=0, max_value=2000, value=48)
            avoidable_transfer = st.selectbox("Transfer avoidable (clinical review)?", ["No","Yes"])
            notes = st.text_area("Adjudication notes")

            save_outcome_btn = st.form_submit_button("Save outcome")

        if save_outcome_btn:
            payload = {
                "final_triage": final_triage,
                "deteriorated_in_transit": deteriorated_in_transit,
                "icu_admission": icu_admission,
                "mortality": mortality,
                "length_of_stay_hours": los_hours,
                "avoidable_transfer": avoidable_transfer,
                "notes": notes
            }
            data_service.save_outcome(case_select, payload)
            audit_service.log(st.session_state.user, st.session_state.role, case_select, "OUTCOME_ADJUDICATED", payload)
            st.success("Outcome saved.")


# =============================================================================
# TAB 5: AUDIT & ML FLYWHEEL
# =============================================================================
with tab_audit:
    st.markdown("## 🧾 Audit Log (Immutable)")

    aud_df = audit_service.recent(200)
    if aud_df.empty:
        st.info("No audit events yet.")
    else:
        st.dataframe(aud_df, use_container_width=True)

    st.markdown("---")
    st.markdown("## 🧠 ML Flywheel Dataset")

    ml_df = data_service.load_ml_features(limit=500)
    st.write(f"Captured feature rows: **{len(ml_df)}**")
    if not ml_df.empty:
        st.dataframe(ml_df.head(50), use_container_width=True)

    st.markdown("---")
    st.markdown("### Export helpers")
    if st.button("Export audit CSV"):
        export_path = str(APP_DIR / "audit_export.csv")
        aud_df.to_csv(export_path, index=False)
        st.success(f"Audit exported to {export_path}")

    if st.button("Export ML dataset JSONL"):
        export_path = str(APP_DIR / "ml_dataset.jsonl")
        with open(export_path, "w") as f:
            for _, r in ml_df.iterrows():
                f.write(r["payload"] + "\n")
        st.success(f"ML dataset exported to {export_path}")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    "AHECN Updated Demo MVP • Rules-first clinical safety + ML recommender • "
    "Capacity Market & Digital Twin enabled • SQLite-audited • Offline-ready"
)
