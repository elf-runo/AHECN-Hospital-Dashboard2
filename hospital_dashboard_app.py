# hospital_dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import math
import os
import csv
from pathlib import Path

# joblib is optional – used only if available
try:
    import joblib  # for loading the ML model
except ImportError:
    joblib = None


# =========================
# AI MODEL LOADING + UTILS
# =========================
@st.cache_resource
def load_triage_model():
    """
    Loads my_model.pkl from the same folder as this app.
    If not found (or cannot load), tries creating a tiny demo model as fallback.
    Returns None if neither is possible.
    """
    if joblib is None:
        return None

    app_dir = Path(__file__).resolve().parent
    model_path = app_dir / "my_model.pkl"

    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.sidebar.warning(f"AI model found but failed to load: {e}")
            return None

    # Fallback demo model (only if sklearn exists)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=300,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            random_state=42
        )
        demo = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        demo.fit(X, y)

        try:
            joblib.dump(demo, app_dir / "demo_model.pkl")
        except Exception:
            pass

        st.sidebar.info("Using demo AI model (my_model.pkl not found).")
        return demo
    except Exception:
        return None


def get_triage_model():
    return load_triage_model()


def get_unique_key(component_type, case_type, case_data, extra_suffix=""):
    """
    Generate stable unique keys per case to avoid DuplicateElementKey errors.
    """
    case_id = case_data.get("case_id", "unknown")
    ts = case_data.get("timestamp", None)
    ts_str = ts.strftime("%Y%m%d%H%M%S") if hasattr(ts, "strftime") else "notime"
    return f"{component_type}_{case_type}_{case_id}_{ts_str}_{extra_suffix}"


# Feedback logging
FEEDBACK_LOG_PATH = "ai_feedback_log.csv"
FEEDBACK_FIELDS = [
    "timestamp_utc",
    "case_id",
    "patient_age",
    "sbp",
    "spo2",
    "hr",
    "ai_suggestion",
    "feedback"
]

def log_ai_feedback(case, features, ai_suggestion, feedback):
    """
    Append feedback to CSV log for future analysis / model improvement.
    `features` is expected as a list [age, sbp, spo2, hr]
    """
    try:
        file_exists = os.path.exists(FEEDBACK_LOG_PATH)
        with open(FEEDBACK_LOG_PATH, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp_utc": datetime.utcnow().isoformat(),
                "case_id": case["case_id"],
                "patient_age": features[0],
                "sbp": features[1],
                "spo2": features[2],
                "hr": features[3],
                "ai_suggestion": ai_suggestion,
                "feedback": feedback
            })
    except Exception:
        # If file write fails on cloud, we silently skip to keep app usable
        pass


# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AHECN Hospital Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# =========================
# PREMIUM CSS STYLING
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
}
.case-card.critical { border-left-color: var(--danger); background: #fff5f5; }
.case-card.urgent { border-left-color: var(--warning); background: #fffbf0; }
.case-card.stable { border-left-color: var(--success); background: #f8fff8; }
.metric-highlight {
    background: var(--gradient);
    color: white;
    padding: 1.5rem;
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
# MEDICAL CATALOGS
# =========================
ICD_CATALOG = [
    {"icd_code": "O72.0", "label": "Third-stage haemorrhage", "case_type": "Maternal", "age_min": 12, "age_max": 55},
    {"icd_code": "O72.1", "label": "Immediate postpartum haemorrhage", "case_type": "Maternal", "age_min": 12, "age_max": 55},
    {"icd_code": "O14.1", "label": "Severe pre-eclampsia", "case_type": "Maternal", "age_min": 12, "age_max": 55},
    {"icd_code": "S06.0", "label": "Concussion", "case_type": "Trauma", "age_min": 0, "age_max": 120},
    {"icd_code": "S06.5", "label": "Traumatic subdural haemorrhage", "case_type": "Trauma", "age_min": 0, "age_max": 120},
    {"icd_code": "I63.9", "label": "Cerebral infarction unspecified", "case_type": "Stroke", "age_min": 18, "age_max": 120},
    {"icd_code": "I21.9", "label": "Acute myocardial infarction unspecified", "case_type": "Cardiac", "age_min": 18, "age_max": 120},
    {"icd_code": "A41.9", "label": "Sepsis unspecified organism", "case_type": "Sepsis", "age_min": 0, "age_max": 120},
    {"icd_code": "J96.0", "label": "Acute respiratory failure", "case_type": "Other", "age_min": 0, "age_max": 120},
]

INTERVENTION_PROTOCOLS = {
    "Maternal": ["IV fluids", "Uterotonics", "TXA", "Oxygen", "BP monitoring", "Fundal massage", "Emergency C-section prep"],
    "Trauma": ["Airway management", "IV access", "Bleeding control", "Immobilization", "Pain management", "Chest tube", "Emergency surgery prep"],
    "Stroke": ["BP control", "Glucose check", "Neurological assessment", "Frequent monitoring", "CT scan", "Thrombolysis prep"],
    "Cardiac": ["Aspirin", "Oxygen", "IV access", "ECG monitoring", "Nitroglycerin", "Cath lab activation"],
    "Sepsis": ["Antibiotics", "IV fluids", "Oxygen", "Blood cultures", "Vasopressors", "Lactate monitoring"],
    "Other": ["Oxygen", "IV access", "Symptom management", "Monitoring", "Specialist consult"]
}

EMT_CREW = [
    {"id": "EMT_001", "name": "John Carter", "level": "ALS", "vehicle": "Ambulance 1", "status": "active"},
    {"id": "EMT_002", "name": "Sarah Connor", "level": "BLS", "vehicle": "Ambulance 2", "status": "active"},
    {"id": "EMT_003", "name": "Mike Rodriguez", "level": "ALS", "vehicle": "Ambulance 3", "status": "available"},
    {"id": "EMT_004", "name": "Lisa Park", "level": "Critical Care", "vehicle": "Mobile ICU", "status": "available"},
]

# =========================
# CLINICALLY-ACCEPTED SCORING ENGINE
# =========================

def calculate_qsofa(rr, sbp, avpu):
    """
    qSOFA adult criteria:
    - RR >= 22 -> 1
    - SBP <= 100 -> 1
    - Altered mentation (not Alert) -> 1
    Score 0-3. qSOFA >=2 = high risk.
    """
    score = 0
    if rr >= 22:
        score += 1
    if sbp <= 100:
        score += 1
    if str(avpu).upper() != "A":  # A=Alert, anything else = altered
        score += 1
    return score


def calculate_news2(age, rr, spo2, sbp, hr, temp_c, avpu, on_oxygen=False, spo2_scale2=False):
    """
    NEWS2 for adults (>=16 yrs). Uses RCP standard bands.
    For MVP: default Scale 1 SpO2 unless spo2_scale2=True.
    """

    # RR scoring
    if rr <= 8: rr_score = 3
    elif 9 <= rr <= 11: rr_score = 1
    elif 12 <= rr <= 20: rr_score = 0
    elif 21 <= rr <= 24: rr_score = 2
    else: rr_score = 3  # >=25

    # SpO2 scoring (Scale 1 default)
    if not spo2_scale2:
        if spo2 <= 91: spo2_score = 3
        elif 92 <= spo2 <= 93: spo2_score = 2
        elif 94 <= spo2 <= 95: spo2_score = 1
        else: spo2_score = 0  # >=96
    else:
        # Scale 2 (COPD/hypercapnic) – use only if clinician selects it
        if spo2 <= 83: spo2_score = 3
        elif 84 <= spo2 <= 85: spo2_score = 2
        elif 86 <= spo2 <= 87: spo2_score = 1
        elif 88 <= spo2 <= 92: spo2_score = 0
        else: spo2_score = 3  # >=93 on Scale 2 is abnormal high

    # Oxygen adds 2 points
    oxygen_score = 2 if on_oxygen else 0

    # SBP scoring
    if sbp <= 90: sbp_score = 3
    elif 91 <= sbp <= 100: sbp_score = 2
    elif 101 <= sbp <= 110: sbp_score = 1
    elif 111 <= sbp <= 219: sbp_score = 0
    else: sbp_score = 3  # >=220

    # HR scoring
    if hr <= 40: hr_score = 3
    elif 41 <= hr <= 50: hr_score = 1
    elif 51 <= hr <= 90: hr_score = 0
    elif 91 <= hr <= 110: hr_score = 1
    elif 111 <= hr <= 130: hr_score = 2
    else: hr_score = 3  # >=131

    # Temp scoring
    if temp_c <= 35.0: temp_score = 3
    elif 35.1 <= temp_c <= 36.0: temp_score = 1
    elif 36.1 <= temp_c <= 38.0: temp_score = 0
    elif 38.1 <= temp_c <= 39.0: temp_score = 1
    else: temp_score = 2  # >=39.1

    # Consciousness scoring (AVPU / new confusion)
    avpu_u = str(avpu).upper()
    if avpu_u == "A":
        conc_score = 0
    else:
        conc_score = 3  # V/P/U/CVPU/new confusion

    total = rr_score + spo2_score + oxygen_score + sbp_score + hr_score + temp_score + conc_score

    parts = {
        "rr": rr_score,
        "spo2": spo2_score,
        "oxygen": oxygen_score,
        "sbp": sbp_score,
        "hr": hr_score,
        "temp": temp_score,
        "conc": conc_score
    }
    return total, parts


def calculate_pews_placeholder(age, rr, spo2, sbp, hr, avpu):
    """
    MVP placeholder PEWS:
    You should replace this later with your chosen PEWS chart.
    For now, return a simple risk score 0-6 based on gross abnormalities.
    """
    score = 0
    # crude pediatric risk flags
    if rr > 40 or rr < 10: score += 2
    if spo2 < 92: score += 2
    if sbp < 80: score += 1
    if hr > 160 or hr < 60: score += 1
    if str(avpu).upper() != "A": score += 2
    return min(score, 6)


def score_based_triage(case, on_oxygen=False, spo2_scale2=False):
    """
    Returns:
      triage_color, explanation_dict
    """
    vitals = case["vitals"]
    age = float(case["patient_age"])
    rr = float(vitals["rr"])
    spo2 = float(vitals["spo2"])
    sbp = float(vitals["sbp"])
    hr = float(vitals["hr"])
    temp_c = float(vitals["temp"])
    avpu = vitals.get("avpu", "A")

    if age < 16:
        pews = calculate_pews_placeholder(age, rr, spo2, sbp, hr, avpu)
        # simple mapping
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

    # Adults (>=16)
    news2_total, news2_parts = calculate_news2(
        age, rr, spo2, sbp, hr, temp_c, avpu,
        on_oxygen=on_oxygen,
        spo2_scale2=spo2_scale2
    )
    qsofa = calculate_qsofa(rr, sbp, avpu)

    # NEWS2 escalation thresholds:
    # 0-4 low risk, 5-6 urgent, >=7 emergency
    if news2_total >= 7:
        triage = "RED"
    elif news2_total >= 5:
        triage = "YELLOW"
    else:
        triage = "GREEN"

    # Upgrade one level if qSOFA >=2
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
    case_types = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]

    referred_cases, received_cases = [], []
    base_time = datetime.now() - timedelta(days=days_back)

    for day in range(days_back):
        daily_cases = random.randint(3, 8)
        for case_num in range(daily_cases):
            case_time = base_time + timedelta(days=day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            case_type = random.choice(case_types)
            age = random.randint(18, 80) if case_type != "Maternal" else random.randint(18, 40)

            if case_type == "Maternal":
                vitals = {"hr": random.randint(90, 140), "sbp": random.randint(80, 160), "rr": random.randint(18, 30),
                          "temp": round(random.uniform(36.5, 38.5), 1), "spo2": random.randint(92, 99), "avpu": "A"}
            elif case_type == "Trauma":
                vitals = {"hr": random.randint(70, 150), "sbp": random.randint(70, 180), "rr": random.randint(16, 35),
                          "temp": round(random.uniform(36.0, 38.0), 1), "spo2": random.randint(88, 98),
                          "avpu": random.choices(["A", "V", "P"], weights=[0.7, 0.2, 0.1])[0]}
            else:
                vitals = {"hr": random.randint(80, 140), "sbp": random.randint(90, 150), "rr": random.randint(18, 32),
                          "temp": round(random.uniform(36.5, 39.0), 1), "spo2": random.randint(86, 95),
                          "avpu": random.choices(["A", "V"], weights=[0.8, 0.2])[0]}

            matching_icd = [icd for icd in ICD_CATALOG if icd["case_type"] == case_type and icd["age_min"] <= age <= icd["age_max"]]
            icd = random.choice(matching_icd) if matching_icd else random.choice([i for i in ICD_CATALOG if i["case_type"] == case_type])

            interventions = random.sample(INTERVENTION_PROTOCOLS[case_type], random.randint(2, 4))
            referring_facility = random.choice(["PHC Mawlai", "CHC Smit", "CHC Pynursla", "Rural Health Center"])
            receiving_facility = random.choice(facilities)
            emt_crew = random.choice(EMT_CREW)
            transport_time = random.randint(20, 90)

            case_id_ref = f"REF_{case_time.strftime('%Y%m%d')}_{case_num:03d}"
            case_id_rec = f"REC_{case_time.strftime('%Y%m%d')}_{case_num:03d}"

            referred_case = {
                "case_id": case_id_ref,
                "timestamp": case_time,
                "referring_facility": referring_facility,
                "receiving_facility": receiving_facility["name"],
                "patient_age": age,
                "patient_sex": "F" if case_type == "Maternal" else random.choice(["M", "F"]),
                "case_type": case_type,
                "icd_code": icd["icd_code"],
                "icd_label": icd["label"],
                "triage_color": random.choices(["RED", "YELLOW", "GREEN"], weights=[0.3, 0.5, 0.2])[0],
                "vitals": vitals,
                "interventions_referring": interventions,
                "status": random.choices(["Accepted", "Rejected", "Pending"], weights=[0.8, 0.1, 0.1])[0],
                "clinical_notes": f"Patient presented with {case_type.lower()} symptoms requiring specialist care"
            }

            received_case = {
                **referred_case,
                "case_id": case_id_rec,
                "transport_time_minutes": transport_time,
                "emt_crew": emt_crew,
                "vehicle_id": emt_crew["vehicle"],
                "interventions_receiving": interventions + random.sample(INTERVENTION_PROTOCOLS[case_type], random.randint(1, 3)),
                "final_outcome": random.choices(["Excellent", "Good", "Fair", "Poor"], weights=[0.4, 0.3, 0.2, 0.1])[0],
                "length_of_stay_hours": random.randint(24, 240),
                "discharge_date": case_time + timedelta(hours=random.randint(24, 240))
            }

            referred_cases.append(referred_case)
            received_cases.append(received_case)

    ref_df = pd.DataFrame(referred_cases)
    rec_df = pd.DataFrame(received_cases)
    ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"])
    rec_df["timestamp"] = pd.to_datetime(rec_df["timestamp"])

    return {
        "referred_cases": ref_df,
        "received_cases": rec_df,
        "facilities": facilities,
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
# UI COMPONENTS
# =========================
def render_premium_header():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.5rem;">🏥 AHECN Hospital Command Center</h1>
        <p style="margin:0; font-size:1.2rem; opacity:0.9;">Advanced Emergency Care Coordination Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    data = st.session_state.premium_data

    total_cases = len(data["referred_cases"])
    critical_cases = len(data["referred_cases"][data["referred_cases"]["triage_color"] == "RED"])
    urgent_cases = len(data["referred_cases"][data["referred_cases"]["triage_color"] == "YELLOW"])
    acceptance_rate = (len(data["referred_cases"][data["referred_cases"]["status"] == "Accepted"]) / total_cases * 100) if total_cases else 0

    with col1:
        st.markdown(f"""<div class="metric-highlight"><div style="font-size:2rem; font-weight:bold;">{total_cases}</div><div>Total Cases</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-highlight"><div style="font-size:2rem; font-weight:bold;">{critical_cases}</div><div>Critical RED</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-highlight"><div style="font-size:2rem; font-weight:bold;">{urgent_cases}</div><div>Urgent YELLOW</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-highlight"><div style="font-size:2rem; font-weight:bold;">{acceptance_rate:.1f}%</div><div>Acceptance Rate</div></div>""", unsafe_allow_html=True)
    with col5:
        avg_transport = data["received_cases"]["transport_time_minutes"].mean()
        st.markdown(f"""<div class="metric-highlight"><div style="font-size:2rem; font-weight:bold;">{avg_transport:.0f}m</div><div>Avg Transport</div></div>""", unsafe_allow_html=True)


def render_case_calendar():
    st.markdown("### 📅 Case Calendar & Timeline")
    col1, col2 = st.columns([2, 1])

    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(
                st.session_state.date_filters["start_date"],
                st.session_state.date_filters["end_date"]
            ),
            key="global_date_range_selector"
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            st.session_state.date_filters["start_date"] = date_range[0]
            st.session_state.date_filters["end_date"] = date_range[1]

        referred_df = st.session_state.premium_data["referred_cases"]
        mask = (
            (referred_df["timestamp"].dt.date >= st.session_state.date_filters["start_date"]) &
            (referred_df["timestamp"].dt.date <= st.session_state.date_filters["end_date"])
        )
        filtered_referred = referred_df[mask]

        if not filtered_referred.empty:
            timeline_data = filtered_referred.copy()
            timeline_data["date"] = timeline_data["timestamp"].dt.date
            timeline_data = timeline_data.groupby("date").size().reset_index(name="count")
            fig = px.line(timeline_data, x="date", y="count", title="Daily Case Volume Trend", markers=True)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cases found in selected date range")

    with col2:
        st.markdown("#### Quick Filters")
        st.multiselect("Case Types", ["Maternal","Trauma","Stroke","Cardiac","Sepsis","Other"], ["Maternal","Trauma","Cardiac"], key="case_type_filter")
        st.multiselect("Triage Levels", ["RED","YELLOW","GREEN"], ["RED","YELLOW"], key="triage_filter")
        facilities = ["Tertiary Central Hospital","District North General","Specialty South Medical","Trauma East Center"]
        st.multiselect("Receiving Facilities", facilities, facilities, key="facility_filter")
        if st.button("Apply Filters", key="apply_filters_btn"):
            st.success(f"Filters applied to {len(filtered_referred)} cases")


def render_interactive_case_list(case_type="referred"):
    key_prefix = f"{case_type}_cases"
    st.markdown(f"### 📋 {'Referred' if case_type == 'referred' else 'Received'} Cases")

    cases_df = st.session_state.premium_data[f"{case_type}_cases"]
    mask = (
        (cases_df["timestamp"].dt.date >= st.session_state.date_filters["start_date"]) &
        (cases_df["timestamp"].dt.date <= st.session_state.date_filters["end_date"])
    )
    filtered_cases = cases_df[mask]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_term = st.text_input("🔍 Search cases...", placeholder="Search by case ID, facility, or diagnosis", key=f"{key_prefix}_search")
    with col2:
        sort_by = st.selectbox("Sort by", ["Timestamp (Newest)","Timestamp (Oldest)","Triage","Case Type"], key=f"{key_prefix}_sort")
    with col3:
        items_per_page = st.selectbox("Items per page", [10,25,50], 0, key=f"{key_prefix}_pagination")

    if search_term:
        filtered_cases = filtered_cases[
            filtered_cases["case_id"].str.contains(search_term, case=False) |
            filtered_cases["referring_facility"].str.contains(search_term, case=False) |
            filtered_cases["icd_label"].str.contains(search_term, case=False) |
            filtered_cases["receiving_facility"].str.contains(search_term, case=False)
        ]

    if sort_by == "Timestamp (Newest)":
        filtered_cases = filtered_cases.sort_values("timestamp", ascending=False)
    elif sort_by == "Timestamp (Oldest)":
        filtered_cases = filtered_cases.sort_values("timestamp", ascending=True)
    elif sort_by == "Triage":
        triage_order = {"RED":1,"YELLOW":2,"GREEN":3}
        filtered_cases = filtered_cases.sort_values("triage_color", key=lambda x: x.map(triage_order))
    else:
        filtered_cases = filtered_cases.sort_values("case_type")

    if filtered_cases.empty:
        st.info("No cases found matching the current filters")
        return

    total_pages = int(len(filtered_cases)/items_per_page) + (1 if len(filtered_cases) % items_per_page else 0)
    page_number = st.number_input("Page", min_value=1, max_value=max(total_pages,1), value=1, key=f"{key_prefix}_page")
    start_idx = (page_number-1)*items_per_page
    paginated_cases = filtered_cases.iloc[start_idx:start_idx+items_per_page]

    st.write(f"Showing {len(paginated_cases)} of {len(filtered_cases)} cases")

    for display_idx, (_, case) in enumerate(paginated_cases.iterrows()):
        render_case_card(case, case_type, display_idx)


def render_case_card(case, case_type, index):
    triage_map = {"RED":"critical","YELLOW":"urgent","GREEN":"stable"}
    triage_class = triage_map.get(case["triage_color"], "")
    case_class = f"case-card {triage_class}"

    st.markdown(f"""
    <div class="{case_class}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div><strong>{case['case_id']}</strong> • {case['icd_label']}</div>
            <div style="background:{'#ff4444' if case['triage_color']=='RED' else '#ffaa00' if case['triage_color']=='YELLOW' else '#00c853'};
                        color:white; padding:0.2rem 0.8rem; border-radius:15px; font-size:0.8rem;">
                {case['triage_color']}
            </div>
        </div>
        <div style="color:#666; font-size:0.9rem; margin-top:0.5rem;">
            {case['referring_facility']} → {case['receiving_facility']} •
            {case['timestamp'].strftime('%Y-%m-%d %H:%M')} •
            {case['patient_age']}{case['patient_sex']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"View full details for {case['case_id']}"):
        render_case_details(case, case_type)

def heuristic_triage(age, sbp, spo2, hr):
    """
    Simple clinical fallback triage if ML model fails.
    Returns (label, reasoning).
    """
    score = 0
    reasons = []

    # Oxygen
    if spo2 < 90:
        score += 2; reasons.append("SpO₂ < 90%")
    elif spo2 < 94:
        score += 1; reasons.append("SpO₂ 90–93%")

    # Blood pressure
    if sbp < 90:
        score += 2; reasons.append("SBP < 90 mmHg")
    elif sbp < 100:
        score += 1; reasons.append("SBP 90–99 mmHg")

    # Heart rate
    if hr > 130:
        score += 2; reasons.append("HR > 130 bpm")
    elif hr > 110:
        score += 1; reasons.append("HR 111–130 bpm")

    # Age risk add-on (light weight)
    if age >= 65:
        score += 1; reasons.append("Age ≥ 65")

    if score >= 4:
        return "RED", "High-risk vitals: " + ", ".join(reasons)
    elif score >= 2:
        return "YELLOW", "Moderate-risk vitals: " + ", ".join(reasons)
    else:
        return "GREEN", "Low-risk vitals"


def safe_ai_predict(model, features):
    """
    Safely run model.predict().
    If prediction fails due to incompatibility, fall back to heuristic.
    Returns: (triage_label, method, error_text, reasoning)
    """
    age, sbp, spo2, hr = features

    if model is not None:
        try:
            pred = model.predict(np.array([features], dtype=float))[0]
            label = str(pred).upper()

            # If model outputs numeric class, map to labels
            if isinstance(pred, (int, np.integer)):
                triage_levels = ["GREEN", "YELLOW", "RED"]
                label = triage_levels[int(pred) % 3]

            if label not in ("RED", "YELLOW", "GREEN"):
                label = "YELLOW"

            return label, "ml_model", None, "ML model prediction"

        except Exception as e:
            # fall through to heuristic
            label, reasoning = heuristic_triage(age, sbp, spo2, hr)
            return label, "heuristic_fallback", str(e), reasoning

    # No model available
    label, reasoning = heuristic_triage(age, sbp, spo2, hr)
    return label, "heuristic_fallback", "Model missing", reasoning

def render_case_details(case, case_type):
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("#### Patient Information")
        st.write(f"**Case ID:** {case['case_id']}")
        st.write(f"**Age/Sex:** {case['patient_age']}{case['patient_sex']}")
        st.write(f"**Diagnosis:** {case['icd_label']} ({case['icd_code']})")
        st.write(f"**Triage:** {case['triage_color']}")
        st.write(f"**Referring Facility:** {case['referring_facility']}")
        st.write(f"**Receiving Facility:** {case['receiving_facility']}")

        st.markdown("#### Vital Signs")
        vitals = case["vitals"]
        st.write(f"**HR:** {vitals['hr']} bpm | **SBP:** {vitals['sbp']} mmHg")
        st.write(f"**RR:** {vitals['rr']} rpm | **SpO₂:** {vitals['spo2']}%")
        st.write(f"**Temp:** {vitals['temp']}°C | **AVPU:** {vitals['avpu']}")

    with col2:
        st.markdown("#### Timeline & Interventions")
        st.markdown("**Referring Interventions:**")
        for itv in case["interventions_referring"]:
            st.markdown(f'<div class="intervention-badge">{itv}</div>', unsafe_allow_html=True)

        if case_type == "received":
            st.markdown("**Receiving Interventions:**")
            for itv in case["interventions_receiving"]:
                st.markdown(f'<div class="intervention-badge">{itv}</div>', unsafe_allow_html=True)

            st.markdown("#### Transport Details")
            st.write(f"**Transport Time:** {case['transport_time_minutes']} minutes")
            st.write(f"**EMT Crew:** {case['emt_crew']['name']} ({case['emt_crew']['level']})")
            st.write(f"**Vehicle:** {case['vehicle_id']}")
            st.write(f"**Outcome:** {case['final_outcome']}")
            st.write(f"**Length of Stay:** {case['length_of_stay_hours']} hours")

    # =========================
    # AI / SCORE-BASED TRIAGE PANEL
    # =========================
    st.markdown("#### 🧠 Clinical Triage Recommendation (Score-Based)")
    st.caption("Uses NEWS2 + qSOFA for adults, PEWS for paediatrics. Deterministic and clinically accepted.")

    vitals = case["vitals"]
    age = float(case["patient_age"])
    sbp = float(vitals["sbp"])
    spo2 = float(vitals["spo2"])
    hr = float(vitals["hr"])
    rr = float(vitals["rr"])
    temp_c = float(vitals["temp"])
    avpu = vitals.get("avpu", "A")

    # Optional oxygen input for NEWS2
    colA, colB = st.columns(2)
    with colA:
        on_oxygen = st.checkbox("Patient on supplemental oxygen?", value=False, key=f"oxy_{case['case_id']}")
    with colB:
        spo2_scale2 = st.checkbox("Use SpO₂ Scale 2 (COPD/hypercapnic)?", value=False, key=f"spo2s2_{case['case_id']}")

    # Show inputs
    c1, c2, c3 = st.columns(3)
    c1.metric("Age", f"{int(age)} yrs")
    c1.metric("SBP", f"{sbp} mmHg")
    c2.metric("SpO₂", f"{spo2}%")
    c2.metric("RR", f"{rr}/min")
    c3.metric("HR", f"{hr} bpm")
    c3.metric("Temp", f"{temp_c} °C")

    # Compute score-based triage
    triage_color, details = score_based_triage(case, on_oxygen=on_oxygen, spo2_scale2=spo2_scale2)

    # Display result
    color_map = {"RED": "#ff4444", "YELLOW": "#ffaa00", "GREEN": "#00c853"}
    triage_hex = color_map[triage_color]

    st.markdown(f"""
    <div style="
        margin: 0.5rem 0 1rem 0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: {triage_hex}15;
        border-left: 6px solid {triage_hex};
    ">
        <h4 style="margin:0; color:{triage_hex};">Recommended Triage: {triage_color}</h4>
        <div style="color:#333; margin-top:0.5rem;">
            <b>System:</b> {details.get("system")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if details.get("system") == "NEWS2 + qSOFA":
        st.write(f"**NEWS2 Total:** {details['news2_total']}  (RR:{details['news2_parts']['rr']}, "
                 f"SpO₂:{details['news2_parts']['spo2']}, O₂:{details['news2_parts']['oxygen']}, "
                 f"SBP:{details['news2_parts']['sbp']}, HR:{details['news2_parts']['hr']}, "
                 f"Temp:{details['news2_parts']['temp']}, AVPU:{details['news2_parts']['conc']})")
        st.write(f"**qSOFA:** {details['qsofa']} (≥2 upgrades risk)")

        if details["news2_total"] >= 7:
            st.info("NEWS2 ≥7 = emergency response threshold.")
        elif details["news2_total"] >= 5:
            st.info("NEWS2 5–6 = urgent clinical review threshold.")
        else:
            st.info("NEWS2 0–4 = low risk / routine monitoring.")

    else:
        st.write(f"**PEWS Score (placeholder):** {details['score']}")
        st.caption(details.get("notes", ""))

    # =========================
    # Optional ML Model Comparison
    # =========================
    model = get_triage_model()
    if model is not None:
        try:
            features = np.array([[age, sbp, spo2, hr]], dtype=float)
            ml_pred = str(model.predict(features)[0])
            st.markdown("#### 🤖 ML Model Suggestion (Optional)")
            st.write(f"**Model output:** {ml_pred}")
            st.caption("For demo only. Production model must be validated against local clinical pathways.")
        except Exception as e:
            st.caption(f"ML model available but could not run here: {e}")



    # Feedback UI (only if AI ran)
    last_pred_key = get_unique_key("ai_last_pred", case_type, case)
    last_feat_key = get_unique_key("ai_last_features", case_type, case)

    if last_pred_key in st.session_state and last_feat_key in st.session_state:
        st.subheader("Was this AI recommendation helpful?")
        fb_radio_key = get_unique_key("ai_fb_radio", case_type, case)
        feedback = st.radio("Feedback", ("Yes", "No", "Unsure"), key=fb_radio_key, horizontal=True)

        fb_submit_key = get_unique_key("ai_fb_submit", case_type, case)
        if st.button("Submit Feedback", key=fb_submit_key):
            log_ai_feedback(
                case,
                st.session_state[last_feat_key],
                st.session_state[last_pred_key],
                feedback
            )
            st.success("Thanks — feedback logged.")


def render_advanced_analytics():
    st.markdown("### 📊 Advanced Analytics Dashboard")
    data = st.session_state.premium_data
    filtered_cases = data["referred_cases"][
        (data["referred_cases"]["timestamp"].dt.date >= st.session_state.date_filters["start_date"]) &
        (data["referred_cases"]["timestamp"].dt.date <= st.session_state.date_filters["end_date"])
    ]

    col1, col2, col3, col4 = st.columns(4)
    total_cases = len(filtered_cases)
    col1.metric("Total Cases", total_cases)

    acceptance_rate = (len(filtered_cases[filtered_cases["status"] == "Accepted"]) / total_cases * 100) if total_cases else 0
    col2.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")

    avg_transport = data["received_cases"]["transport_time_minutes"].mean()
    col3.metric("Avg Transport Time", f"{avg_transport:.1f} min")

    critical_cases = len(filtered_cases[filtered_cases["triage_color"] == "RED"])
    col4.metric("Critical Cases", critical_cases)

    if filtered_cases.empty:
        st.info("No data available for analytics with current filters")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Case Type Analysis")
        case_counts = filtered_cases["case_type"].value_counts()
        st.plotly_chart(px.pie(values=case_counts.values, names=case_counts.index, title="Cases by Type"), use_container_width=True)

    with col2:
        st.markdown("#### Triage Distribution")
        triage_counts = filtered_cases["triage_color"].value_counts()
        st.plotly_chart(px.bar(x=triage_counts.index, y=triage_counts.values, title="Cases by Triage Level"), use_container_width=True)


def render_emt_tracking():
    st.markdown("### 🚑 Real-time EMT Tracking")
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("#### Active Transports")
        active_transports = st.session_state.premium_data["received_cases"].tail(3)
        for _, transport in active_transports.iterrows():
            progress = random.randint(30,90)
            st.markdown(f"""
            <div class="premium-card">
                <strong>{transport['case_id']}</strong><br>
                <small>{transport['emt_crew']['name']} • {transport['vehicle_id']}</small>
                <div style="margin-top:0.8rem;">
                    <div style="background:#f0f0f0; border-radius:10px; height:10px;">
                        <div style="background:#2196f3; width:{progress}%; height:100%; border-radius:10px;"></div>
                    </div>
                </div>
                <div style="margin-top:0.5rem; font-size:0.9rem; color:#666;">
                    {progress}% complete • ETA {transport['transport_time_minutes']} min
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Crew Status")
        for idx, crew in enumerate(st.session_state.premium_data["emt_crews"]):
            status_color = "🟢" if crew["status"]=="active" else "🟡"
            st.write(f"{status_color} **{crew['name']}**")
            st.write(f"{crew['level']} • {crew['vehicle']}")
            if idx < len(st.session_state.premium_data["emt_crews"]) - 1:
                st.write("---")


def render_quick_actions():
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh All Data", use_container_width=True, key="refresh_data_btn_main"):
            st.session_state.premium_data = generate_premium_synthetic_data(days_back=60)
            st.success("Data refreshed successfully!")
        st.button("📊 Generate Report", use_container_width=True, key="generate_report_btn_main")

    with col2:
        st.button("🚨 Emergency Mode", use_container_width=True, key="emergency_mode_btn_main")
        st.button("📋 Case Summary", use_container_width=True, key="case_summary_btn_main")

    with col3:
        st.button("📧 Notify Staff", use_container_width=True, key="notify_staff_btn_main")
        st.button("🖨️ Export Data", use_container_width=True, key="export_data_btn_main")


def render_dashboard_overview():
    col1, col2 = st.columns([2,1])
    with col1:
        render_case_calendar()
        st.markdown("### 📈 Recent Activity")
        recent_cases = st.session_state.premium_data["referred_cases"].tail(10)
        for _, case in recent_cases.iterrows():
            st.markdown(f"""
            <div class="timeline-event">
                <strong>{case['case_id']}</strong> - {case['icd_label']}<br>
                <small>{case['timestamp'].strftime('%Y-%m-%d %H:%M')} • {case['triage_color']} • {case['referring_facility']}</small>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🖥️ System Status")
        st.write("Database: ✅ Optimal")
        st.write("API Services: ✅ Stable")
        st.write("EMT Tracking: ✅ Active")
        st.write("Analytics: ⚠️ Processing")


def render_premium_sidebar():
    st.sidebar.markdown("### 🔔 Live Alerts")
    critical_cases = st.session_state.premium_data["referred_cases"][
        st.session_state.premium_data["referred_cases"]["triage_color"]=="RED"
    ].tail(3)

    for _, case in critical_cases.iterrows():
        st.sidebar.error(f"**{case['case_id']}**\n\n{case['icd_label']}\n\n*{case['timestamp'].strftime('%H:%M')} • {case['referring_facility']}*")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Performance")
    metrics_df = st.session_state.premium_data["referred_cases"]
    acceptance_rate = (len(metrics_df[metrics_df["status"]=="Accepted"]) / len(metrics_df) * 100) if len(metrics_df) else 0
    st.sidebar.metric("Response Rate", f"{acceptance_rate:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Status")
    app_dir = Path(__file__).resolve().parent
    if (app_dir / "my_model.pkl").exists():
        st.sidebar.success("AI Model: Ready")
    else:
        st.sidebar.warning("AI Model: Missing")


# =========================
# MAIN APP
# =========================
def main():
    initialize_session_state()
    render_premium_header()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard",
        "📤 Referred Cases",
        "🏥 Received Cases",
        "📊 Analytics",
        "🚑 EMT Tracking",
        "⚡ Quick Actions"
    ])

    with tab1:
        render_dashboard_overview()
    with tab2:
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
