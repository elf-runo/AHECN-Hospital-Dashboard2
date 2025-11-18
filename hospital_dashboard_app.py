# hospital_dashboard_app.py - COMPLETE SELF-CONTAINED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import math

# === CSS STYLING ===
st.markdown("""
<style>
:root {
    --primary: #1f77b4;
    --secondary: #ff7f0e;
    --success: #2ca02c;
    --danger: #d62728;
    --warning: #ff7f0e;
    --info: #17a2b8;
}

.card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid var(--primary);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.triage-red { border-left-color: var(--danger); background: #ffebee; }
.triage-yellow { border-left-color: var(--warning); background: #fff3e0; }
.triage-green { border-left-color: var(--success); background: #e8f5e8; }

.metric-card {
    background: white;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.intervention-item {
    background: white;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
    border-left: 4px solid #3498db;
}
</style>
""", unsafe_allow_html=True)

# === MEDICAL DATA CATALOGS ===
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
    "Maternal": ["IV fluids", "Uterotonics", "TXA", "Oxygen", "BP monitoring"],
    "Trauma": ["Airway management", "IV access", "Bleeding control", "Immobilization", "Pain management"],
    "Stroke": ["BP control", "Glucose check", "Neurological assessment", "Frequent monitoring"],
    "Cardiac": ["Aspirin", "Oxygen", "IV access", "ECG monitoring", "Nitroglycerin"],
    "Sepsis": ["Antibiotics", "IV fluids", "Oxygen", "Blood cultures", "Vasopressors"],
    "Other": ["Oxygen", "IV access", "Symptom management", "Monitoring"]
}

EMT_INTERVENTIONS = {
    "Airway": ["Oxygen administration", "Airway positioning", "Suction", "Advanced airway"],
    "Breathing": ["Chest seal", "Needle decompression", "Ventilation assistance"],
    "Circulation": ["IV access", "Fluid resuscitation", "Bleeding control", "Tourniquet"],
    "Disability": ["Spinal immobilization", "Head stabilization", "Seizure management"]
}

# === TRIAGING ALGORITHM ===
def _num(x):
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    try: return float(s)
    except Exception: return None

def _clip(v, lo, hi):
    x = _num(v)
    if x is None: return None
    return max(lo, min(hi, x))

def validate_vitals(hr, rr, sbp, temp, spo2):
    return dict(
        hr   = _clip(hr,   20, 240),
        rr   = _clip(rr,    5,  60),
        sbp  = _clip(sbp,  50, 260),
        temp = _clip(temp, 32,  42),
        spo2 = _clip(spo2, 50, 100),
    )

def calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device="Air", spo2_scale=1):
    rr, spo2, sbp, hr, temp = (_num(rr), _num(spo2), _num(sbp), _num(hr), _num(temp))
    avpu = "A" if avpu is None else str(avpu).strip().upper()
    spo2_scale = int(spo2_scale) if spo2_scale else 1
    o2_device = "Air" if not o2_device else str(o2_device).strip()

    hits, score = [], 0

    # RR scoring
    if rr is None: pass
    elif rr <= 8:      score += 3; hits.append("NEWS2 RR ≤8 =3")
    elif 9 <= rr <=11: score += 1; hits.append("NEWS2 RR 9–11 =1")
    elif 12 <= rr <=20:                 hits.append("NEWS2 RR 12–20 =0")
    elif 21 <= rr <=24: score += 2; hits.append("NEWS2 RR 21–24 =2")
    else:               score += 3; hits.append("NEWS2 RR ≥25 =3")

    # SpO2 scoring
    def spo2_s1(s): return 3 if s<=91 else 2 if s<=93 else 1 if s<=95 else 0
    def spo2_s2(s): return 3 if s<=83 else 2 if s<=85 else 1 if s<=90 else 0 if s<=92 else 0
    if spo2 is not None:
        pts = spo2_s1(spo2) if spo2_scale==1 else spo2_s2(spo2)
        score += pts; hits.append(f"NEWS2 SpO₂ (scale {spo2_scale}) +{pts}")
    if str(o2_device).lower() != "air":
        score += 2; hits.append("NEWS2 Supplemental O₂ +2")

    # SBP scoring
    if sbp is not None:
        if sbp <= 90:        score += 3; hits.append("NEWS2 SBP ≤90 =3")
        elif sbp <=100:      score += 2; hits.append("NEWS2 SBP 91–100 =2")
        elif sbp <=110:      score += 1; hits.append("NEWS2 SBP 101–110 =1")
        elif sbp <=219:                     hits.append("NEWS2 SBP 111–219 =0")
        else:                score += 3; hits.append("NEWS2 SBP ≥220 =3")

    # HR scoring
    if hr is not None:
        if hr <= 40:         score += 3; hits.append("NEWS2 HR ≤40 =3")
        elif hr <= 50:       score += 1; hits.append("NEWS2 HR 41–50 =1")
        elif hr <= 90:                      hits.append("NEWS2 HR 51–90 =0")
        elif hr <=110:       score += 1; hits.append("NEWS2 HR 91–110 =1")
        elif hr <=130:       score += 2; hits.append("NEWS2 HR 111–130 =2")
        else:                score += 3; hits.append("NEWS2 HR ≥131 =3")

    # Temp scoring
    if temp is not None:
        if temp <= 35.0:         score += 3; hits.append("NEWS2 Temp ≤35.0 =3")
        elif temp <= 36.0:       score += 1; hits.append("NEWS2 Temp 35.1–36.0 =1")
        elif temp <= 38.0:                        hits.append("NEWS2 Temp 36.1–38.0 =0")
        elif temp <= 39.0:       score += 1; hits.append("NEWS2 Temp 38.1–39.0 =1")
        else:                    score += 2; hits.append("NEWS2 Temp ≥39.1 =2")

    # AVPU scoring
    if avpu != "A":
        score += 3; hits.append("NEWS2 AVPU ≠ A =3")

    return score, hits, (5 <= score < 7), (score >= 7)

def safe_calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device="Air", spo2_scale=1):
    try:
        raw = calc_NEWS2(rr, spo2, sbp, hr, temp, avpu, o2_device, spo2_scale)
        return raw
    except Exception as e:
        return 0, [f"NEWS2 error: {str(e)}"], False, False

def calc_qSOFA(rr, sbp, avpu):
    rr, sbp = _num(rr), _num(sbp)
    avpu = "A" if avpu is None else str(avpu).strip().upper()
    hits, score = [], 0
    if rr is not None and rr >= 22: hits.append("RR ≥22"); score += 1
    if sbp is not None and sbp <= 100: hits.append("SBP ≤100"); score += 1
    if avpu != "A": hits.append("Altered mentation"); score += 1
    return score, hits, (score >= 2)

def calc_MEOWS(hr, rr, sbp, temp, spo2):
    hr, rr, sbp, temp, spo2 = _num(hr), _num(rr), _num(sbp), _num(temp), _num(spo2)
    red, yellow = [], []
    if sbp is not None:
        if sbp < 90 or sbp > 160: red.append("SBP critical")
        elif sbp < 100 or sbp > 150: yellow.append("SBP borderline")
    if hr is not None:
        if hr > 120 or hr < 50: red.append("HR critical")
        elif hr > 100: yellow.append("HR high")
    if rr is not None:
        if rr > 30 or rr < 10: red.append("RR critical")
        elif rr > 21: yellow.append("RR high")
    if temp is not None:
        if temp >= 38.0 or temp < 35.0: red.append("Temp critical")
        elif temp >= 37.6: yellow.append("Temp high")
    if spo2 is not None:
        if spo2 < 94: red.append("SpO₂ <94%")
        elif spo2 < 96: yellow.append("SpO₂ 94–95%")
    return {"red": red, "yellow": yellow}

def calc_PEWS(age, rr, hr, behavior="Normal", spo2=None):
    age, rr, hr, spo2 = _num(age), _num(rr), _num(hr), _num(spo2)
    if age is None: return 0, {"detail": "age missing"}, False, False

    # Age-based parameters
    if age < 1:         rr_y, rr_r = (40, 50), (50, 60); hr_y, hr_r = (140, 160), (160, 200)
    elif age < 5:       rr_y, rr_r = (30, 40), (40, 60); hr_y, hr_r = (130, 150), (150, 200)
    elif age < 12:      rr_y, rr_r = (24, 30), (30, 60); hr_y, hr_r = (120, 140), (140, 200)
    else:               rr_y, rr_r = (20, 24), (24, 60); hr_y, hr_r = (110, 130), (130, 200)

    sc = 0
    # RR scoring
    if rr is not None:
        if rr >= rr_r[1] or rr <= rr_r[0]: sc += 2
        elif rr >= rr_y[1] or rr <= rr_y[0]: sc += 1
    
    # HR scoring
    if hr is not None:
        if hr >= hr_r[1] or hr <= hr_r[0]: sc += 2
        elif hr >= hr_y[1] or hr <= hr_y[0]: sc += 1
    
    # SpO2 scoring
    if spo2 is not None:
        if spo2 < 92: sc += 2
        elif spo2 < 95: sc += 1

    # Behavior scoring
    beh = str(behavior or "Normal").lower()
    if beh == "lethargic": sc += 2
    elif beh == "irritable": sc += 1

    return sc, {"age": age}, (sc >= 6), (sc >= 4)

def triage_decision(vitals, context):
    v = validate_vitals(vitals.get("hr"), vitals.get("rr"), vitals.get("sbp"),
                        vitals.get("temp"), vitals.get("spo2"))
    avpu = vitals.get("avpu","A")
    reasons = []

    news2_score, news2_hits, news2_review, news2_urgent = safe_calc_NEWS2(
        v["rr"], v["spo2"], v["sbp"], v["hr"], v["temp"], avpu,
        context.get("o2_device", "Air"), context.get("spo2_scale", 1)
    )
    
    q_score, q_hits, q_high = (
        calc_qSOFA(v["rr"], v["sbp"], avpu) if context.get("infection") else (0, [], False)
    )
    
    meows = (
        calc_MEOWS(v["hr"], v["rr"], v["sbp"], v["temp"], v["spo2"])
        if context.get("pregnant") else {"red": [], "yellow": []}
    )
    
    pews_sc, pews_meta, pews_high, pews_watch = (
        calc_PEWS(context.get("age"), v["rr"], v["hr"], context.get("behavior","Normal"), v["spo2"])
        if (context.get("age") is not None and context.get("age") < 18)
        else (0, {}, False, False)
    )

    colour = "GREEN"
    
    # RED criteria
    if (news2_urgent or q_high or 
        (context.get("pregnant") and len(meows["red"]) > 0) or 
        (context.get("age") is not None and context.get("age") < 18 and pews_high)):
        colour = "RED"
    
    # YELLOW criteria
    elif colour == "GREEN" and (
        news2_review or 
        (context.get("pregnant") and len(meows["yellow"]) > 0) or 
        (context.get("age") is not None and context.get("age") < 18 and pews_watch)):
        colour = "YELLOW"

    # Build reasons
    if news2_urgent: reasons.append(f"NEWS2 {news2_score} (≥7)")
    elif news2_review: reasons.append(f"NEWS2 {news2_score} (≥5)")
    if q_high: reasons.append(f"qSOFA {q_score} (≥2)")
    if context.get("pregnant") and meows["red"]: reasons.append("MEOWS red band")
    if context.get("pregnant") and meows["yellow"] and colour == "YELLOW": reasons.append("MEOWS yellow band")
    if (context.get("age") is not None and context.get("age") < 18 and pews_high): 
        reasons.append(f"PEWS {pews_sc} (≥6)")
    if (context.get("age") is not None and context.get("age") < 18 and pews_watch and colour == "YELLOW"): 
        reasons.append(f"PEWS {pews_sc} (≥4)")

    details = {
        "NEWS2": {"score": news2_score, "review": news2_review, "urgent": news2_urgent},
        "qSOFA": {"score": q_score, "high": q_high},
        "MEOWS": meows,
        "PEWS": {"score": pews_sc, "high": pews_high, "watch": pews_watch},
        "reasons": reasons
    }
    return colour, details

# === DATA GENERATION ===
def generate_comprehensive_synthetic_data():
    """Generate synthetic medical data"""
    
    facilities = [
        "Tertiary Central Hospital", "District North General", 
        "Specialty South Medical", "Trauma East Center", "Community West Hospital"
    ]
    
    case_types = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]
    
    # Generate referred cases
    referred_cases = []
    base_time = time.time() - 30 * 24 * 3600
    
    for i in range(150):
        case_type = random.choice(case_types)
        age = random.randint(18, 80) if case_type != "Maternal" else random.randint(18, 40)
        
        # Generate medically appropriate vitals
        if case_type == "Maternal":
            vitals = {
                "hr": random.randint(90, 140), "sbp": random.randint(80, 160),
                "rr": random.randint(18, 30), "temp": round(random.uniform(36.5, 38.5), 1),
                "spo2": random.randint(92, 99), "avpu": "A"
            }
        elif case_type == "Trauma":
            vitals = {
                "hr": random.randint(70, 150), "sbp": random.randint(70, 180),
                "rr": random.randint(16, 35), "temp": round(random.uniform(36.0, 38.0), 1),
                "spo2": random.randint(88, 98), "avpu": random.choices(["A", "V", "P"], weights=[0.7, 0.2, 0.1])[0]
            }
        else:
            vitals = {
                "hr": random.randint(80, 140), "sbp": random.randint(90, 150),
                "rr": random.randint(18, 32), "temp": round(random.uniform(36.5, 39.0), 1),
                "spo2": random.randint(86, 95), "avpu": random.choices(["A", "V"], weights=[0.8, 0.2])[0]
            }
        
        # Calculate triage
        context = {
            "age": age,
            "pregnant": (case_type == "Maternal"),
            "infection": (case_type in ["Sepsis", "Other"]),
            "o2_device": "Air",
            "spo2_scale": 1,
            "behavior": "Normal"
        }
        triage_color, score_details = triage_decision(vitals, context)
        
        # Select ICD code
        matching_icd = [icd for icd in ICD_CATALOG if icd["case_type"] == case_type and icd["age_min"] <= age <= icd["age_max"]]
        icd = random.choice(matching_icd) if matching_icd else random.choice([icd for icd in ICD_CATALOG if icd["case_type"] == case_type])
        
        # Generate interventions
        interventions = random.sample(INTERVENTION_PROTOCOLS[case_type], random.randint(2, 4))
        
        referred_cases.append({
            "case_id": f"REF_{1000 + i}",
            "timestamp": datetime.fromtimestamp(base_time + random.randint(0, 30 * 24 * 3600)),
            "referring_facility": random.choice(["PHC Mawlai", "CHC Smit", "CHC Pynursla"]),
            "receiving_facility": random.choice(facilities),
            "patient_age": age,
            "patient_sex": "F" if case_type == "Maternal" else random.choice(["M", "F"]),
            "case_type": case_type,
            "icd_code": icd["icd_code"],
            "icd_label": icd["label"],
            "triage_color": triage_color,
            "vitals": vitals,
            "interventions_referring": interventions,
            "status": random.choices(["Accepted", "Rejected", "Pending"], weights=[0.8, 0.1, 0.1])[0],
        })
    
    # Generate received cases
    received_cases = []
    for i in range(150):
        base_case = random.choice(referred_cases)
        
        received_cases.append({
            **base_case,
            "case_id": f"REC_{2000 + i}",
            "transport_time_minutes": random.randint(20, 90),
            "interventions_receiving": base_case["interventions_referring"] + random.sample(
                INTERVENTION_PROTOCOLS[base_case["case_type"]], random.randint(1, 3)
            ),
            "final_outcome": random.choices(["Good", "Fair", "Poor"], weights=[0.7, 0.2, 0.1])[0],
            "length_of_stay_hours": random.randint(24, 240)
        })
    
    return {
        "referred_cases": pd.DataFrame(referred_cases),
        "received_cases": pd.DataFrame(received_cases),
        "facilities": facilities
    }

# === DASHBOARD COMPONENTS ===
def triaging_tool_interface():
    """Interactive triaging tool"""
    st.markdown("### 🏥 Real-Time Triaging Tool")
    
    with st.form("triage_assessment"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Details")
            age = st.number_input("Age", 0, 120, 35)
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            pregnant = st.checkbox("Pregnant") if sex == "Female" else False
            chief_complaint = st.selectbox("Chief Complaint", 
                                         ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"])
            
        with col2:
            st.subheader("Vital Signs")
            hr = st.number_input("Heart Rate", 20, 240, 80)
            sbp = st.number_input("SBP", 50, 260, 120)
            rr = st.number_input("Respiratory Rate", 5, 60, 16)
            temp = st.number_input("Temperature °C", 32.0, 42.0, 37.0, step=0.1)
            spo2 = st.number_input("SpO₂ %", 50, 100, 98)
            avpu = st.selectbox("AVPU", ["A", "V", "P", "U"])
        
        col3, col4 = st.columns(2)
        with col3:
            o2_device = st.selectbox("O₂ Device", ["Air", "O2"])
            spo2_scale = st.selectbox("SpO₂ Scale", [1, 2])
        with col4:
            infection_suspected = st.checkbox("Infection Suspected")
            behavior = st.selectbox("Behavior (PEWS)", ["Normal", "Irritable", "Lethargic"])
        
        if st.form_submit_button("Calculate Triage Score", type="primary"):
            vitals = {"hr": hr, "rr": rr, "sbp": sbp, "temp": temp, "spo2": spo2, "avpu": avpu}
            context = {
                "age": age,
                "pregnant": pregnant,
                "infection": infection_suspected,
                "o2_device": o2_device,
                "spo2_scale": spo2_scale,
                "behavior": behavior
            }
            
            triage_color, score_details = triage_decision(vitals, context)
            
            st.markdown(f"### Triage Decision: **{triage_color}**")
            
            if triage_color == "RED":
                st.error("🚨 CRITICAL: Immediate physician assessment required")
            elif triage_color == "YELLOW":
                st.warning("⚠️ URGENT: Assessment within 30 minutes")
            else:
                st.success("✅ STABLE: Routine assessment")
            
            with st.expander("Score Details"):
                st.write("**NEWS2 Score:**", score_details["NEWS2"])
                if infection_suspected:
                    st.write("**qSOFA Score:**", score_details["qSOFA"])
                if pregnant:
                    st.write("**MEOWS Assessment:**", score_details["MEOWS"])
                if age < 18:
                    st.write("**PEWS Score:**", score_details["PEWS"])
                st.write("**Reasons:**", ", ".join(score_details["reasons"]))

def intervention_dashboard(data, dashboard_mode):
    """Intervention management dashboard"""
    st.markdown("### 💊 Intervention Management")
    
    if dashboard_mode == "📤 Referring Center":
        st.info("**Referring Facility Interventions**")
        recent_cases = data["referred_cases"].tail(5)
        
        for _, case in recent_cases.iterrows():
            with st.expander(f"{case['case_id']} - {case['icd_label']} ({case['triage_color']})"):
                st.write("**Interventions Applied:**")
                for intervention in case['interventions_referring']:
                    st.write(f"✅ {intervention}")
    
    else:
        st.info("**Receiving Facility Interventions**")
        recent_cases = data["received_cases"].tail(5)
        
        for _, case in recent_cases.iterrows():
            with st.expander(f"{case['case_id']} - {case['icd_label']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Referring Interventions:**")
                    for iv in case['interventions_referring']:
                        st.write(f"• {iv}")
                with col2:
                    st.write("**Receiving Interventions:**")
                    for iv in case['interventions_receiving']:
                        st.write(f"• {iv}")
                st.metric("Outcome", case['final_outcome'])

def enhanced_analytical_tools(data):
    """Advanced analytics dashboard"""
    st.markdown("### 📊 Advanced Medical Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Triage distribution
        triage_counts = data["referred_cases"]["triage_color"].value_counts()
        fig = px.pie(values=triage_counts.values, names=triage_counts.index,
                    color=triage_counts.index,
                    color_discrete_map={'RED': 'red', 'YELLOW': 'yellow', 'GREEN': 'green'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Case type distribution
        case_counts = data["referred_cases"]["case_type"].value_counts()
        fig = px.bar(x=case_counts.values, y=case_counts.index, orientation='h',
                    title="Cases by Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", len(data["referred_cases"]))
    col2.metric("Critical Cases", len(data["referred_cases"][data["referred_cases"]["triage_color"] == "RED"]))
    col3.metric("Avg Transport Time", f"{data['received_cases']['transport_time_minutes'].mean():.1f} min")

# === MAIN APP ===
def main():
    st.title("🏥 AHECN Hospital Dashboard MVP")
    st.markdown("### Complete Medical Dashboard with Triaging & Analytics")
    
    # Initialize session state
    if 'medical_data' not in st.session_state:
        st.session_state.medical_data = generate_comprehensive_synthetic_data()
    
    # Dashboard selector
    dashboard_mode = st.radio(
        "**Dashboard Mode:**",
        ["🏥 Receiving Center", "📤 Referring Center", "🔬 Triaging Tool", "📊 Analytics"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if dashboard_mode == "🔬 Triaging Tool":
        triaging_tool_interface()
        
    elif dashboard_mode == "📊 Analytics":
        enhanced_analytical_tools(st.session_state.medical_data)
        
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if dashboard_mode == "🏥 Receiving Center":
                render_receiving_center()
            else:
                render_referring_center()
        
        with col2:
            intervention_dashboard(st.session_state.medical_data, dashboard_mode)

def render_receiving_center():
    """Receiving center view"""
    st.info("**Receiving Center Dashboard**")
    
    # Capacity metrics
    st.subheader("Current Capacity")
    cap_cols = st.columns(4)
    cap_cols[0].metric("ICU Beds", "8/12", "67%")
    cap_cols[1].metric("ED Beds", "15/25", "60%")
    cap_cols[2].metric("Ventilators", "4/8", "50%")
    cap_cols[3].metric("Specialists", "3/5", "-2")
    
    # Incoming cases
    st.subheader("Incoming Cases")
    incoming = st.session_state.medical_data["referred_cases"].tail(5)
    
    for _, case in incoming.iterrows():
        triage_class = f"triage-{case['triage_color'].lower()}"
        st.markdown(f"""
        <div class="card {triage_class}">
            <strong>{case['case_id']}</strong> | {case['icd_label']} | {case['triage_color']}<br>
            From: {case['referring_facility']} | Age: {case['patient_age']}{case['patient_sex']}<br>
            Vitals: HR {case['vitals']['hr']} | SBP {case['vitals']['sbp']} | SpO2 {case['vitals']['spo2']}%
        </div>
        """, unsafe_allow_html=True)

def render_referring_center():
    """Referring center view"""
    st.info("**Referring Center Dashboard**")
    
    # Quick referral
    st.subheader("Quick Referral")
    with st.form("quick_referral"):
        col1, col2 = st.columns(2)
        with col1:
            case_type = st.selectbox("Case Type", ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"])
            age = st.number_input("Patient Age", 0, 120, 35)
        with col2:
            hr = st.number_input("Heart Rate", 20, 240, 80)
            sbp = st.number_input("SBP", 50, 260, 120)
        
        if st.form_submit_button("🚀 Find Optimal Facility"):
            vitals = {"hr": hr, "sbp": sbp, "rr": 16, "temp": 37.0, "spo2": 98, "avpu": "A"}
            context = {"age": age, "pregnant": False, "infection": False}
            triage_color, _ = triage_decision(vitals, context)
            st.success(f"Recommended: Tertiary Central Hospital | Triage: {triage_color}")

# Sidebar
def render_sidebar():
    st.sidebar.title("AHECN Dashboard")
    st.sidebar.markdown("---")
    
    st.sidebar.info("**System Status**")
    st.sidebar.metric("Data Integrity", "100%")
    st.sidebar.metric("Uptime", "99.8%")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.medical_data = generate_comprehensive_synthetic_data()
        st.rerun()

if __name__ == "__main__":
    main()
    render_sidebar()
