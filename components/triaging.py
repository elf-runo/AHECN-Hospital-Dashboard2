# components/triaging.py
import streamlit as st
from components.helpers import (
    validate_vitals,
    safe_calc_NEWS2,
    calc_qSOFA,
    calc_MEOWS,
    calc_PEWS
)

def triage_decision(vitals, context):
    """Enhanced triaging algorithm"""
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
        "NEWS2": {"score": news2_score, "hits": news2_hits, "review": news2_review, "urgent": news2_urgent},
        "qSOFA": {"score": q_score, "hits": q_hits, "high": q_high},
        "MEOWS": meows,
        "PEWS": {"score": pews_sc, "high": pews_high, "watch": pews_watch},
        "reasons": reasons
    }
    return colour, details

def triaging_tool_interface():
    """Interactive triaging tool interface"""
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
            
            # Display results
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
