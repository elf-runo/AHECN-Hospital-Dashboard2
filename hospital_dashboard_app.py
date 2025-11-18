# app.py - Main Streamlit application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import math
import json

# Import custom modules
from data.synthetic_data import generate_comprehensive_synthetic_data
from components.triaging import triaging_tool_interface, triage_decision
from components.interventions import intervention_dashboard, emt_real_time_simulation
from components.analytics import enhanced_analytical_tools
from utils.styling import load_css

# Initialize session state
def init_session_state():
    if 'medical_data' not in st.session_state:
        st.session_state.medical_data = generate_comprehensive_synthetic_data()
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0

# Main application
def main():
    # Load CSS and initialize
    load_css()
    init_session_state()
    
    st.title("🏥 AHECN Hospital Dashboard MVP")
    st.markdown("### Intelligent Dashboard for Emergency Care Coordination")
    
    # Dashboard mode selector
    dashboard_mode = st.radio(
        "**Dashboard Mode:**",
        ["🏥 Receiving Center", "📤 Referring Center", "🔬 Triaging Tool", "📊 Analytics", "🚑 EMT Simulation"],
        horizontal=True,
        key="dashboard_mode"
    )
    
    st.markdown("---")
    
    # Route to appropriate dashboard component
    if dashboard_mode == "🔬 Triaging Tool":
        triaging_tool_interface()
        
    elif dashboard_mode == "📊 Analytics":
        enhanced_analytical_tools(st.session_state.medical_data)
        
    elif dashboard_mode == "🚑 EMT Simulation":
        emt_real_time_simulation()
        
    else:
        render_main_dashboard(dashboard_mode)

def render_main_dashboard(mode):
    """Render the main dashboard based on mode"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if mode == "🏥 Receiving Center":
            render_receiving_center()
        else:
            render_referring_center()
    
    with col2:
        intervention_dashboard(st.session_state.medical_data, mode)
        
        # Refresh button
        if st.button("🔄 Refresh Data", key="refresh_main"):
            st.session_state.medical_data = generate_comprehensive_synthetic_data()
            st.session_state.refresh_count += 1
            st.rerun()

def render_receiving_center():
    """Receiving center dashboard"""
    st.info("**Receiving Center Dashboard**")
    
    # Capacity overview
    st.subheader("Current Capacity")
    cap_cols = st.columns(4)
    cap_cols[0].metric("ICU Beds", "8/12", "67%", delta_color="off")
    cap_cols[1].metric("ED Beds", "15/25", "60%", delta_color="off")
    cap_cols[2].metric("Ventilators", "4/8", "50%", delta_color="off")
    cap_cols[3].metric("Specialists", "3/5", "-2", delta_color="inverse")
    
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
    """Referring center dashboard"""
    st.info("**Referring Center Dashboard**")
    
    # Quick referral form
    st.subheader("Quick Referral")
    with st.form("quick_referral"):
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            case_type = st.selectbox("Case Type", ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"])
            age = st.number_input("Patient Age", 0, 120, 35)
        with ref_col2:
            hr = st.number_input("Heart Rate", 20, 240, 80)
            sbp = st.number_input("SBP", 50, 260, 120)
        
        if st.form_submit_button("🚀 Find Optimal Facility"):
            vitals = {"hr": hr, "sbp": sbp, "rr": 16, "temp": 37.0, "spo2": 98, "avpu": "A"}
            context = {"age": age, "pregnant": False, "infection": False}
            triage_color, _ = triage_decision(vitals, context)
            st.success(f"Recommended: Tertiary Central Hospital | Triage: {triage_color}")

# Sidebar component
def render_sidebar():
    """Render the sidebar with alerts and metrics"""
    st.sidebar.markdown("### 🔴 Medical Alerts")
    
    critical_cases = st.session_state.medical_data["referred_cases"][
        st.session_state.medical_data["referred_cases"]["triage_color"] == "RED"
    ].tail(3)
    
    for _, case in critical_cases.iterrows():
        st.sidebar.error(f"""
        **{case['case_id']}** - {case['icd_label']}
        {case['triage_color']} | HR: {case['vitals']['hr']} | SBP: {case['vitals']['sbp']}
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 System Status")
    st.sidebar.metric("Data Integrity", "100%", "No issues")
    st.sidebar.metric("Response Time", "2.3s", "Optimal")
    st.sidebar.metric("Uptime", "99.8%", "Stable")
    st.sidebar.metric("Data Refresh", st.session_state.refresh_count)

if __name__ == "__main__":
    main()
    render_sidebar()
