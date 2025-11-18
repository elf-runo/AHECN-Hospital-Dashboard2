# hospital_dashboard_premium.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta
import time
import random
import math
import json
from streamlit_calendar import calendar
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic

# === PAGE CONFIG ===
st.set_page_config(
    page_title="AHECN Hospital Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# === PREMIUM CSS STYLING ===
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
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.premium-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
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

.case-card:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
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

.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 0.5rem;
}

.status-active { background: var(--success); }
.status-pending { background: var(--warning); }
.status-completed { background: var(--accent); }

.calendar-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# === ENHANCED MEDICAL DATA CATALOGS ===
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

# === ENHANCED DATA GENERATION ===
def generate_premium_synthetic_data(days_back=30):
    """Generate comprehensive synthetic data with realistic timelines"""
    
    facilities = [
        {"name": "Tertiary Central Hospital", "type": "Tertiary", "lat": 25.578, "lon": 91.893, "beds": 500},
        {"name": "District North General", "type": "District", "lat": 25.591, "lon": 91.878, "beds": 200},
        {"name": "Specialty South Medical", "type": "Specialty", "lat": 25.565, "lon": 91.901, "beds": 150},
        {"name": "Trauma East Center", "type": "Trauma", "lat": 25.572, "lon": 91.885, "beds": 300},
    ]
    
    case_types = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]
    
    referred_cases = []
    received_cases = []
    
    base_time = datetime.now() - timedelta(days=days_back)
    
    for day in range(days_back):
        daily_cases = random.randint(3, 8)
        for case_num in range(daily_cases):
            case_time = base_time + timedelta(days=day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            case_type = random.choice(case_types)
            age = random.randint(18, 80) if case_type != "Maternal" else random.randint(18, 40)
            
            # Generate realistic vitals
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
            
            # Select ICD code
            matching_icd = [icd for icd in ICD_CATALOG if icd["case_type"] == case_type and icd["age_min"] <= age <= icd["age_max"]]
            icd = random.choice(matching_icd) if matching_icd else random.choice([icd for icd in ICD_CATALOG if icd["case_type"] == case_type])
            
            # Generate interventions
            interventions = random.sample(INTERVENTION_PROTOCOLS[case_type], random.randint(2, 4))
            
            # Select facilities
            referring_facility = random.choice(["PHC Mawlai", "CHC Smit", "CHC Pynursla", "Rural Health Center"])
            receiving_facility = random.choice(facilities)
            emt_crew = random.choice(EMT_CREW)
            
            # Calculate transport details
            transport_time = random.randint(20, 90)
            route_coordinates = generate_route_coordinates()
            
            case_id = f"REF_{case_time.strftime('%Y%m%d')}_{case_num:03d}"
            
            referred_case = {
                "case_id": case_id,
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
                "case_id": case_id.replace("REF", "REC"),
                "transport_time_minutes": transport_time,
                "emt_crew": emt_crew,
                "vehicle_id": emt_crew["vehicle"],
                "route_coordinates": route_coordinates,
                "interventions_receiving": interventions + random.sample(INTERVENTION_PROTOCOLS[case_type], random.randint(1, 3)),
                "final_outcome": random.choices(["Excellent", "Good", "Fair", "Poor"], weights=[0.4, 0.3, 0.2, 0.1])[0],
                "length_of_stay_hours": random.randint(24, 240),
                "discharge_date": case_time + timedelta(hours=random.randint(24, 240))
            }
            
            referred_cases.append(referred_case)
            received_cases.append(received_case)
    
    return {
        "referred_cases": pd.DataFrame(referred_cases),
        "received_cases": pd.DataFrame(received_cases),
        "facilities": facilities,
        "emt_crews": EMT_CREW
    }

def generate_route_coordinates():
    """Generate realistic route coordinates"""
    base_lat, base_lon = 25.578, 91.893
    coordinates = []
    for i in range(10):
        lat = base_lat + random.uniform(-0.02, 0.02)
        lon = base_lon + random.uniform(-0.02, 0.02)
        coordinates.append([lat, lon])
    return coordinates

# === PREMIUM DASHBOARD COMPONENTS ===
def render_premium_header():
    """Render premium header with key metrics"""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.5rem;">🏥 AHECN Hospital Command Center</h1>
        <p style="margin:0; font-size:1.2rem; opacity:0.9;">Advanced Emergency Care Coordination Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-highlight">
            <div style="font-size:2rem; font-weight:bold;">47</div>
            <div>Active Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-highlight">
            <div style="font-size:2rem; font-weight:bold;">12</div>
            <div>Critical RED</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-highlight">
            <div style="font-size:2rem; font-weight:bold;">28</div>
            <div>Urgent YELLOW</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-highlight">
            <div style="font-size:2rem; font-weight:bold;">92%</div>
            <div>Acceptance Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-highlight">
            <div style="font-size:2rem; font-weight:bold;">34m</div>
            <div>Avg Response</div>
        </div>
        """, unsafe_allow_html=True)

def render_case_calendar(data):
    """Interactive calendar for case management"""
    st.markdown("### 📅 Case Calendar & Timeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Date range selector
        date_range = st.date_input(
            "Select Date Range",
            [datetime.now().date() - timedelta(days=7), datetime.now().date()],
            key="date_range_selector"
        )
        
        # Filter cases by date range
        filtered_referred = data["referred_cases"][
            (data["referred_cases"]["timestamp"].dt.date >= date_range[0]) &
            (data["referred_cases"]["timestamp"].dt.date <= date_range[1])
        ]
        
        # Case timeline visualization
        timeline_data = filtered_referred.groupby(filtered_referred["timestamp"].dt.date).size().reset_index(name='count')
        fig = px.line(timeline_data, x='timestamp', y='count', 
                     title="Daily Case Volume Trend",
                     markers=True)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Quick Filters")
        
        # Case type filter
        case_types = st.multiselect(
            "Case Types",
            ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"],
            default=["Maternal", "Trauma", "Cardiac"]
        )
        
        # Triage filter
        triage_levels = st.multiselect(
            "Triage Levels",
            ["RED", "YELLOW", "GREEN"],
            default=["RED", "YELLOW"]
        )
        
        # Facility filter
        facilities = st.multiselect(
            "Receiving Facilities",
            ["Tertiary Central Hospital", "District North General", "Specialty South Medical", "Trauma East Center"],
            default=["Tertiary Central Hospital"]
        )
        
        if st.button("Apply Filters", type="primary"):
            st.success(f"Filtered {len(filtered_referred)} cases")

def render_interactive_case_list(data, case_type="referred"):
    """Interactive case list with detailed views"""
    st.markdown(f"### 📋 {'Referred' if case_type == 'referred' else 'Received'} Cases")
    
    cases_df = data[f"{case_type}_cases"]
    
    # Search and filter
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search cases...", placeholder="Search by case ID, facility, or diagnosis")
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Timestamp", "Triage", "Case Type"])
    
    with col3:
        items_per_page = st.selectbox("Items per page", [10, 25, 50], index=0)
    
    # Filter cases based on search
    if search_term:
        filtered_cases = cases_df[
            cases_df["case_id"].str.contains(search_term, case=False) |
            cases_df["referring_facility"].str.contains(search_term, case=False) |
            cases_df["icd_label"].str.contains(search_term, case=False)
        ]
    else:
        filtered_cases = cases_df.copy()
    
    # Sort cases
    if sort_by == "Timestamp":
        filtered_cases = filtered_cases.sort_values("timestamp", ascending=False)
    elif sort_by == "Triage":
        filtered_cases = filtered_cases.sort_values("triage_color", ascending=True)
    else:
        filtered_cases = filtered_cases.sort_values("case_type", ascending=True)
    
    # Pagination
    total_pages = max(1, len(filtered_cases) // items_per_page)
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_cases = filtered_cases.iloc[start_idx:end_idx]
    
    # Display cases
    for idx, case in paginated_cases.iterrows():
        case_class = f"case-card {case['triage_color'].lower()}"
        
        with st.container():
            st.markdown(f"""
            <div class="{case_class}" onclick="selectCase('{case['case_id']}')">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{case['case_id']}</strong> • {case['icd_label']}
                    </div>
                    <div style="background: {'#ff4444' if case['triage_color'] == 'RED' else '#ffaa00' if case['triage_color'] == 'YELLOW' else '#00c853'}; 
                                color: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                        {case['triage_color']}
                    </div>
                </div>
                <div style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
                    {case['referring_facility']} → {case['receiving_facility']} • 
                    {case['timestamp'].strftime('%Y-%m-%d %H:%M')} • 
                    {case['patient_age']}{case['patient_sex']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Case details expander
            with st.expander(f"View full details for {case['case_id']}", expanded=False):
                render_case_details(case, case_type)

def render_case_details(case, case_type):
    """Render detailed case information"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Patient Information")
        st.write(f"**Case ID:** {case['case_id']}")
        st.write(f"**Age/Sex:** {case['patient_age']}{case['patient_sex']}")
        st.write(f"**Diagnosis:** {case['icd_label']} ({case['icd_code']})")
        st.write(f"**Triage:** {case['triage_color']}")
        st.write(f"**Referring Facility:** {case['referring_facility']}")
        st.write(f"**Receiving Facility:** {case['receiving_facility']}")
        
        st.markdown("#### Vital Signs")
        vitals = case['vitals']
        st.write(f"**HR:** {vitals['hr']} bpm | **SBP:** {vitals['sbp']} mmHg")
        st.write(f"**RR:** {vitals['rr']} rpm | **SpO2:** {vitals['spo2']}%")
        st.write(f"**Temp:** {vitals['temp']}°C | **AVPU:** {vitals['avpu']}")
    
    with col2:
        st.markdown("#### Timeline & Interventions")
        
        # Interventions timeline
        st.markdown("**Referring Interventions:**")
        for intervention in case['interventions_referring']:
            st.markdown(f'<div class="intervention-badge">{intervention}</div>', unsafe_allow_html=True)
        
        if case_type == "received":
            st.markdown("**Receiving Interventions:**")
            for intervention in case['interventions_receiving']:
                st.markdown(f'<div class="intervention-badge">{intervention}</div>', unsafe_allow_html=True)
            
            st.markdown("#### Transport Details")
            st.write(f"**Transport Time:** {case['transport_time_minutes']} minutes")
            st.write(f"**EMT Crew:** {case['emt_crew']['name']} ({case['emt_crew']['level']})")
            st.write(f"**Vehicle:** {case['vehicle_id']}")
            st.write(f"**Outcome:** {case['final_outcome']}")
            st.write(f"**Length of Stay:** {case['length_of_stay_hours']} hours")
            
            # Route visualization
            if case.get('route_coordinates'):
                st.markdown("#### Transport Route")
                render_route_map(case['route_coordinates'])

def render_route_map(coordinates):
    """Render interactive route map"""
    if coordinates:
        m = folium.Map(location=coordinates[0], zoom_start=12)
        
        # Add route line
        folium.PolyLine(
            coordinates,
            weight=5,
            color='blue',
            opacity=0.7
        ).add_to(m)
        
        # Add start and end markers
        folium.Marker(
            coordinates[0],
            popup='Pickup Location',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        folium.Marker(
            coordinates[-1],
            popup='Destination Hospital',
            icon=folium.Icon(color='red', icon='hospital-o')
        ).add_to(m)
        
        folium_static(m, width=400, height=300)

def render_advanced_analytics(data):
    """Premium analytics dashboard"""
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(data["referred_cases"])
        st.metric("Total Cases", total_cases)
    
    with col2:
        acceptance_rate = (len(data["referred_cases"][data["referred_cases"]["status"] == "Accepted"]) / total_cases) * 100
        st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
    
    with col3:
        avg_transport = data["received_cases"]["transport_time_minutes"].mean()
        st.metric("Avg Transport Time", f"{avg_transport:.1f} min")
    
    with col4:
        critical_cases = len(data["referred_cases"][data["referred_cases"]["triage_color"] == "RED"])
        st.metric("Critical Cases", critical_cases)
    
    # Advanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Case type distribution with outcomes
        st.markdown("#### Case Type Analysis")
        case_outcomes = data["received_cases"].groupby(['case_type', 'final_outcome']).size().unstack(fill_value=0)
        fig = px.bar(case_outcomes, barmode='stack', title="Case Outcomes by Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Triage efficiency
        st.markdown("#### Triage Efficiency")
        triage_times = data["received_cases"].groupby('triage_color')['transport_time_minutes'].mean().reset_index()
        fig = px.bar(triage_times, x='triage_color', y='transport_time_minutes', 
                    color='triage_color', title="Average Transport Time by Triage")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("#### Performance Trends")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Weekly trends
        weekly_data = data["referred_cases"].copy()
        weekly_data['week'] = weekly_data['timestamp'].dt.isocalendar().week
        weekly_trends = weekly_data.groupby('week').size()
        fig = px.line(x=weekly_trends.index, y=weekly_trends.values, title="Weekly Case Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Facility performance
        facility_performance = data["referred_cases"]["receiving_facility"].value_counts()
        fig = px.pie(values=facility_performance.values, names=facility_performance.index, 
                    title="Case Distribution by Facility")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Outcome analysis
        outcome_dist = data["received_cases"]["final_outcome"].value_counts()
        fig = px.bar(x=outcome_dist.values, y=outcome_dist.index, orientation='h',
                    title="Patient Outcomes Distribution")
        st.plotly_chart(fig, use_container_width=True)

def render_emt_tracking(data):
    """Real-time EMT and ambulance tracking"""
    st.markdown("### 🚑 Real-time EMT Tracking")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Active transports
        st.markdown("#### Active Transports")
        active_transports = data["received_cases"].tail(3)  # Simulate active transports
        
        for _, transport in active_transports.iterrows():
            progress = random.randint(30, 90)
            
            st.markdown(f"""
            <div class="premium-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{transport['case_id']}</strong><br>
                        <small>{transport['emt_crew']['name']} • {transport['vehicle_id']}</small>
                    </div>
                    <div style="background: #e3f2fd; padding: 0.5rem 1rem; border-radius: 10px;">
                        {progress}% Complete
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <div style="background: #f0f0f0; border-radius: 10px; height: 10px;">
                        <div style="background: #2196f3; width: {progress}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                    ETA: {transport['transport_time_minutes']} min • {transport['referring_facility']} → {transport['receiving_facility']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # EMT crew status
        st.markdown("#### Crew Status")
        for crew in data["emt_crews"]:
            status_color = "🟢" if crew["status"] == "active" else "🟡" if crew["status"] == "available" else "🔴"
            st.write(f"{status_color} **{crew['name']}**")
            st.write(f"   {crew['level']} • {crew['vehicle']}")
            st.write("---")

# === MAIN PREMIUM DASHBOARD ===
def main():
    # Initialize session state
    if 'premium_data' not in st.session_state:
        st.session_state.premium_data = generate_premium_synthetic_data(days_back=60)
    
    # Render premium header
    render_premium_header()
    
    # Main navigation tabs
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
        render_interactive_case_list(st.session_state.premium_data, "referred")
    
    with tab3:
        render_interactive_case_list(st.session_state.premium_data, "received")
    
    with tab4:
        render_advanced_analytics(st.session_state.premium_data)
    
    with tab5:
        render_emt_tracking(st.session_state.premium_data)
    
    with tab6:
        render_quick_actions()
    
    # Sidebar
    render_premium_sidebar()

def render_dashboard_overview():
    """Main dashboard overview"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Case calendar and timeline
        render_case_calendar(st.session_state.premium_data)
        
        # Recent activity
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
        # System status
        st.markdown("### 🖥️ System Status")
        
        status_items = [
            {"name": "Database", "status": "Optimal", "color": "success"},
            {"name": "API Services", "status": "Stable", "color": "success"},
            {"name": "EMT Tracking", "status": "Active", "color": "success"},
            {"name": "Analytics", "status": "Processing", "color": "warning"},
        ]
        
        for item in status_items:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                <span>{item['name']}</span>
                <span style="background: {'#00c853' if item['color'] == 'success' else '#ffab00'}; 
                            color: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                    {item['status']}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("### 📋 Quick Stats")
        stats = st.session_state.premium_data["referred_cases"]
        
        st.metric("Cases Today", len(stats[stats["timestamp"].dt.date == datetime.now().date()]))
        st.metric("Cases This Week", len(stats[stats["timestamp"].dt.isocalendar().week == datetime.now().isocalendar()[1]]))
        st.metric("Critical Pending", len(stats[(stats["triage_color"] == "RED") & (stats["status"] == "Pending")]))

def render_quick_actions():
    """Quick action buttons for common tasks"""
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.session_state.premium_data = generate_premium_synthetic_data(days_back=60)
            st.success("Data refreshed successfully!")
        
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Generating comprehensive report...")
    
    with col2:
        if st.button("🚨 Emergency Mode", use_container_width=True, type="secondary"):
            st.warning("Emergency mode activated - prioritizing critical cases")
        
        if st.button("📋 Case Summary", use_container_width=True):
            st.info("Displaying case summary...")
    
    with col3:
        if st.button("📧 Notify Staff", use_container_width=True):
            st.success("Staff notification sent!")
        
        if st.button("🖨️ Export Data", use_container_width=True):
            st.info("Preparing data export...")
    
    # Quick referral form
    st.markdown("### 🏃 Quick Referral")
    with st.form("quick_referral_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Patient Name")
            age = st.number_input("Age", 1, 120, 35)
            chief_complaint = st.selectbox("Chief Complaint", ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"])
        
        with col2:
            hr = st.number_input("Heart Rate", 20, 240, 80)
            sbp = st.number_input("SBP", 50, 260, 120)
            spo2 = st.number_input("SpO₂", 50, 100, 98)
        
        clinical_notes = st.text_area("Clinical Notes")
        
        if st.form_submit_button("🚀 Create Emergency Referral", type="primary"):
            st.success(f"Emergency referral created for {patient_name}!")

def render_premium_sidebar():
    """Premium sidebar with additional features"""
    st.sidebar.markdown("### 🔔 Live Alerts")
    
    # Critical alerts
    critical_cases = st.session_state.premium_data["referred_cases"][
        st.session_state.premium_data["referred_cases"]["triage_color"] == "RED"
    ].tail(3)
    
    for _, case in critical_cases.iterrows():
        st.sidebar.error(f"""
        **{case['case_id']}**
        {case['icd_label']}
        *{case['timestamp'].strftime('%H:%M')} • {case['referring_facility']}*
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Performance")
    
    # Performance metrics
    metrics = st.session_state.premium_data["referred_cases"]
    st.sidebar.metric("Response Rate", "94%", "2%")
    st.sidebar.metric("Avg Decision Time", "8.2min", "-1.3min")
    st.sidebar.metric("System Uptime", "99.9%", "0%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Tools")
    
    if st.sidebar.button("🔄 Force Refresh", key="sidebar_refresh"):
        st.session_state.premium_data = generate_premium_synthetic_data(days_back=60)
        st.rerun()
    
    if st.sidebar.button("📋 Data Summary", key="sidebar_summary"):
        st.info("Displaying data summary...")

if __name__ == "__main__":
    main()
