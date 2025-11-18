# data/synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import math
from data.icd_catalog import ICD_CATALOG, INTERVENTION_PROTOCOLS, EMT_INTERVENTIONS
from components.triaging import triage_decision

def generate_comprehensive_synthetic_data():
    """Generate 150+ synthetic cases with full medical details"""
    
    facilities = [
        "Tertiary Central Hospital", "District North General", 
        "Specialty South Medical", "Trauma East Center", "Community West Hospital"
    ]
    
    case_types = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]
    
    # Generate 150 referred cases
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
            "score_details": score_details,
            "interventions_referring": interventions,
            "status": random.choices(["Accepted", "Rejected", "Diverted"], weights=[0.8, 0.1, 0.1])[0],
            "outcome": random.choices(["Good", "Fair", "Poor"], weights=[0.7, 0.2, 0.1])[0] if i > 50 else "Pending"
        })
    
    # Generate 150 received cases with EMT data
    received_cases = []
    for i in range(150):
        base_case = random.choice(referred_cases)
        
        # Generate EMT interventions
        emt_updates = []
        transport_time = random.randint(20, 90)
        
        for minute in range(0, transport_time, 15):
            if minute > 0 and random.random() < 0.3:
                intervention_type = random.choice(list(EMT_INTERVENTIONS.keys()))
                intervention = random.choice(EMT_INTERVENTIONS[intervention_type])
                
                emt_updates.append({
                    "minute": minute,
                    "intervention": f"{intervention_type}: {intervention}",
                    "vitals_change": {
                        "hr": random.randint(-10, 10),
                        "sbp": random.randint(-5, 15),
                        "spo2": random.randint(0, 5)
                    },
                    "notes": f"Patient responded well to {intervention.lower()}"
                })
        
        received_cases.append({
            **base_case,
            "case_id": f"REC_{2000 + i}",
            "timestamp": datetime.fromtimestamp(base_time + random.randint(0, 30 * 24 * 3600)),
            "transport_time_minutes": transport_time,
            "emt_updates": emt_updates,
            "interventions_receiving": base_case["interventions_referring"] + random.sample(
                INTERVENTION_PROTOCOLS[base_case["case_type"]], random.randint(1, 3)
            ),
            "final_outcome": base_case["outcome"],
            "length_of_stay_hours": random.randint(24, 240) if base_case["outcome"] != "Poor" else random.randint(6, 48)
        })
    
    return {
        "referred_cases": pd.DataFrame(referred_cases),
        "received_cases": pd.DataFrame(received_cases),
        "facilities": facilities
    }
