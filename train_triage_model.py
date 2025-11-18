# train_triage_model.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from sklearn.ensemble import RandomForestClassifier
import joblib

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

def generate_synthetic_for_training(days_back=90):
    case_types = ["Maternal", "Trauma", "Stroke", "Cardiac", "Sepsis", "Other"]
    referred_cases = []
    base_time = datetime.now() - timedelta(days=days_back)

    random.seed(42)

    for day in range(days_back):
        daily_cases = random.randint(5, 12)
        for case_num in range(daily_cases):
            case_time = base_time + timedelta(days=day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            case_type = random.choice(case_types)
            age = random.randint(18, 80) if case_type != "Maternal" else random.randint(18, 40)

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
                    "spo2": random.randint(88, 98), "avpu": random.choice(["A", "V", "P"])
                }
            else:
                vitals = {
                    "hr": random.randint(80, 140), "sbp": random.randint(90, 150),
                    "rr": random.randint(18, 32), "temp": round(random.uniform(36.5, 39.0), 1),
                    "spo2": random.randint(86, 95), "avpu": random.choice(["A", "V"])
                }

            triage_color = random.choices(["RED", "YELLOW", "GREEN"], weights=[0.3, 0.5, 0.2])[0]

            referred_cases.append({
                "case_id": f"REF_{case_time.strftime('%Y%m%d')}_{case_num:03d}",
                "patient_age": age,
                "triage_color": triage_color,
                "vitals": vitals,
            })

    return pd.DataFrame(referred_cases)

def main():
    df = generate_synthetic_for_training(days_back=90)

    X = []
    y = []

    for _, row in df.iterrows():
        v = row["vitals"]
        X.append([
            float(row["patient_age"]),
            float(v["sbp"]),
            float(v["spo2"]),
            float(v["hr"]),
        ])
        y.append(row["triage_color"])

    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    out_path = os.path.join(os.path.dirname(__file__), "my_model.pkl")
    joblib.dump(clf, out_path)
    print(f"Model trained and saved to {out_path}")
    print("Example prediction:", clf.predict([[40, 120, 98, 80]]))

if __name__ == "__main__":
    main()
