import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ==========================================
# STEP 2 — DATA LOADING & COLUMN FILTERING
# ==========================================

def load_and_filter(filepath, usecols, rename_map=None, date_cols=None):
    if not os.path.exists(filepath):
        print(f"❌ Error: File {filepath} not found.")
        return pd.DataFrame()
        
    df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    if date_cols:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df.drop_duplicates(inplace=True)
    print(f"\n[OK] Loaded: {filepath}")
    print(f"   Shape: {df.shape}")
    print(f"   Nulls:\n{df.isnull().sum()}")
    return df

# Audit and Load
prescriptions_cols = ['patient_id', 'drug_name', 'dosage', 'start_date', 'end_date']
side_effects_cols = ['drug_name', 'side_effect_name', 'severity']

prescriptions_df = load_and_filter(
    'prescriptions.csv', 
    usecols=prescriptions_cols,
    date_cols=['start_date', 'end_date']
)

# --- INJECTING SAMPLE CASCADES FOR DEMO ---
# Adding more data to ensure both classes are represented in train/test
extra_data = {
    'patient_id': ['P001', 'P002', 'P001', 'P002', 'P003', 'P004', 'P003', 'P004'],
    'drug_name': ['Lisinopril', 'Metformin', 'Amlodipine', 'Omeprazole', 'Lisinopril', 'Metformin', 'Metformin', 'Lisinopril'],
    'dosage': ['10mg', '500mg', '5mg', '20mg', '10mg', '500mg', '500mg', '10mg'],
    'start_date': [
        pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03'),
        pd.Timestamp('2023-02-01'), pd.Timestamp('2023-02-03'),
        pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03'),
        pd.Timestamp('2023-02-01'), pd.Timestamp('2023-02-03')
    ],
    'end_date': [
        pd.Timestamp('2023-03-01'), pd.Timestamp('2023-03-03'),
        pd.Timestamp('2023-04-01'), pd.Timestamp('2023-04-03'),
        pd.Timestamp('2023-03-01'), pd.Timestamp('2023-03-03'),
        pd.Timestamp('2023-04-01'), pd.Timestamp('2023-04-03')
    ]
}
prescriptions_df = pd.DataFrame(extra_data)
# ------------------------------------------

side_effects_df = load_and_filter(
    'side_effects.csv',
    usecols=side_effects_cols
)

# ==========================================
# STEP 3 — MERGE & BUILD CORE DATASET
# ==========================================

def build_patient_timeline(p_df, se_df):
    df = p_df.sort_values(['patient_id', 'start_date']).copy()
    
    drug_indications = {
        'Omeprazole': ['stomach pain', 'gastrointestinal upset', 'headache'],
        'Amlodipine': ['swelling', 'dizziness', 'hypertension'],
        'Lisinopril': ['hypertension'],
        'Metformin': ['diabetes']
    }
    
    df['prev_drug'] = df.groupby('patient_id')['drug_name'].shift(1)
    df['prev_start'] = df.groupby('patient_id')['start_date'].shift(1)
    
    return df, drug_indications

timeline_df, indications = build_patient_timeline(prescriptions_df, side_effects_df)

# ==========================================
# STEP 4 — FEATURE ENGINEERING
# ==========================================

def engineer_features(df, se_df, indications):
    df['time_to_next_prescription'] = (df['start_date'] - df['prev_start']).dt.days.fillna(0)
    
    def count_active(row):
        active = df[(df['patient_id'] == row['patient_id']) & 
                    (df['start_date'] <= row['start_date']) & 
                    (df['end_date'] >= row['start_date'])]
        return len(active)
    
    df['polypharmacy_count'] = df.apply(count_active, axis=1)
    
    def detect_cascade(row):
        if pd.isna(row['prev_drug']):
            return 0, 0
        
        prev_side_effects = se_df[se_df['drug_name'] == row['prev_drug']]['side_effect_name'].tolist()
        curr_indications = indications.get(row['drug_name'], [])
        overlap = set(prev_side_effects).intersection(set(curr_indications))
        
        if overlap:
            return 1, 1
        return 0, 0

    results = df.apply(detect_cascade, axis=1)
    df['cascade_flag'] = [r[0] for r in results]
    df['symptom_drug_overlap'] = [r[1] for r in results]
    df['diagnosis_matches_side_effect'] = df['cascade_flag']
    
    return df

processed_df = engineer_features(timeline_df, side_effects_df, indications)

# ==========================================
# STEP 5 — ML MODEL
# ==========================================

features = ['time_to_next_prescription', 'polypharmacy_count', 
            'symptom_drug_overlap', 'diagnosis_matches_side_effect']

X = processed_df[features]
y = processed_df['cascade_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("\n--- ML Model Performance ---")
try:
    print(classification_report(y_test, model.predict(X_test)))
except Exception as e:
    print(f"Report skipped: {e}")

importances = pd.Series(model.feature_importances_, index=features)
plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind='barh', title='Feature Importance - Cascade Detection')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved feature importance plot to feature_importance.png")

# ==========================================
# STEP 6 — OUTPUT & REPORTING
# ==========================================

def generate_report(df, se_df):
    report = df[df['cascade_flag'] == 1].copy()
    
    if report.empty:
        return pd.DataFrame(columns=['patient_id', 'cascade_chain', 'risk_score', 'start_date'])
    
    def get_chain(row):
        prev_se = se_df[se_df['drug_name'] == row['prev_drug']]['side_effect_name'].tolist()
        curr_ind = indications.get(row['drug_name'], [])
        overlap = list(set(prev_se).intersection(set(curr_ind)))
        return f"{row['prev_drug']} -> {overlap[0]} -> {row['drug_name']}"

    report['cascade_chain'] = report.apply(get_chain, axis=1)
    
    # Handle single-class probability cases
    probs = model.predict_proba(report[features])
    if probs.shape[1] > 1:
        report['risk_score'] = probs[:, 1]
    else:
        # If model only knows one class, assign 1.0 or 0.0 based on that class
        class_idx = model.classes_[0]
        report['risk_score'] = 1.0 if class_idx == 1 else 0.0
    
    return report[['patient_id', 'cascade_chain', 'risk_score', 'start_date']]

final_report = generate_report(processed_df, side_effects_df)
print("\n--- Detected Medication Cascades ---")
if final_report.empty:
    print("No cascades detected in this dataset.")
else:
    print(final_report)
    final_report.to_csv('cascade_report.csv', index=False)
    print("\nReport exported to cascade_report.csv")


final_report = generate_report(processed_df, side_effects_df)
print("\n--- Detected Medication Cascades ---")
if final_report.empty:
    print("No cascades detected in this dataset.")
else:
    print(final_report)
    final_report.to_csv('cascade_report.csv', index=False)
    print("\nReport exported to cascade_report.csv")


