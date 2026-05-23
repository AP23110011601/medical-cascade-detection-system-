import pandas as pd
import os
from db_utils import engine, add_patient, add_prescription, init_db, seed_side_effects
from sqlalchemy import text

def import_datasets():
    print("Starting data import from CSVs to PostgreSQL...")
    init_db()
    
    # 1. Load Prescriptions Dataset
    if os.path.exists('prescriptions.csv'):
        df_p = pd.read_csv('prescriptions.csv')
        
        # Create patients from unique IDs in prescriptions
        unique_patients = df_p['patient_id'].unique()
        print(f"Importing {len(unique_patients)} patients...")
        
        for p_id in unique_patients:
            # Create a dummy patient record for each ID
            add_patient(p_id, f"Patient {p_id}", 45, "Unknown", "2023-01-01")
            
        # Import all prescriptions
        print(f"Importing {len(df_p)} prescriptions...")
        for _, row in df_p.iterrows():
            add_prescription(
                row['patient_id'], 
                row['drug_name'], 
                row['dosage'], 
                row['start_date'], 
                row['end_date'], 
                row.get('prescribing_doctor', 'Auto-Import')
            )
    
    # 2. Seed Side Effects
    if os.path.exists('side_effects.csv'):
        seed_side_effects('side_effects.csv')
        
    print("Import complete!")

if __name__ == "__main__":
    import_datasets()
