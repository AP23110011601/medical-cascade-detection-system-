import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv() # This will automatically pick up the .env in the current folder

DATABASE_URL = os.getenv("DATABASE_URL")

def get_engine():
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        print(f"Error creating engine: {e}")
        return None

engine = get_engine()

def init_db():
    """Initialize the database tables if they don't exist."""
    with engine.connect() as conn:
        # Create Patients table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100),
                age INTEGER,
                gender VARCHAR(20),
                admission_date DATE
            )
        """))
        
        # Create Prescriptions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prescriptions (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                drug_name VARCHAR(100),
                dosage VARCHAR(50),
                start_date DATE,
                end_date DATE,
                prescribing_doctor VARCHAR(100)
            )
        """))
        
        # Create Side Effects table (lookup)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS side_effects (
                id SERIAL PRIMARY KEY,
                drug_name VARCHAR(100),
                side_effect_name VARCHAR(100),
                severity VARCHAR(50)
            )
        """))
        conn.commit()

def add_patient(patient_id, name, age, gender, admission_date):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO patients (patient_id, name, age, gender, admission_date)
            VALUES (:id, :name, :age, :gender, :date)
            ON CONFLICT (patient_id) DO NOTHING
        """), {"id": patient_id, "name": name, "age": age, "gender": gender, "date": admission_date})
        conn.commit()

def add_prescription(patient_id, drug_name, dosage, start_date, end_date, doctor):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO prescriptions (patient_id, drug_name, dosage, start_date, end_date, prescribing_doctor)
            VALUES (:id, :drug, :dose, :start, :end, :doctor)
        """), {"id": patient_id, "drug": drug_name, "dose": dosage, "start": start_date, "end": end_date, "doctor": doctor})
        conn.commit()

def get_patient_history(patient_id):
    query = "SELECT * FROM prescriptions WHERE patient_id = :id ORDER BY start_date"
    return pd.read_sql(text(query), engine, params={"id": patient_id})

def get_patient_details(patient_id):
    query = "SELECT * FROM patients WHERE patient_id = :id"
    df = pd.read_sql(text(query), engine, params={"id": patient_id})
    return df.iloc[0] if not df.empty else None

def get_all_side_effects():
    return pd.read_sql("SELECT * FROM side_effects", engine)

def seed_side_effects(csv_path):
    """Seed the side_effects table from a CSV file."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_sql('side_effects', engine, if_exists='replace', index=False)
        print("Side effects seeded successfully.")
