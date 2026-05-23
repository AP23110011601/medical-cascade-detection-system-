import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from db_utils import engine, get_all_side_effects
import joblib

def create_training_report():
    print("Starting Advanced Model Training Pipeline...")
    
    # 1. Load Data from PostgreSQL
    try:
        history = pd.read_sql("SELECT * FROM prescriptions", engine)
        se_df = get_all_side_effects()
        
        if history.empty:
            print("[ERROR] No data found in database. Run import_data.py first.")
            return
    except Exception as e:
        print(f"[ERROR] Database Error: {e}")
        return

    # 2. Feature Engineering & Labeling
    processed = []
    drug_indications = {
        'Omeprazole': ['stomach pain', 'gastrointestinal upset', 'headache'],
        'Amlodipine': ['swelling', 'dizziness', 'hypertension'],
        'Lisinopril': ['hypertension'],
        'Metformin': ['diabetes'],
        'Atorvastatin': ['high cholesterol', 'muscle pain'],
        'Simvastatin': ['high cholesterol', 'muscle weakness'],
        'Albuterol': ['asthma'],
        'Levothyroxine': ['hypothyroidism']
    }

    for pid in history['patient_id'].unique():
        p_hist = history[history['patient_id'] == pid].sort_values('start_date')
        p_hist['prev_drug'] = p_hist['drug_name'].shift(1)
        p_hist['prev_start'] = p_hist['start_date'].shift(1)
        processed.append(p_hist)
    
    full_df = pd.concat(processed)
    full_df['time_to_next'] = (pd.to_datetime(full_df['start_date']) - pd.to_datetime(full_df['prev_start'])).dt.days.fillna(0)
    
    def label_cascade(row):
        if pd.isna(row['prev_drug']): return 0
        prev_se = se_df[se_df['drug_name'] == row['prev_drug']]['side_effect_name'].tolist()
        curr_ind = drug_indications.get(row['drug_name'], [])
        return 1 if set(prev_se).intersection(set(curr_ind)) else 0

    full_df['label'] = full_df.apply(label_cascade, axis=1)
    
    # Prepare Features
    # Using 'label' as proxy for overlap for training
    X = full_df[['time_to_next', 'label']].rename(columns={'label': 'overlap'})
    y = full_df['label']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Multi-Model Training
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    print("\n--- Model Evaluation Results ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")

    # 5. Confusion Matrix (Using Best Model: Random Forest)
    best_model = models["Random Forest"]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    print("\n--- Confusion Matrix (Random Forest) ---")
    print(cm)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_best))

    # 6. Save Artifacts
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(best_model, 'models/cascade_model.pkl')
    
    # Save Metrics for App
    report = classification_report(y_test, y_pred_best, output_dict=True)
    import json
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_best),
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall'],
        "f1": report['macro avg']['f1-score'],
        "total_samples": len(y_test)
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)

    # NEW: Save Correlation Matrix
    corr = X.corr()
    corr.to_json('models/correlation.json')
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig('models/correlation.png')
    
    print("\n[OK] Best model saved to 'models/cascade_model.pkl'")
    print("[OK] Metrics saved to 'models/metrics.json'")
    print("[OK] Correlation Matrix saved to 'models/correlation.png'")

    # Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Cascade'], yticklabels=['Safe', 'Cascade'])
    plt.title('Confusion Matrix: Medication Cascade Detection')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('models/confusion_matrix.png')
    print("[OK] Confusion Matrix plot saved to 'models/confusion_matrix.png'")

if __name__ == "__main__":
    create_training_report()
