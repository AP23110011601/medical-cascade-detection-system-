import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from db_utils import engine, init_db, add_patient, add_prescription, get_patient_history, get_patient_details, get_all_side_effects, seed_side_effects
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="CascadeCore AI | Clinical Decision Support", page_icon="🛡️", layout="wide")

# --- PROFESSIONAL THEME & CSS ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
    /* Main Background and Font */
    .main { 
        background-color: #F0F2F6; 
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Header */
    .stApp header { background-color: transparent; }
    
    /* Card-like containers */
    div.stBlock {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #1E293B; color: white; }
    
    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 12px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
    }
    
    /* Title and Subtitle */
    h1 { color: #0F172A; font-weight: 800; letter-spacing: -0.025em; }
    h2, h3 { color: #1E293B; font-weight: 600; }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #2563EB; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- DB INITIALIZATION ---
@st.cache_resource
def startup():
    try:
        init_db()
        if os.path.exists('side_effects.csv'):
            seed_side_effects('side_effects.csv')
        return True
    except Exception as e:
        st.error(f"⚠️ Clinical DB Sync Error: {e}")
        return False

db_ready = startup()

# --- HELPER: GET DYNAMIC DRUG LIST ---
def get_drug_list():
    try:
        df = pd.read_sql("SELECT DISTINCT drug_name FROM side_effects", engine)
        if df.empty:
            return ["Lisinopril", "Metformin", "Atorvastatin", "Omeprazole", "Amlodipine", "Albuterol", "Levothyroxine"]
        return sorted(df['drug_name'].tolist())
    except:
        return ["Lisinopril", "Metformin", "Atorvastatin", "Omeprazole", "Amlodipine"]

# --- ML TRAINING ENGINE ---
@st.cache_resource
def train_model():
    try:
        history = pd.read_sql("SELECT * FROM prescriptions", engine)
        se_df = get_all_side_effects()
        if history.empty or se_df.empty: return None, []

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
        
        processed = []
        for pid in history['patient_id'].unique():
            p_hist = history[history['patient_id'] == pid].sort_values('start_date')
            p_hist['prev_drug'] = p_hist['drug_name'].shift(1)
            p_hist['prev_start'] = p_hist['start_date'].shift(1)
            processed.append(p_hist)
        
        full_df = pd.concat(processed)
        full_df['time_to_next'] = (pd.to_datetime(full_df['start_date']) - pd.to_datetime(full_df['prev_start'])).dt.days.fillna(0)
        
        def label_cascade(row):
            if pd.isna(row['prev_drug']): return 0, 0
            prev_se = se_df[se_df['drug_name'] == row['prev_drug']]['side_effect_name'].tolist()
            curr_ind = drug_indications.get(row['drug_name'], [])
            overlap = 1 if set(prev_se).intersection(set(curr_ind)) else 0
            return overlap, overlap

        labels = full_df.apply(label_cascade, axis=1)
        full_df['label'] = [l[0] for l in labels]
        full_df['overlap'] = [l[1] for l in labels]
        
        features = ['time_to_next', 'overlap']
        X = full_df[features]
        y = full_df['label']
        
        if len(y.unique()) < 2:
            X = pd.concat([X, pd.DataFrame([[100, 0], [10, 1]], columns=features)])
            y = pd.concat([y, pd.Series([0, 1])])

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, features
    except Exception as e:
        return None, []

trained_model, feature_names = train_model()

# --- APP NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=80)
    st.title("CascadeCore AI")
    st.caption("Advanced Decision Support v2.0")
    st.markdown("---")
    user_role = st.radio("System Access", ["Receptionist", "Clinical Doctor", "System Analytics"])
    
    st.markdown("---")
    with st.expander("🔬 Validation Scenarios"):
        st.subheader("🔴 Flagged Cascades")
        st.caption("Test the detection logic:")
        st.info("**Pattern 1:** Metformin → Omeprazole")
        st.info("**Pattern 2:** Lisinopril → Amlodipine")
        
        st.subheader("🟢 Safe Sequences")
        st.caption("No flag will be raised:")
        st.success("**Case 1:** Albuterol → Levothyroxine")
        st.success("**Case 2:** Metformin → Sertraline")

# --- RECEPTIONIST VIEW ---
if user_role == "Receptionist":
    st.title("📋 Patient Intake & Records")
    tab1, tab2 = st.tabs(["🆕 Register Patient", "📂 Historical Records"])
    
    with tab1:
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                p_id = st.text_input("National Patient ID", placeholder="e.g. PX-990", key="reg_id")
                p_name = st.text_input("Patient Full Name", key="reg_name")
            with col2:
                p_age = st.number_input("Age (Years)", min_value=0, max_value=120, value=35, key="reg_age")
                p_date = st.date_input("Registration Date", key="reg_date")
            
            if st.button("Initialize Clinical Profile", key="reg_btn"):
                if p_id and p_name:
                    add_patient(p_id, p_name, p_age, "Not Specified", p_date)
                    st.toast("✅ Profile Created")
                else:
                    st.warning("Required fields missing")

    with tab2:
        search_id = st.text_input("🔍 Search Clinical ID", placeholder="Search by ID...", key="search_receptionist")
        if search_id:
            details = get_patient_details(search_id)
            if details is not None:
                st.markdown(f"### {details['name']} (Age: {details['age']})")
                history = get_patient_history(search_id)
                if not history.empty:
                    st.dataframe(history[['drug_name', 'dosage', 'start_date', 'end_date']], use_container_width=True)
                
                with st.expander("➕ Update Medication History"):
                    c1, c2 = st.columns(2)
                    with c1:
                        drug = st.selectbox("Select Medication", get_drug_list())
                        dose = st.text_input("Dosage Instructions")
                    with c2:
                        s_date = st.date_input("Start Date")
                        e_date = st.date_input("End Date")
                    if st.button("Append Record"):
                        add_prescription(search_id, drug, dose, s_date, e_date, "In-take Desk")
                        st.success("Record Saved")
                        st.rerun()

# --- CLINICAL DOCTOR VIEW ---
elif user_role == "Clinical Doctor":
    st.title("🩺 Clinical Diagnostics Dashboard")
    patient_id = st.text_input("🔍 Patient ID Search", placeholder="e.g. P001")
    
    if patient_id:
        details = get_patient_details(patient_id)
        if details is not None:
            st.markdown(f"## {details['name']}")
            st.caption(f"Profile Summary | Age: {details['age']} | ID: {patient_id}")
            
            history = get_patient_history(patient_id)
            if not history.empty:
                audit_key = f"audit_{patient_id}"
                if audit_key not in st.session_state: st.session_state[audit_key] = False
                
                if st.button("🛡️ Execute AI Cascade Audit"):
                    st.session_state[audit_key] = True

                if st.session_state[audit_key]:
                    with st.spinner("Analyzing pharmacological interactions..."):
                        se_df = get_all_side_effects()
                        history = history.sort_values('start_date')
                        history['start_date'] = pd.to_datetime(history['start_date'])
                        history['end_date'] = pd.to_datetime(history['end_date'])
                        history['prev_drug'] = history['drug_name'].shift(1)
                        history['prev_start'] = history['start_date'].shift(1)
                        
                        def analyze(row):
                            if pd.isna(row['prev_drug']): return 0, ""
                            prev_se = se_df[se_df['drug_name'] == row['prev_drug']]['side_effect_name'].tolist()
                            # Enhanced map for demo
                            ind_map = {
                                'Omeprazole': ['stomach pain', 'gastrointestinal upset', 'headache'], 
                                'Amlodipine': ['dizziness', 'swelling', 'hypertension'],
                                'Betahistine': ['dizziness']
                            }
                            curr_ind = ind_map.get(row['drug_name'], [])
                            overlap = list(set(prev_se).intersection(set(curr_ind)))
                            return (1, overlap[0]) if overlap else (0, "")

                        res = history.apply(analyze, axis=1)
                        history['is_cascade'] = [r[0] for r in res]
                        history['overlap_desc'] = [r[1] for r in res]
                        
                        if trained_model:
                            history['time_to_next'] = (history['start_date'] - history['prev_start']).dt.days.fillna(0)
                            history['risk'] = trained_model.predict_proba(history[['time_to_next', 'is_cascade']].rename(columns={'is_cascade': 'overlap'}))[:, 1]
                        else:
                            history['risk'] = history['is_cascade'].astype(float) * 0.8
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Robust Gantt-style Bar Chart
                            history['duration'] = (history['end_date'] - history['start_date']).dt.days
                            # Ensure at least 1 day for visibility
                            history.loc[history['duration'] <= 0, 'duration'] = 1
                            
                            fig = px.bar(history, 
                                        base="start_date", 
                                        x="duration", 
                                        y="drug_name", 
                                        color="risk",
                                        color_continuous_scale="RdBu_r",
                                        orientation='h',
                                        text="drug_name",
                                        title="Clinical History & Treatment Durations")
                            
                            fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(family="Inter, sans-serif", color="white"),
                                xaxis_title="Timeline",
                                yaxis_title="",
                                showlegend=False,
                                height=450
                            )
                            
                            fig.update_xaxes(
                                type='date',
                                gridcolor="rgba(255,255,255,0.1)",
                                tickfont=dict(color="white")
                            )
                            
                            fig.update_yaxes(showticklabels=False) # Names are inside bars
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            max_risk = history['risk'].max()
                            fig_g = go.Figure(go.Indicator(
                                mode = "gauge+number", 
                                value = max_risk * 100, 
                                title = {'text': "Cascade Probability (%)", 'font': {'size': 20, 'color': 'white'}},
                                number = {'font': {'color': 'white', 'size': 50}},
                                gauge = {
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                    'bar': {'color': "#2563EB"},
                                    'bgcolor': "rgba(255,255,255,0.1)",
                                    'borderwidth': 2,
                                    'bordercolor': "white",
                                    'steps' : [
                                        {'range': [0, 30], 'color': "#065F46"},
                                        {'range': [30, 70], 'color': "#854D0E"},
                                        {'range': [70, 100], 'color': "#991B1B"}]
                                }
                            ))
                            fig_g.update_layout(height=400, margin=dict(l=30, r=30, t=80, b=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
                            st.plotly_chart(fig_g, use_container_width=True)

                        if history['is_cascade'].any():
                            st.error("⚠️ **Potential Medication Cascade Identified**")
                            st.markdown("The system has detected a drug sequence where the subsequent medication may be treating a side effect of the prior medication.")
                            st.table(history[history['is_cascade']==1][['prev_drug', 'overlap_desc', 'drug_name']])
                        else:
                            st.success("✅ **Pharmacological Audit Complete**")
                            st.markdown("No suspicious prescription sequences or cascade patterns were detected in the current history.")
                else:
                    st.info("Ready for Audit. Click the button above to begin pharmacological screening.")
                    st.subheader("Active Medications")
                    st.dataframe(history[['drug_name', 'dosage', 'start_date', 'end_date']], use_container_width=True)
            else:
                st.info("No prescription history found for this patient.")

# --- ANALYTICS ---
elif user_role == "System Analytics":
    st.title("📊 Population Health & Model Performance")
    
    # Load Metrics
    metrics = {}
    if os.path.exists('models/metrics.json'):
        import json
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
    with col2: st.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
    with col3: st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
    with col4: st.metric("F1-Score", f"{metrics.get('f1', 0)*100:.1f}%")

    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("📉 Confusion Matrix")
        if os.path.exists('models/confusion_matrix.png'):
            st.image('models/confusion_matrix.png', caption="Model Prediction Performance")
        else:
            st.info("Run train_model.py to generate matrix.")
            
    with c2:
        st.subheader("🧠 Model Rationale")
        st.write("""
        **Why we use Random Forest:**
        *   **Non-Linear Patterns**: Handles complex medical interactions that simple linear models miss.
        *   **Feature Importance**: Clearly identifies which clinical factors (like drug overlap) are driving the risk.
        *   **Robustness**: Highly resistant to outliers in patient history data.
        *   **Clinical Logic**: Naturally mimics the "Decision Tree" thinking process used by human clinicians.
        """)

    st.markdown("---")
    if trained_model:
        c_pie, c_heat = st.columns(2)
        with c_pie:
            st.subheader("🧩 Feature Influence")
            importances = pd.Series(trained_model.feature_importances_, index=feature_names)
            fig_p = px.pie(values=importances.values, names=importances.index, hole=0.4, 
                         template="plotly_dark")
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=400)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c_heat:
            st.subheader("🌡️ Feature Correlation Heatmap")
            if os.path.exists('models/correlation.json'):
                corr_df = pd.read_json('models/correlation.json')
                fig_h = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', 
                                 template="plotly_dark", title="Symptom vs Timeline Correlation")
                fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=400)
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("Run train_model.py to generate correlation data.")
else:
    st.info("Select a role in the sidebar to enter the portal.")
