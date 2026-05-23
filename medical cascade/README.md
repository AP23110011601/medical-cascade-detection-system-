# 🏥 CascadeCore AI: Medication Cascade Detection System

CascadeCore AI is a professional Clinical Decision Support System (CDSS) designed to identify **Medication Cascades**. A medication cascade occurs when a drug's side effect is misidentified as a new medical condition, leading to the prescription of a second, unnecessary drug.

## 🚀 Features
- **Role-Based Access**: Specialized portals for Receptionists (In-take) and Clinical Doctors (Diagnostics).
- **Automated Detection**: Uses Machine Learning and Pharmacological mapping to flag suspicious prescription sequences.
- **Visual Clinical Timeline**: Professional Gantt-style charts showing treatment trajectories and risk levels.
- **PostgreSQL Integration**: Robust data persistence for patient records and prescription history.
- **Explainable AI (XAI)**: Real-time insights into why a sequence was flagged, showing the specific symptom overlap.

## 🧠 Machine Learning Model
The system employs a **Random Forest Classifier** (Scikit-Learn) trained on clinical history patterns.
- **Features Analyzed**:
  - `Overlap`: Boolean indicator if the second drug's indication matches the first drug's side effect.
  - `Time_to_Next`: Duration (in days) between consecutive prescriptions.
- **Model Output**: Probability score (0-100%) indicating the confidence level of a detected cascade.

## 🛠️ Technology Stack
- **Frontend**: Streamlit (Premium UI with Custom CSS)
- **Data Processing**: Pandas, NumPy
- **Database**: PostgreSQL
- **ORM/Connection**: SQLAlchemy, Psycopg2
- **Machine Learning**: Scikit-Learn (Random Forest)
- **Visualizations**: Plotly (Timeline & Gauge Charts)
- **Environment**: Python-Dotenv

## 📋 Installation & Setup

### 1. Database Configuration
1. Install **PostgreSQL** on your system.
2. Create a database named `ads` (or as specified in your `.env`).
3. Update the `.env` file in the project root with your credentials:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=ads
   DB_USER=postgres
   DB_PASSWORD=your_password
   DATABASE_URL=postgresql://postgres:your_password@localhost:5432/ads
   ```

### 2. Dependencies
Install the required Python libraries:
```bash
pip install streamlit pandas numpy sqlalchemy psycopg2-binary scikit-learn plotly python-dotenv
```

### 3. Initialize the System
Run the setup script to create tables and seed pharmacological data:
```bash
python setup_database.py
```

### 4. Data Import (Optional)
To import the provided 100-record dataset into the database:
```bash
python import_data.py
```

## 🏃 How to Run
Launch the application using Streamlit:
```bash
streamlit run app.py
```

## 💡 Example Test Cases
| Pattern | Detection Result | Logic |
| :--- | :--- | :--- |
| **Metformin → Omeprazole** | 🔴 Flagged | Metformin causes GI upset; Omeprazole treats it. |
| **Lisinopril → Amlodipine** | 🔴 Flagged | Lisinopril causes dizziness; Amlodipine treats it. |
| **Albuterol → Levothyroxine** | 🟢 Safe | No pharmacological overlap detected. |

---
*Developed for clinical innovation and patient safety.* 🛡️🩺
