import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.markdown("""
    <style>
    .stApp {
        background-color: #ffeaf4;
        font-family: 'Segoe UI', sans-serif;
    }

    .custom-title {
        color: #d63384;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .custom-subheader {
        color: #b30059;
        font-size: 22px;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    div[data-testid="stForm"] {
        background-color: #fff0f5;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #ffb6c1;
        box-shadow: 2px 2px 12px rgba(220, 20, 60, 0.1);
    }

    button[kind="primary"] {
        background-color: #ff69b4;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }

    button[kind="primary"]:hover {
        background-color: #ff85c1;
    }

    .stForm {
        max-width: 95%;
        margin: auto;
    }

    .element-container:has(.stForm) {
        max-width: 90vw;
    }
    
    .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model_compressed.pkl")
scaler = joblib.load("scaler.pkl")
scaler_columns = joblib.load("scaler_columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")
dl_model = load_model("dl_model.h5")
meta_model = joblib.load('meta_model.pkl')

st.markdown('''
    <div class="custom-title">
        Prediksi Status Mahasiswa<br>
        🎓 Jaya Jaya Institut 🎓
    </div>
''', unsafe_allow_html=True)

with st.form("student_form"):
    st.markdown('<div class="custom-subheader">Masukkan Data Mahasiswa</div>', unsafe_allow_html=True)
    st.markdown("**Harap masukkan informasi yang diperlukan untuk mendapatkan prediksi status mahasiswa.**")

    col1, col2,col3 = st.columns(3)

    with col1:
        st.markdown("Data Mahasiswa")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"])
        marital_status = {
            "Single": 1, "Married": 2, "Widower": 3,
            "Divorced": 4, "Facto Union": 5, "Legally Separated": 6
        }.get(marital_status, 0)

        gender = st.radio("Gender", ["Male", "Female"])
        gender = 1 if gender == "Male" else 0

        age_at_enrollment = st.number_input("Age at enrollment:", min_value=0, max_value=100, step=1, value=0)

        application_mode = st.number_input("Application mode:", min_value=0, step=1, value=0)
        application_order = st.slider("Application order (0-9):", min_value=0, max_value=9, value=0)
        course = st.number_input("Course:", min_value=0, step=1, value=0)

        daytime_evening_attendance = st.radio("Daytime/evening attendance", ["Daytime", "Evening"])
        daytime_evening_attendance = 1 if daytime_evening_attendance == "Daytime" else 0

        previous_qualification = st.number_input("Previous qualification:", min_value=0, step=1, value=0)
        previous_qualification_grade = st.number_input("Previous qualification grade (0-200):", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        admission_grade = st.number_input("Admission grade (0-200):", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        nationality = st.number_input("Nationality:", min_value=0, step=1, value=0)
        mothers_qualification = st.number_input("Mother's qualification:", min_value=0, step=1, value=0)
        fathers_qualification = st.number_input("Father's qualification:", min_value=0, step=1, value=0)
        mothers_occupation = st.number_input("Mother's occupation:", min_value=0, step=1, value=0)
        fathers_occupation = st.number_input("Father's occupation:", min_value=0, step=1, value=0)

    with col2:
        st.markdown("Data Finansial")
        international = st.radio("International", ["Yes", "No"])
        international = 1 if international == "Yes" else 0

        displaced = st.radio("Displaced", ["Yes", "No"])
        displaced = 1 if displaced == "Yes" else 0

        educational_special_needs = st.radio("Educational special needs", ["Yes", "No"])
        educational_special_needs = 1 if educational_special_needs == "Yes" else 0

        tuition_fees_up_to_date = st.radio("Tuition fees up to date", ["Yes", "No"])
        tuition_fees_up_to_date = 1 if tuition_fees_up_to_date == "Yes" else 0

        debtor = st.radio("Debtor", ["Yes", "No"])
        debtor = 1 if debtor == "Yes" else 0

        scholarship_holder = st.radio("Scholarship holder", ["Yes", "No"])
        scholarship_holder = 1 if scholarship_holder == "Yes" else 0

        unemployment_rate = st.number_input("Unemployment rate:", min_value=0.0, step=0.1, value=0.0)
        inflation_rate = st.number_input("Inflation rate:", min_value=0.0, step=0.1, value=0.0)
        gdp = st.number_input("GDP:", min_value=-9.0, step=0.1, value=0.0)
    
    with col3:
        st.markdown("Data Akademik")
        curricular_units_1st_sem_credited = st.number_input("Curricular units 1st sem (credited):", min_value=0, step=1, value=0)
        curricular_units_1st_sem_enrolled = st.number_input("Curricular units 1st sem (enrolled):", min_value=0, step=1, value=0)
        curricular_units_1st_sem_evaluations = st.number_input("Curricular units 1st sem (evaluations):", min_value=0, step=1, value=0)
        curricular_units_1st_sem_approved = st.number_input("Curricular units 1st sem (approved):", min_value=0, step=1, value=0)
        curricular_units_1st_sem_grade = st.number_input("Curricular units 1st sem (grade):", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        curricular_units_1st_sem_without_evaluations = st.number_input("Curricular units 1st sem (without evaluations):", min_value=0, step=1, value=0)

        curricular_units_2nd_sem_credited = st.number_input("Curricular units 2nd sem (credited):", min_value=0, step=1, value=0)
        curricular_units_2nd_sem_enrolled = st.number_input("Curricular units 2nd sem (enrolled):", min_value=0, step=1, value=0)
        curricular_units_2nd_sem_evaluations = st.number_input("Curricular units 2nd sem (evaluations):", min_value=0, step=1, value=0)
        curricular_units_2nd_sem_approved = st.number_input("Curricular units 2nd sem (approved):", min_value=0, step=1, value=0)
        curricular_units_2nd_sem_grade = st.number_input("Curricular units 2nd sem (grade):", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular units 2nd sem (without evaluations):", min_value=0, step=1, value=0)

    submitted = st.form_submit_button("Prediksi Status Mahasiswa")

def safe_divide(numerator, denominator):
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator).replace(0, np.nan)
    return (numerator / denominator).fillna(0)

if submitted:
    numeric_fields = {
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Previous qualification": previous_qualification,
        "Previous qualification grade": previous_qualification_grade,
        "Admission grade": admission_grade,
        "Nationality": nationality,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        "Age at enrollment": age_at_enrollment,
        "Curricular units 1st sem credited": curricular_units_1st_sem_credited,
        "Curricular units 1st sem enrolled": curricular_units_1st_sem_enrolled,
        "Curricular units 1st sem evaluations": curricular_units_1st_sem_evaluations,
        "Curricular units 1st sem approved": curricular_units_1st_sem_approved,
        "Curricular units 1st sem grade": curricular_units_1st_sem_grade,
        "Curricular units 1st sem without evaluations": curricular_units_1st_sem_without_evaluations,
        "Curricular units 2nd sem credited": curricular_units_2nd_sem_credited,
        "Curricular units 2nd sem enrolled": curricular_units_2nd_sem_enrolled,
        "Curricular units 2nd sem evaluations": curricular_units_2nd_sem_evaluations,
        "Curricular units 2nd sem approved": curricular_units_2nd_sem_approved,
        "Curricular units 2nd sem grade": curricular_units_2nd_sem_grade,
        "Curricular units 2nd sem without evaluations": curricular_units_2nd_sem_without_evaluations,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate
    }
    
    negative_fields = [name for name, value in numeric_fields.items() if value < 0]

    if negative_fields:
        st.error(f"⚠️ Nilai tidak boleh negatif untuk kolom: {', '.join(negative_fields)}")
    else:
        data = {
            'Marital_status': marital_status,
            'Application_mode': application_mode,
            'Application_order': application_order,
            'Course': course,
            'Daytime_evening_attendance': daytime_evening_attendance,
            'Previous_qualification': previous_qualification,
            'Previous_qualification_grade': previous_qualification_grade,
            'Admission_grade': admission_grade,
            'Nacionality': nationality,
            'Mothers_qualification': mothers_qualification,
            'Fathers_qualification': fathers_qualification,
            'Mothers_occupation': mothers_occupation,
            'Fathers_occupation': fathers_occupation,
            'Displaced': displaced,
            'Educational_special_needs': educational_special_needs,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_fees_up_to_date,
            'Gender': gender,
            'Scholarship_holder': scholarship_holder,
            'Age_at_enrollment': age_at_enrollment,
            'International': international,
            'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
            'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
            'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
            'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
            'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
            'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_sem_without_evaluations,
            'Curricular_units_2nd_sem_credited': curricular_units_2nd_sem_credited,
            'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
            'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
            'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
            'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
            'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations,
            'Unemployment_rate': unemployment_rate,
            'Inflation_rate': inflation_rate,
            'GDP': gdp
        }
        
        df = pd.DataFrame([data])
        df['pass_rate_1st'] = safe_divide(df['Curricular_units_1st_sem_approved'], df['Curricular_units_1st_sem_enrolled'])
        df['pass_rate_2nd'] = safe_divide(df['Curricular_units_2nd_sem_approved'], df['Curricular_units_2nd_sem_enrolled'])
        df['pass_rate_total'] = safe_divide(
            df['Curricular_units_1st_sem_approved'] + df['Curricular_units_2nd_sem_approved'],
            df['Curricular_units_1st_sem_enrolled'] + df['Curricular_units_2nd_sem_enrolled']
        )
        df['missing_eval_1st'] = safe_divide(df['Curricular_units_1st_sem_without_evaluations'], df['Curricular_units_1st_sem_enrolled'])
        df['missing_eval_2nd'] = safe_divide(df['Curricular_units_2nd_sem_without_evaluations'], df['Curricular_units_2nd_sem_enrolled'])
        df['missing_eval_total'] = safe_divide(
            df['Curricular_units_1st_sem_without_evaluations'] + df['Curricular_units_2nd_sem_without_evaluations'],
            df['Curricular_units_1st_sem_enrolled'] + df['Curricular_units_2nd_sem_enrolled']
        )
        df['avg_grade'] = (df['Curricular_units_1st_sem_grade'] + df['Curricular_units_2nd_sem_grade']) / 2
        df['grade_gap'] = df['Admission_grade'] - df['avg_grade']
        df['total_enrolled'] = df['Curricular_units_1st_sem_enrolled'] + df['Curricular_units_2nd_sem_enrolled']
        df['total_approved'] = df['Curricular_units_1st_sem_approved'] + df['Curricular_units_2nd_sem_approved']
        df['total_evaluations'] = df['Curricular_units_1st_sem_evaluations'] + df['Curricular_units_2nd_sem_evaluations']
        df['total_failed'] = df['total_enrolled'] - df['total_approved']
        df['unit_completion_ratio'] = safe_divide(df['total_approved'], df['total_enrolled'])
        df['financial_risk'] = (1 - df['Tuition_fees_up_to_date']) + df['Debtor'] + df['Scholarship_holder']
        df['special_case'] = df['Displaced'] + df['Educational_special_needs'] + df['International']
    
        df.fillna(0, inplace=True)
        drop_cols = ['missing_eval_1st', 'missing_eval_2nd', 'pass_rate_1st', 'pass_rate_2nd']
        df = df.drop(drop_cols, axis=1, errors='ignore')
        df = df.reindex(columns=scaler_columns, fill_value=0)
        X_scaled = scaler.transform(df)
    
        # Prediction
        proba_rf = rf_model.predict_proba(X_scaled)
        proba_xgb = xgb_model.predict_proba(X_scaled)
        proba_dl = dl_model.predict(X_scaled)
        X_meta = np.hstack([proba_rf, proba_xgb, proba_dl])
        final_proba = meta_model.predict_proba(X_meta)
        final_pred = np.argmax(final_proba, axis=1)
        predicted_label = label_encoders['Status'].inverse_transform(final_pred.astype(int))
    
        if predicted_label[0].lower() == "dropout":
            st.error(f"⚠️ Prediksi Status Mahasiswa: **{predicted_label[0]}**")
        else:
            st.success(f"🎉 Prediksi Status Mahasiswa: **{predicted_label[0]}**")
