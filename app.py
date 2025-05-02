import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model dan alat bantu
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model_compressed.pkl")
scaler = joblib.load("scaler.pkl")
scaler_columns = joblib.load("scaler_columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")
dl_model = load_model("dl_model.h5")
meta_model = joblib.load('meta_model.pkl')

# Judul
st.title("Prediksi Status Mahasiswa - Jaya Jaya Institut")

# Input Form
with st.form("student_form"):
    st.subheader("Masukkan Data Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        marital_status = st.selectbox(
            "Marital Status",
            options=["Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"]
        )
        marital_status_map = {
            "Single": 1,
            "Married": 2,
            "Widower": 3,
            "Divorced": 4,
            "Facto Union": 5,
            "Legally Separated": 6
        }
        marital_status = marital_status_map.get(marital_status, 0)

        application_mode = st.number_input("Application mode (1-1st phase, 39-Over 23 years old, 42-Transfer, etc.): ", step=1, value=0)
        application_order = st.slider("Application order (0 - first choice, 9 - last choice): ", min_value=0, max_value=9, value=0)
        course = st.number_input("Course (33 - Biofuel Production Technologies, 171 - Animation and Multimedia Design, etc.): ", step=1, value=0)

        daytime_evening_attendance = st.radio(
            "Daytime/evening attendance",
            options=["Daytime", "Evening"]
        )
        daytime_evening_attendance_map = {
            "Daytime": 1,
            "Evening": 0
        }
        daytime_evening_attendance = daytime_evening_attendance_map.get(daytime_evening_attendance, 0)
        
        previous_qualification = st.number_input("Previous qualification (1-Secondary education, 2-Bachelor's degree, etc.): ", step=1, value=0)
        previous_qualification_grade = st.number_input("Previous qualification grade (0 to 200): ", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        admission_grade = st.number_input("Admission grade (0 to 200): ", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        nationality = st.number_input("Nationality (1-Portuguese, 2-German, etc.): ", step=1, value=0)
        mothers_qualification = st.number_input("Mother's qualification (1-Secondary education, 2-Bachelor's degree, etc.): ", step=1, value=0)
        fathers_qualification = st.number_input("Father's qualification (1-Secondary education, 2-Bachelor's degree, etc.): ", step=1, value=0)
        mothers_occupation = st.number_input("Mother's occupation (0-Student, 1-Executive, etc.): ", step=1, value=0)
        fathers_occupation = st.number_input("Father's occupation (0-Student, 1-Executive, etc.): ", step=1, value=0)

    with col2:
        displaced = st.radio(
            "Displaced",
            options=["Yes", "No"]
        )
        displaced_map = {
            "Yes": 1,
            "No": 0
        }
        displaced = displaced_map.get(displaced, 0)

        educational_special_needs = st.radio("Educational special needs", options=["Yes", "No"])
        educational_special_needs_map = {
            "Yes": 1,
            "No": 0
        }
        educational_special_needs = educational_special_needs_map.get(educational_special_needs, 0)

        debtor = st.radio("Debtor", options=["Yes", "No"])
        debtor_map = {
            "Yes": 1,
            "No": 0
        }
        debtor = debtor_map.get(debtor, 0)

        tuition_fees_up_to_date = st.radio("Tuition fees up to date", options=["Yes", "No"])
        tuition_fees_up_to_date_map = {
            "Yes": 1,
            "No": 0
        }
        tuition_fees_up_to_date = tuition_fees_up_to_date_map.get(tuition_fees_up_to_date, 0)

        gender = st.radio(
            "Gender",
            options=["Male", "Female"]
        )
        gender_map = {
            "Male": 1,
            "Female": 0
        }
        gender = gender_map.get(gender, 0)

        scholarship_holder = st.radio("Scholarship holder", options=["Yes", "No"])
        scholarship_holder_map = {
            "Yes": 1,
            "No": 0
        }
        scholarship_holder = scholarship_holder_map.get(scholarship_holder, 0)

        age_at_enrollment = st.number_input("Age at enrollment: ", min_value=0, max_value=100, step=1, value=0)
        international = st.radio("International", options=["Yes", "No"])
        international_map = {
            "Yes": 1,
            "No": 0
        }
        international = international_map.get(international, 0)

        curricular_units_1st_sem_credited = st.number_input("Curricular units 1st sem (credited): ", step=1, value=0)
        curricular_units_1st_sem_enrolled = st.number_input("Curricular units 1st sem (enrolled): ", step=1, value=0)
        curricular_units_1st_sem_evaluations = st.number_input("Curricular units 1st sem (evaluations): ", step=1, value=0)
        curricular_units_1st_sem_approved = st.number_input("Curricular units 1st sem (approved): ", step=1, value=0)
        curricular_units_1st_sem_grade = st.number_input("Curricular units 1st sem (grade): ", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        curricular_units_1st_sem_without_evaluations = st.number_input("Curricular units 1st sem (without evaluations): ", step=1, value=0)

        curricular_units_2nd_sem_credited = st.number_input("Curricular units 2nd sem (credited): ", step=1, value=0)
        curricular_units_2nd_sem_enrolled = st.number_input("Curricular units 2nd sem (enrolled): ", step=1, value=0)
        curricular_units_2nd_sem_evaluations = st.number_input("Curricular units 2nd sem (evaluations): ", step=1, value=0)
        curricular_units_2nd_sem_approved = st.number_input("Curricular units 2nd sem (approved): ", step=1, value=0)
        curricular_units_2nd_sem_grade = st.number_input("Curricular units 2nd sem (grade): ", min_value=0.0, max_value=200.0, step=0.1, value=0.0)
        curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular units 2nd sem (without evaluations): ", step=1, value=0)

        unemployment_rate = st.number_input("Unemployment rate: ", min_value=0.0, step=0.1, value=0.0)
        inflation_rate = st.number_input("Inflation rate: ", min_value=0.0, step=0.1, value=0.0)
        gdp = st.number_input("GDP: ", min_value=-9.0, step=0.1, value=0.0)

    submitted = st.form_submit_button("Prediksi Status Mahasiswa")

def safe_divide(numerator, denominator):
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator).replace(0, np.nan)
    return (numerator / denominator).fillna(0)

# Proses Prediksi
if submitted:
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

    # Feature Engineering
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

    # Hilangkan NaN
    df.fillna(0, inplace=True)

    # Drop fitur yang tidak dipakai
    drop_cols = ['missing_eval_1st', 'missing_eval_2nd', 'pass_rate_1st', 'pass_rate_2nd']
    df = df.drop(drop_cols, axis=1, errors='ignore')

    # Pastikan kolom df sesuai dengan scaler_columns
    df = df.reindex(columns=scaler_columns, fill_value=0)

    # Melakukan transformasi dengan scaler yang sudah dilatih
    X_scaled = scaler.transform(df)
    
    # Proses prediksi dengan model yang sudah dilatih
    proba_rf = rf_model.predict_proba(X_scaled)
    proba_xgb = xgb_model.predict_proba(X_scaled)
    proba_dl = dl_model.predict(X_scaled)

    # Ensemble prediksi
    X_meta = np.hstack([proba_rf, proba_xgb, proba_dl])
    final_proba = meta_model.predict_proba(X_meta)
    final_pred = np.argmax(final_proba, axis=1)

    # Mapping label ke status mahasiswa
    predicted_label = label_encoders['Status'].inverse_transform(final_pred.astype(int))

    # Menampilkan hasil prediksi
    st.write(f"Prediksi Status Mahasiswa: {predicted_label[0]}")
