import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model dan alat bantu
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
dl_model = load_model("dl_model.h5")

# Judul
st.title("Prediksi Status Mahasiswa - Jaya Jaya Institut")

# Input Form
with st.form("student_form"):
    st.subheader("Masukkan Data Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        
        marital_status = int(input("Marital status (1: single, 2: married, 3: widower, 4: divorced, 5: facto union, 6: legally separated): "))
        application_mode = int(input("Application mode (1-1st phase, 39-Over 23 years old, 42-Transfer, etc.): "))
        application_order = int(input("Application order (0 - first choice, 9 - last choice): "))
        course = int(input("Course (33 - Biofuel Production Technologies, 171 - Animation and Multimedia Design, etc.): "))
        daytime_evening_attendance = int(input("Daytime/evening attendance (1 – daytime, 0 – evening): "))
        previous_qualification = int(input("Previous qualification (1-Secondary education, 2-Bachelor's degree, etc.): "))
        
        previous_qualification_grade = float(input("Previous qualification grade (0 to 200): "))
        admission_grade = float(input("Admission grade (0 to 200): "))
        
        nationality = int(input("Nationality (1-Portuguese, 2-German, etc.): "))
        mothers_qualification = int(input("Mother's qualification (1-Secondary education, 2-Bachelor's degree, etc.): "))
        fathers_qualification = int(input("Father's qualification (1-Secondary education, 2-Bachelor's degree, etc.): "))
        mothers_occupation = int(input("Mother's occupation (0-Student, 1-Executive, etc.): "))
        fathers_occupation = int(input("Father's occupation (0-Student, 1-Executive, etc.): "))

    with col2:
        displaced = int(input("Displaced (1-yes, 0-no): "))
        educational_special_needs = int(input("Educational special needs (1-yes, 0-no): "))
        debtor = int(input("Debtor (1-yes, 0-no): "))
        tuition_fees_up_to_date = int(input("Tuition fees up to date (1-yes, 0-no): "))
        gender = int(input("Gender (1-male, 0-female): "))
        scholarship_holder = int(input("Scholarship holder (1-yes, 0-no): "))
        age_at_enrollment = int(input("Age at enrollment: "))
        international = int(input("International (1-yes, 0-no): "))
    
        curricular_units_1st_sem_credited = int(input("Curricular units 1st sem (credited): "))
        curricular_units_1st_sem_enrolled = int(input("Curricular units 1st sem (enrolled): "))
        curricular_units_1st_sem_evaluations = int(input("Curricular units 1st sem (evaluations): "))
        curricular_units_1st_sem_approved = int(input("Curricular units 1st sem (approved): "))
        curricular_units_1st_sem_grade = float(input("Curricular units 1st sem (grade): "))
        curricular_units_1st_sem_without_evaluations = int(input("Curricular units 1st sem (without evaluations): "))
    
        curricular_units_2nd_sem_credited = int(input("Curricular units 2nd sem (credited): "))
        curricular_units_2nd_sem_enrolled = int(input("Curricular units 2nd sem (enrolled): "))
        curricular_units_2nd_sem_evaluations = int(input("Curricular units 2nd sem (evaluations): "))
        curricular_units_2nd_sem_approved = int(input("Curricular units 2nd sem (approved): "))
        curricular_units_2nd_sem_grade = float(input("Curricular units 2nd sem (grade): "))
        curricular_units_2nd_sem_without_evaluations = int(input("Curricular units 2nd sem (without evaluations): "))
    
        unemployment_rate = float(input("Unemployment rate: "))
        inflation_rate = float(input("Inflation rate: "))
        gdp = float(input("GDP: "))

    submitted = st.form_submit_button("Prediksi Status Mahasiswa")

# Fungsi bantu
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

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

    # Feature engineering
    df['pass_rate_total'] = safe_divide(
        curricular_units_1st_sem_approved + curricular_units_2nd_sem_approved,
        curricular_units_1st_sem_enrolled + curricular_units_2nd_sem_enrolled
    )
    df['avg_grade'] = (curricular_units_1st_sem_grade + curricular_units_2nd_sem_grade) / 2
    df['grade_gap'] = admission_grade - df['avg_grade']
    df['total_enrolled'] = curricular_units_1st_sem_enrolled + curricular_units_2nd_sem_enrolled
    df['total_approved'] = curricular_units_1st_sem_approved + curricular_units_2nd_sem_approved
    df['total_failed'] = df['total_enrolled'] - df['total_approved']
    df['unit_completion_ratio'] = safe_divide(df['total_approved'], df['total_enrolled'])
    df['financial_risk'] = (1 - tuition_fees_up_to_date) + debtor + scholarship_holder
    df['special_case'] = displaced + educational_special_needs + international

    # Drop kolom-kolom tidak digunakan (optional)
    drop_cols = ['pass_rate_total']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Scaling
    X_scaled = scaler.transform(df)

    # Predict dari 3 model
    proba_rf = rf_model.predict_proba(X_scaled)
    proba_xgb = xgb_model.predict_proba(X_scaled)
    proba_dl = dl_model.predict(X_scaled)

    # Voting ensemble (ambil rata-rata probabilitas)
    final_proba = (proba_rf + proba_xgb + proba_dl) / 3
    final_pred = np.argmax(final_proba, axis=1)

    # Inverse label
    label_encoder_status = label_encoders['Status']
    predicted_label = label_encoder_status.inverse_transform(final_pred.astype(int))

    st.success(f"🎯 Prediksi Status Mahasiswa: **{predicted_label[0]}**")
