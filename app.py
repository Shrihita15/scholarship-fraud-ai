import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Scholarship Fraud Detection")
model = joblib.load("fraud_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# User input form
with st.form("fraud_form"):
    income_cert = st.number_input("Claimed Income (₹)", value=15000)
    actual_income = st.number_input("Actual Family Income (₹)", value=300000)
    attendance = st.slider("Attendance %", 0, 100, 42)
    verified = st.selectbox("Documents Verified", encoders['Documents_Verified'].classes_)
    enroll_status = st.selectbox("Enrollment Status", encoders['Enrollment_Status'].classes_)
    state = st.selectbox("Application State", encoders['Application_State'].classes_)
    spent = st.selectbox("Spent On", encoders['Spent_On'].classes_)
    scholarship_amt = st.number_input("Scholarship Amount (₹)", value=50000)
    submitted = st.form_submit_button("Check for Fraud")

if submitted:
    sample = pd.DataFrame([{ 
        'Income_Certificate_Amount': income_cert,
        'Actual_Income': actual_income,
        'Attendance': attendance,
        'Documents_Verified': encoders['Documents_Verified'].transform([verified])[0],
        'Enrollment_Status': encoders['Enrollment_Status'].transform([enroll_status])[0],
        'Application_State': encoders['Application_State'].transform([state])[0],
        'Scholarship_Amount': scholarship_amt,
        'Income_Ratio': income_cert / (actual_income + 1),
        'Low_Attendance': int(attendance < 60),
        'Fake_Income_Claim': int(income_cert < (actual_income / 2)),
        'Non_Education_Spend': int(encoders['Spent_On'].transform([spent])[0] != encoders['Spent_On'].transform(['Education'])[0])
    }])

    result = model.predict(sample)[0]
    st.success("✅ Genuine Application" if result == 0 else "❌ Fraud Detected!")
