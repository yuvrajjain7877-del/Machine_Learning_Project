import streamlit as  st
import pandas as pd
import joblib


model = joblib.load("SVM_heart.pkl")
scaler = joblib.load("Scaler_heart.pkl")
expected_columns = joblib.load("Columns_heart.pkl")



st.title("Heart Stroke Prediction by Yuvraj !! 😎😎")
st.markdown("Provide the Following Details:")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_HR = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("OldPeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


if st.button("Predict"):
    raw_input = {
        "Age" : age,
        "RestingBP" : resting_bp,
        "Cholesterol" : cholesterol,
        "FastingBS" : fasting_bs,
        "MaxHR" : max_HR,
        "Oldpeak" : oldpeak,
        "Sex" : 1 if sex == "F" else 0,
        "ChestPainType_ATA" : 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP" : 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA" : 1 if chest_pain == "TA" else 0,
        "RestingECG_Normal" : 1 if resting_ecg == "Normal" else 0 ,
        "RestingECG_ST" : 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina": 1 if exercise_angina == "Y" else 0 ,
        "ST_Slope_Flat" : 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up" : 1 if st_slope == "Up" else 0,
    }
    
    input_df = pd.DataFrame([raw_input])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    numeric_columns = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]
    
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    
    prediction  = model.predict(input_df)[0]
    
    if prediction ==1:
        st.error("High Risk of Heart Disease 😈😈")
    else:
        st.success("Low Risk of Heart Disease 🥰🥰")
