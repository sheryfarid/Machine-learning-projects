import streamlit as st
import pickle
import numpy as np

import os


# Load model and scaler
model = pickle.load(open('diabetes_prediction/model.pkl', 'rb'))
scaler = pickle.load(open('diabetes_prediction/scaler.pkl', 'rb'))

st.title("Diabetes Prediction App")

st.write("Enter patient information below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    result = "🟥 Positive" if prediction[0] == 1 else "🟩 Negative"
    
    st.subheader(f"Prediction Result: {result}")
