import streamlit as st
import pickle
import numpy as np

with open('knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Heart Disease Prediction")
st.write("Enter patient's details below:")

age = st.number_input("Age", min_value = 1, max_value = 120, value = 30)
sex = st.number_input("Sex", min_value = 0, max_value = 1, value = 0)
cp = st.number_input("Chest Pain Type", min_value=0, value=0)
trestbps = st.number_input("Resting Blood Pressure", min_value=0, value=120)
chol = st.number_input("Cholesterol", min_value=0, value=200)
fbs = st.number_input("Fasting Blood Sugar", min_value=0, max_value=1, value=0)
restecg = st.number_input("Resting ECG", min_value=0, value=0)
thalach = st.number_input("Maximum Heart Rate", min_value=0, value=150)
exang = st.number_input("Exercise Induced Angina", min_value=0, max_value=1, value=0)
oldpeak = st.number_input("Oldpeak", min_value=0.0, value=0.0)
slope = st.number_input("Slope", min_value=0, value=0)
ca = st.number_input("Number of Major Vessels", min_value=0, value=0)
thal = st.number_input("Thal", min_value=0, value=0)

if st.button("Predict"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]])

    input_data_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error("Heart Disease Detected!")
    else:
        st.success("No Heart Disease Detected!")