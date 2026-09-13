import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

st.title("EV Purchase Predictor")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (USD)", min_value=0, value=60000)
commute = st.number_input("Daily Commute (km)", min_value=0.0, value=15.0)
cars = st.number_input("Number of Cars Owned", min_value=0, value=1)
charge_home = st.number_input("Charging Stations Near Home", min_value=0, value=3)
charge_work = st.number_input("Charging Stations Near Work", min_value=0, value=3)
env_concern = st.slider("Environmental Concern Level", 1, 5, 3)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
city = st.selectbox("City Type", ["Urban", "Suburban", "Rural"])
car_type = st.selectbox("Current Car Type", ["Sedan", "SUV", "Hatchback", "Truck"])
home_charge = st.selectbox("Home Charging Possible", ["Yes", "No"])
subsidy = st.selectbox("Subsidy Available", ["Yes", "No"])
anxiety = st.selectbox("Range Anxiety Level", ["Low", "Medium", "High"])

if st.button("Predict"):
    payload = {
        "Age": age, "Annual_Income_USD": income, "Daily_Commute_km": commute,
        "Number_of_Cars_Owned": cars, "Charging_Stations_Near_Home": charge_home,
        "Charging_Stations_Near_Work": charge_work, "Environmental_Concern_Level": env_concern,
        "Gender": gender, "City_Type": city, "Current_Car_Type": car_type,
        "Home_Charging_Possible": home_charge, "Subsidy_Available": subsidy,
        "Range_Anxiety_Level": anxiety
    }

    response = requests.post(API_URL, json=payload)

    probability = response.json()["probability"]
    st.metric("Purchase Probability", f"{probability:.1%}")