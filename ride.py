import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('ride_demand_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🚕 Ride Demand Prediction")

# Example feature names — replace with actual ones
feature_names = ['hour', 'is_raining','traffic_level', 'surge_multiplier' ]

# Input fields
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    st.success(f"📊 Predicted Ride Demand: {prediction[0][0]:.2f}")
