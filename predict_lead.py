import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

# Load the trained model and scaler
model = joblib.load('model/model_xgboost.pkl')
scaler = joblib.load('model/scaler.pkl')

# Streamlit UI for input collection
st.title("Lead Data Classification with XGBoost")

# Lead Status Mapping
lead_status_mapping = {'baru': 1, 'mencoba menghubungi': 2, 'dihubungi': 3, 'sukses': 4, 'diskualifikasi': 5}
lead_status_input = st.selectbox("Select Lead Status", options=list(lead_status_mapping.keys()))
lead_status = lead_status_mapping[lead_status_input]

# Other inputs
response_time = st.number_input("Enter Response Time (in seconds)", min_value=0.0, step=0.1)
interaction_level = st.number_input("Enter Interaction Level (scale 1-10)", min_value=0.0, max_value=10.0, step=0.1)
source = st.selectbox("Select Source", options=['ad', 'email', 'referral', 'search engine', 'social media'])

# Prepare the input data for prediction
input_data = {
    'lead_status': lead_status,
    'response_time': response_time,
    'interaction_level': interaction_level,
    'source': source
}

input_df = pd.DataFrame([input_data])

# Encode categorical features
input_df = pd.get_dummies(input_df)
required_columns = ['response_time', 'interaction_level', 'lead_status', 'source_ad', 'source_email', 'source_referral', 'source_search engine', 'source_social media']

for col in required_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure the order of columns is correct
input_df = input_df[required_columns]

# Standardize numerical features
input_df[['response_time', 'interaction_level', 'lead_status']] = scaler.transform(input_df[['response_time', 'interaction_level', 'lead_status']])

# Make the prediction when the user clicks "Predict"
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the result in the Streamlit interface
    st.write(f"Predicted Class: {int(prediction[0])}")
    st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")
