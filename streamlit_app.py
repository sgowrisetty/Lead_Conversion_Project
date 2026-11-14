!pip install streamlit

import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("final_random_forest_model.pkl")
    st.success("Model loaded successfully")
except Exception as e:
    st.error(str(e))
    raise

st.title("Lead Conversion Prediction System")

age = st.number_input("Age", min_value=18, max_value=70, value=30)

website_visits = st.number_input(
    "Website Visits",
    min_value=0,
    value=5
)

time_spent_on_website = st.number_input(
    "Time Spent on Website",
    min_value=0,
    value=500
)

page_views_per_visit = st.number_input(
    "Page Views Per Visit",
    min_value=0.0,
    value=5.0
)

if st.button("Predict"):

    sample_data = pd.DataFrame({
        "age": [age],
        "website_visits": [website_visits],
        "time_spent_on_website": [time_spent_on_website],
        "page_views_per_visit": [page_views_per_visit]
    })

    prediction = model.predict(sample_data)[0]

    if prediction == 1:
        st.success("Lead Likely to Convert")
    else:
        st.error("Lead Unlikely to Convert")