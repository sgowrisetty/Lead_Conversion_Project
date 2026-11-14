import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_random_forest_model.pkl")

st.title("Lead Conversion Prediction")

age = st.number_input("Age", 18, 70, 30)

current_occupation = st.selectbox(
    "Current Occupation",
    ["Professional","Student","Unemployed"]
)

first_interaction = st.selectbox(
    "First Interaction",
    ["Website","Mobile App"]
)

profile_completed = st.selectbox(
    "Profile Completed",
    ["Low","Medium","High"]
)

website_visits = st.number_input(
    "Website Visits",
    min_value=0
)

time_spent_on_website = st.number_input(
    "Time Spent On Website",
    min_value=0
)

page_views_per_visit = st.number_input(
    "Page Views Per Visit",
    min_value=0.0
)

last_activity = st.selectbox(
    "Last Activity",
    ["Email Activity","Phone Activity","Website Activity"]
)

print_media_type1 = st.selectbox(
    "Newspaper Ad",
    ["Yes","No"]
)

print_media_type2 = st.selectbox(
    "Magazine Ad",
    ["Yes","No"]
)

digital_media = st.selectbox(
    "Digital Media",
    ["Yes","No"]
)

educational_channels = st.selectbox(
    "Educational Channels",
    ["Yes","No"]
)

referral = st.selectbox(
    "Referral",
    ["Yes","No"]
)

if st.button("Predict"):

    data = pd.DataFrame({
        "age":[age],
        "current_occupation":[current_occupation],
        "first_interaction":[first_interaction],
        "profile_completed":[profile_completed],
        "website_visits":[website_visits],
        "time_spent_on_website":[time_spent_on_website],
        "page_views_per_visit":[page_views_per_visit],
        "last_activity":[last_activity],
        "print_media_type1":[print_media_type1],
        "print_media_type2":[print_media_type2],
        "digital_media":[digital_media],
        "educational_channels":[educational_channels],
        "referral":[referral]
    })

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    st.write("Prediction:", prediction)
    st.write("Probability:", round(probability,4))