
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Page title
st.set_page_config(page_title="AI Procrastination Detector", layout="centered")
st.title("🧠 AI Procrastination Detector")
st.write("Detect whether a user is procrastinating using Machine Learning")

# Sample dataset
data = {
    'screen_time_hours': [2, 5, 8, 6, 1, 7, 9],
    'social_media_hours': [0.5, 3, 5, 4, 0.2, 4.5, 6],
    'idle_time_hours': [0.2, 1.5, 3, 2, 0.1, 2.5, 3.5],
    'task_completed': [5, 2, 1, 2, 6, 1, 0],
    'procrastination': [0, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Split data
X = df.drop('procrastination', axis=1)
y = df['procrastination']

# Train model
model = LogisticRegression()
model.fit(X, y)

# User input
st.subheader("📊 Enter User Activity Details")

screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 4.0)
social_media = st.slider("Social Media Usage (hours)", 0.0, 8.0, 2.0)
idle_time = st.slider("Idle Time (hours)", 0.0, 6.0, 1.0)
tasks_done = st.slider("Tasks Completed", 0, 10, 3)

# Prediction
if st.button("Detect Procrastination"):
    user_data = np.array([[screen_time, social_media, idle_time, tasks_done]])
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.error("😴 User is Procrastinating")
    else:
        st.success("🚀 User is Productive")

st.markdown("---")
st.caption("AICTE Internship Project | AI Procrastination Detector")