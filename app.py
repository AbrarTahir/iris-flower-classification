import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("iris_model.pkl")

st.set_page_config(page_title="🌸 Iris Flower Classifier", page_icon="🌼")
st.title("🌸 Iris Flower Prediction App")
st.write("Enter the flower measurements below to predict its species.")

# User input sliders
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 6.3)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.3)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 6.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 2.5)

# Predict button
if st.button("🔍 Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_class = prediction[0]
    st.success(f"🌼 Predicted Class: **{predicted_class}**")

st.info("Example Input → (6.3, 3.3, 6.0, 2.5) → Expected: Iris-virginica")
