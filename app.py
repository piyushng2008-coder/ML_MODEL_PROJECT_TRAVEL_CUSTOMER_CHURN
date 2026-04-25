import streamlit as st
import pickle

# Set the title of the web app
st.title("Customer Churn Prediction App")
st.write("Welcome to my Machine Learning application!")

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: model.pkl not found. Please upload it.")

# A simple button to test the app
if st.button("Predict Churn"):
    st.write("The model is ready to make predictions!")