import streamlit as st
import pandas as pd
import pickle

# 1. Set up the page configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Travel Customer Churn Prediction App")
st.markdown("Enter the customer's details below to predict if they are likely to churn (leave) or stay.")
st.markdown("---")

# 2. Load the trained machine learning model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Create a Sidebar for User Inputs
st.sidebar.header("📝 Customer Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 90, 30)
    
    frequent_flyer = st.sidebar.selectbox("Is a Frequent Flyer?", ["Yes", "No"])
    frequent_flyer_val = 1 if frequent_flyer == "Yes" else 0
    
    # Assuming Income Class is encoded as 1 to 5, adjust if your dataset is different
    income_class = st.sidebar.selectbox("Annual Income Class", [1, 2, 3, 4, 5])
    
    services_opted = st.sidebar.slider("Number of Services Opted", 1, 10, 2)
    
    social_media_synced = st.sidebar.selectbox("Account Synced to Social Media?", ["Yes", "No"])
    social_media_val = 1 if social_media_synced == "Yes" else 0
    
    booked_hotel = st.sidebar.selectbox("Booked Hotel Recently?", ["Yes", "No"])
    booked_hotel_val = 1 if booked_hotel == "Yes" else 0

    # PERFECTLY MATCHING THE JUPYTER NOTEBOOK ORDER
    data = {
        'Age': age,
        'FrequentFlyer': frequent_flyer_val,
        'AnnualIncomeClass': income_class,
        'ServicesOpted': services_opted,
        'AccountSyncedToSocialMedia': social_media_val,
        'BookedHotelOrNot': booked_hotel_val
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# 4. Store the user inputs
input_df = user_input_features()

# Display the user's inputs
st.subheader("Customer Data Summary")
st.write(input_df)

st.markdown("---")

# 5. Prediction Logic
if st.button("🔍 Predict Churn Risk", type="primary"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Churn!** This customer is likely to leave.")
        st.write(f"Confidence: **{prediction_proba[0][1] * 100:.2f}%**")
    else:
        st.success("✅ **Safe Customer!** This customer is likely to stay.")
        st.write(f"Confidence: **{prediction_proba[0][0] * 100:.2f}%**")
