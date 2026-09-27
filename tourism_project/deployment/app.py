
import streamlit as st
import pandas as pd
import joblib
import os

# Define model path (absolute path from repository root)
MODEL_PATH = "tourism_project/deployment/xgboost_model.pkl"

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        return None

model = load_model()

st.title("Tourism Package Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

if model:
    st.header("Customer Information")

    # Input fields for customer features
    age = st.slider("Age", 18, 70, 30)
    typeofcontact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government Sector'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    numberofpersonvisiting = st.slider("Number of Persons Visiting", 1, 6, 2)
    preferredpropertystar = st.slider("Preferred Property Star Rating", 3, 5, 4)
    maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    numberoftrips = st.slider("NumberOfTrips Annually", 0, 20, 5)
    passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    owncar = st.selectbox("Owns Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    numberofchildrenvisiting = st.slider("Number of Children Visiting", 0, 3, 0)
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Executive', 'AVP', 'VP', 'Director'])
    monthlyincome = st.number_input("Monthly Income", 1000, 100000, 25000)
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    productpitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    numberoffollowups = st.slider("Number of Follow-ups", 0, 6, 2)
    durationofpitch = st.slider("Duration of Pitch (minutes)", 5, 60, 20)

    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [typeofcontact],
        'CityTier': [citytier],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [numberofpersonvisiting],
        'PreferredPropertyStar': [preferredpropertystar],
        'MaritalStatus': [maritalstatus],
        'NumberOfTrips': [numberoftrips],
        'Passport': [passport],
        'OwnCar': [owncar],
        'NumberOfChildrenVisiting': [numberofchildrenvisiting],
        'Designation': [designation],
        'MonthlyIncome': [monthlyincome],
        'PitchSatisfactionScore': [pitchsatisfactionscore],
        'ProductPitched': [productpitched],
        'NumberOfFollowups': [numberoffollowups],
        'DurationOfPitch': [durationofpitch]
    })

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[:, 1][0]

        if prediction == 1:
            st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba:.2f})")
        else:
            st.info(f"The customer is not likely to purchase the package. (Probability: {prediction_proba:.2f})")
else:
    st.warning("Model could not be loaded. Please ensure the training step was successful.")
