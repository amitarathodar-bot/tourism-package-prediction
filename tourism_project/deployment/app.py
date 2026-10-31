import streamlit as st
import pandas as pd
import joblib
import os

st.title("Tourism Package Purchase Prediction App")
st.write("Adjust the parameters below to predict if a customer will buy the tourism package (ProdTaken = Yes).")

# Load pre-trained joblib model
model_path = "best_model.joblib"
if not os.path.exists(model_path):
    model_path = "tourism_project/deployment/best_model.joblib"

try:
    model = joblib.load(model_path)
    st.success("Model successfully loaded!")
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

st.header("Customer Profile Input Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"], index=0)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    duration_of_pitch = st.slider("Duration of Pitch (mins)", 0, 120, 15)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"], index=0)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    num_person_visiting = st.slider("Number of Persons Visiting", 1, 10, 3)
    num_followups = st.slider("Number of Follow-ups", 1, 10, 4)
    product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"], index=1)

with col2:
    pref_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5], index=0)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"], index=0)
    num_trips = st.slider("Number of Trips", 1, 20, 3)
    passport = st.selectbox("Has Passport?", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")  # Buyers typically have a passport
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car?", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    num_children = st.slider("Number of Children Visiting", 0, 5, 1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], index=0)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=22000)

# Map UI inputs into the exact format expected by the model preprocessing pipeline
input_dict = {
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": pref_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}

input_df = pd.DataFrame([input_dict])

# Ensure feature column names align exactly with model expectations if defined
if hasattr(model, "feature_names_in_"):
    expected_cols = model.feature_names_in_
    # Generate aligned dataset matching training's one-hot encoded structure
    input_encoded = pd.get_dummies(input_df)
    # Add missing encoded columns as zeros
    for col in expected_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0.0
    # Order columns matching training
    input_final = input_encoded[expected_cols]
else:
    input_final = input_df

st.subheader("Parameters compiled for inference:")
st.dataframe(input_final)

if st.button("Predict Package Purchase"):
    try:
        prediction = model.predict(input_final)
        probability = model.predict_proba(input_final)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Inference Result:")
        if prediction[0] == 1:
            st.success(f"Prediction: Customer WILL purchase the package! \U0001F389 (Probability: {probability:.2%})")
        else:
            st.warning(f"Prediction: Customer is UNLIKELY to purchase the package. (Probability: {probability:.2%})")
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
