import streamlit as st
import numpy as np
from joblib import load

# Load the ANN model and scaler
try:
    loaded_objects = load('ann_model_and_scaler.joblib')  # Load the tuple
    model = loaded_objects[0]  # Extract the model from the tuple
    scaler = loaded_objects[1]  # Extract the scaler from the tuple
except Exception as e:
    st.error(f"Error loading model and scaler: {e}")
    model = None
    scaler = None

# Main function
def main():
    st.markdown(
    """
    <div style="text-align: center;">
        <h3>Out-of-Hospital Cardiac Arrest Prediction</h3>
        <p style="font-size: 14px;">Patient Information</p>
    </div>
    """,
    unsafe_allow_html=True,
)

    # Initialize session state for input fields if not already done
    if 'inputs' not in st.session_state:
        st.session_state['inputs'] = {
            "name": "",
            "age": 28,
            "gender": "Male",
            "chest_pain_type": "Typical Angina",
            "resting_blood_pressure": 92,
            "serum_cholesterol": 85,
            "fasting_blood_sugar": "Below 120",
            "ecg_result": "Normal",
            "max_heart_rate": 67,
            "exercise_angina": "No",
            "oldpeak": 0.0,
            "st_slope": "Upsloping"
        }

    if 'current_result' not in st.session_state:
        st.session_state['current_result'] = None

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['inputs']["age"] = st.number_input("Age", min_value=0, max_value=100, value=st.session_state['inputs']["age"], step=1)
        st.session_state['inputs']["gender"] = st.selectbox("Gender", options=["Male", "Female"], index=0 if st.session_state['inputs']["gender"] == "Male" else 1)
        st.session_state['inputs']["chest_pain_type"] = st.selectbox(
            "Chest Pain Type", 
            options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            index=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(st.session_state['inputs']["chest_pain_type"])
        )
        st.session_state['inputs']["resting_blood_pressure"] = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=200, value=st.session_state['inputs']["resting_blood_pressure"], step=1)
        st.session_state['inputs']["serum_cholesterol"] = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=529, value=st.session_state['inputs']["serum_cholesterol"], step=1)
        st.session_state['inputs']["fasting_blood_sugar"] = st.selectbox(
            "Fasting Blood Sugar",
            options=["Below 120", "Above 120"],
            index=0 if st.session_state['inputs']["fasting_blood_sugar"] == "Below 120" else 1
        )

    with col2:
        st.session_state['inputs']["ecg_result"] = st.selectbox(
            "Resting ECG",
            options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
            index=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(st.session_state['inputs']["ecg_result"])
        )
        st.session_state['inputs']["max_heart_rate"] = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=220, value=st.session_state['inputs']["max_heart_rate"], step=1)
        st.session_state['inputs']["exercise_angina"] = st.selectbox("Exercise Angina", options=["No", "Yes"], index=0 if st.session_state['inputs']["exercise_angina"] == "No" else 1)
        st.session_state['inputs']["oldpeak"] = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=st.session_state['inputs']["oldpeak"], step=0.1)
        st.session_state['inputs']["st_slope"] = st.selectbox("ST Slope", options=["Upsloping", "Flat", "Downsloping"], index=["Upsloping", "Flat", "Downsloping"].index(st.session_state['inputs']["st_slope"]))

    # Buttons
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        calculate_clicked = st.button("Calculate")
    with button_col2:
        reset_clicked = st.button("Reset All")

    # Calculate logic
    if calculate_clicked:
        if model and scaler:
            inputs = st.session_state['inputs']
            features = np.array([[inputs["age"], 1 if inputs["gender"] == "Male" else 0,
                                {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}[inputs["chest_pain_type"]], 
                                inputs["resting_blood_pressure"], inputs["serum_cholesterol"], 
                                0 if inputs["fasting_blood_sugar"] == "Below 120" else 1, 
                                {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[inputs["ecg_result"]], 
                                inputs["max_heart_rate"], 1 if inputs["exercise_angina"] == "Yes" else 0,
                                inputs["oldpeak"], {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[inputs["st_slope"]]]])

            # Apply the scaler to the features
            features_scaled = scaler.transform(features)

            # Predict using the ANN model
            prediction = model.predict(features_scaled).flatten()[0]

            if prediction >= 0.0131:
                st.session_state['current_result'] = f"The patient is at **high risk** of cardiac arrest."
            else:
                st.session_state['current_result'] = f"The patient is at **low risk** of cardiac arrest."

    # Display result
    if st.session_state['current_result']:
        st.write(st.session_state['current_result'])

    # Reset logic
    if reset_clicked:
        for key in st.session_state.keys():
         del st.session_state[key]
    st.query_params.clear()

if __name__ == "__main__":
    main()
