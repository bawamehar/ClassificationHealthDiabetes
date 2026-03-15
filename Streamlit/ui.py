
import streamlit as st
import joblib
import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd

st.title("Diabetes Prediction App")

# Ensure src is importable (preprocess and dataload live under src/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / 'src'))
try:
    from dataload import read_data
    from preprocess import preprocess_data, transform_user_input, get_feature_names
except Exception:
    # if imports fail, show message and stop further UI actions
    st.error("Unable to import preprocessing helpers from src/. Make sure the project structure is intact.")

st.markdown("### Enter patient data (using the 7 original features)")

# Use the original 7 features used previously in UI/training
ui_features = ['HighBP', 'HighChol', 'BMI', 'GenHlth', 'Age', 'Income', 'Education']
user_values = {}
cols = st.columns(2)
user_values['HighBP'] = cols[0].selectbox('HighBP', [0, 1], index=1)
user_values['HighChol'] = cols[1].selectbox('HighChol', [0, 1], index=1)
user_values['BMI'] = float(cols[0].number_input('BMI', min_value=5.0, max_value=80.0, value=25.0, step=0.1))
user_values['GenHlth'] = float(cols[1].number_input('GenHlth (1=excellent to 5=poor)', min_value=1.0, max_value=5.0, value=3.0, step=1.0))
user_values['Age'] = float(cols[0].number_input('Age (category 1-13)', min_value=1.0, max_value=13.0, value=5.0, step=1.0))
user_values['Income'] = float(cols[1].number_input('Income (1-8)', min_value=1.0, max_value=20.0, value=4.0, step=1.0))
user_values['Education'] = float(cols[0].number_input('Education (1-6)', min_value=1.0, max_value=20.0, value=3.0, step=1.0))


if st.button("Predict"):
    # Build DataFrame only with the UI features (order doesn't matter, transform will align)
    user_df = pd.DataFrame([user_values])

    # Let preprocess.transform_user_input align and scale the input
    try:
        user_scaled = transform_user_input(user_df)
    except Exception as e:
        st.error(f"Error transforming input: {e}")
        user_scaled = user_df.values

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '../models/AdaBoostClassifier.joblib')
    model = joblib.load(model_path)

    # Ensure feature count matches what the model expects
    n_features = getattr(model, 'n_features_in_', None)
    if n_features is not None and user_scaled.shape[1] < n_features:
        user_scaled = np.pad(user_scaled, ((0, 0), (0, n_features - user_scaled.shape[1])), 'constant')
    elif n_features is not None and user_scaled.shape[1] > n_features:
        user_scaled = user_scaled[:, :n_features]

    prediction = model.predict(user_scaled)
    st.success(f"Predicted Diabetes: {'Yes' if int(prediction[0]) == 1 else 'No'}")

    # --- send the original 7 UI values to FastAPI to persist in the DB ---
    try:
        import requests
        api_payload = {
            'bmi': float(user_values['BMI']),
            'age': float(user_values['Age']),
            'genhlth': float(user_values['GenHlth']),
            'income': float(user_values['Income']),
            'highbp': float(user_values['HighBP']),
            'highchol': float(user_values['HighChol']),
            'education': float(user_values['Education'])
        }
        api_url = 'http://127.0.0.1:8000/predict'
        resp = requests.post(api_url, json=api_payload, timeout=5)
        if resp.ok:
            st.info('Saved patient to backend DB')
            # print to Streamlit server console as well
            print('Backend response:', resp.json())
        else:
            st.warning(f'Backend returned status {resp.status_code}: {resp.text}')
    except Exception as e:
        st.error(f'Could not send data to backend: {e}')

