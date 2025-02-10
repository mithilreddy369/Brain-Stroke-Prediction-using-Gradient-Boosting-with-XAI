import streamlit as st
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load the saved LGBM model
try:
    lgb_model = joblib.load('lgb_model.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to load and preprocess the dataset
def load_and_preprocess_data():
    try:
        data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    except FileNotFoundError:
        st.error("Error: /content/healthcare-dataset-stroke-data.csv not found. Please upload this file.")
        st.stop()
    
    # Drop rows with NaN values in the 'bmi' column
    data.dropna(subset=['bmi'], inplace=True)
    
    # Convert categorical features to numerical using predefined encoding
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    data['ever_married'] = data['ever_married'].map({'No': 0, 'Yes': 1})
    data['Residence_type'] = data['Residence_type'].map({'Rural': 0, 'Urban': 1})
    
    # Convert 'work_type' and 'smoking_status' to numerical using Label Encoding
    label_encoder = LabelEncoder()
    data['work_type'] = label_encoder.fit_transform(data['work_type'])
    data['smoking_status'] = label_encoder.fit_transform(data['smoking_status'])
    
    return data

# Function to decode categorical values back to their original form for better readability
def decode_input(input_data):
    gender_map = {0: 'Female', 1: 'Male', 2: 'Other'}
    ever_married_map = {0: 'No', 1: 'Yes'}
    work_type_map = {0: 'Govt_job', 1: 'Never_worked', 2: 'Private', 3: 'Self-employed', 4: 'Children'}
    residence_type_map = {0: 'Rural', 1: 'Urban'}
    smoking_status_map = {0: 'Unknown', 1: 'Formerly Smoked', 2: 'Never Smoked', 3: 'Smokes'}
    decoded_data = {
        'Gender': gender_map[input_data['gender'][0]],
        'Age': input_data['age'][0],
        'Hypertension': 'Yes' if input_data['hypertension'][0] == 1 else 'No',
        'Heart Disease': 'Yes' if input_data['heart_disease'][0] == 1 else 'No',
        'Ever Married': ever_married_map[input_data['ever_married'][0]],
        'Work Type': work_type_map[input_data['work_type'][0]],
        'Residence Type': residence_type_map[input_data['Residence_type'][0]],
        'Avg Glucose Level': input_data['avg_glucose_level'][0],
        'BMI': input_data['bmi'][0],
        'Smoking Status': smoking_status_map[input_data['smoking_status'][0]]
    }
    return decoded_data

# Load and preprocess the dataset
data = load_and_preprocess_data()

# Add Bootstrap CSS
st.markdown(
    """
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

# Create a form for user input
with st.form(key='my_form'):
    st.header("Brain Stroke Prediction")
    
    # Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male", "Other"], index=0)
    
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=80)
    
    with col3:
        hypertension = st.selectbox("Hypertension", options=["No", "Yes"], index=0)
    
    with col4:
        heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"], index=0)
    
    # Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ever_married = st.selectbox("Ever Married", options=["No", "Yes"], index=1)
    
    with col2:
        work_type = st.selectbox("Work Type", options=["Govt_job", "Never_worked", "Private", "Self-employed", "Children"], index=3)
    
    with col3:
        residence_type = st.selectbox("Residence Type", options=["Rural", "Urban"], index=0)
    
    with col4:
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=104.12)
    
    # Row 3
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=23.5)
    
    with col2:
        smoking_status = st.selectbox("Smoking Status", options=["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"], index=1)
    
    with col3:
        pass  # Empty column
    
    with col4:
        pass  # Empty column
    
    # Row 4 - Predict Button
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pass  # Empty column
    
    with col2:
        submit_button = st.form_submit_button(label='Predict', use_container_width=True)
    
    with col3:
        pass  # Empty column
    
    with col4:
        pass  # Empty column

# Map input values to numerical values
gender_map = {"Female": 0, "Male": 1, "Other": 2}
hypertension_map = {"No": 0, "Yes": 1}
heart_disease_map = {"No": 0, "Yes": 1}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "Children": 4}
residence_type_map = {"Rural": 0, "Urban": 1}
smoking_status_map = {"Unknown": 0, "Formerly Smoked": 1, "Never Smoked": 2, "Smokes": 3}

input_data = {
    'gender': [gender_map[gender]],
    'age': [age],
    'hypertension': [hypertension_map[hypertension]],
    'heart_disease': [heart_disease_map[heart_disease]],
    'ever_married': [ever_married_map[ever_married]],
    'work_type': [work_type_map[work_type]],
    'Residence_type': [residence_type_map[residence_type]],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status_map[smoking_status]]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Get the numerical columns from the original data
numerical_cols = data.select_dtypes(include=np.number).columns.tolist()

# Exclude 'stroke' and 'id' from the numerical columns if they exist
numerical_cols = [col for col in numerical_cols if col not in ['stroke', 'id']]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the numerical columns of the original data
scaler.fit(data[numerical_cols])

# Transform the input data using the fitted scaler
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Make a prediction using the loaded model
if submit_button:
    try:
        prediction_proba = lgb_model.predict_proba(input_df)
        predicted_class = lgb_model.predict(input_df)
        st.subheader("Prediction Results:")
        st.write(f"Prediction Probabilities: {prediction_proba}")
        
        if predicted_class[0] == 1:
            st.markdown('<h3 style="color:red;">Predicted Class: Stroke</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color:green;">Predicted Class: No Stroke</h3>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
