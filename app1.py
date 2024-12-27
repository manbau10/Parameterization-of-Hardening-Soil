import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm

# Set page config to wide mode
st.set_page_config(layout="wide")

st.title('Hardening Soil Parameters Prediction')
st.image("Image 2.png")
st.markdown("<div style='text-align: center; color: gray; padding-bottom: 20px'>Conceptual workflow of the parameterization procedure (Geo-slope International Ltd)</div>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header('Basic Input Parameters')

# Input parameters in sidebar
poisson_ratio = st.sidebar.number_input('Poisson\'s Ratio', min_value=0.0, max_value=0.5, value=0.2)
void_ratio = st.sidebar.number_input('Initial void ratio', min_value=0.0, value=0.7)
unit_weight = st.sidebar.number_input('Unit weight (kN/m³)', min_value=0.0, value=18.0)
response_type = st.sidebar.selectbox('Response type', ['Drained'])
ocr = st.sidebar.number_input('O.C. ratio (1-3)', min_value=1.0, max_value=3.0, value=1.0)
k0nc = st.sidebar.selectbox('K0nc', ['Auto-Calculated'])

confidence_level = st.sidebar.selectbox(
    'Select Confidence Interval',
    ['80%', '90%', '95%']
)

# Data source selection
st.header('Data Input')
data_source = st.radio(
    "Choose your data source:",
    ["Use test data", "Upload your own data"]
)

# Input file requirements
st.markdown("""
### Input Data Requirements
The data should contain 102 rows and the following 6 columns:
1. Deviatoric Stress (q) - 100 kPa
2. Deviatoric Stress (q) - 200 kPa
3. Deviatoric Stress (q) - 400 kPa
4. Volumetric Strain - 100 kPa
5. Volumetric Strain - 200 kPa
6. Volumetric Strain - 400 kPa
""")

@st.cache_resource
def load_models():
    import os
    
    # Debug: Print current working directory
    st.write("Current working directory:", os.getcwd())
    
    # Custom optimizer configuration
    custom_objects = {
        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)
    }
    
    models = {}
    model_files = {
        'Eur_ref': 'Model_NC_1200_R2_Eur_ref_LSTM.h5',
        'E50_ref': 'Model_NC_1200_R3_E50_ref_LSTM.h5',
        'Eoed_ref': 'Model_NC_1200_R3_Eoed_ref_LSTM.h5',
        'm_Exponent': 'Model_NC_1200_R2_m_Exponent_LSTM.h5',
        'Cohesion': 'Model_NC_1200_R2_Cohesion_LSTM.h5',
        'InternalFrictionAngle': 'Model_NC_1200_R2_InternalFrictionAngle_LSTM.h5',
        'Rf': 'Model_NC_1200_R1_Rf_LSTM.h5'
    }
    
    # Debug: List files in current directory
    st.write("Files in current directory:", os.listdir())
    
    # Load models with custom object scope
    with tf.keras.utils.custom_object_scope(custom_objects):
        for param, file in model_files.items():
            # Debug: Print file path being attempted
            full_path = os.path.join(os.getcwd(), file)
            st.write(f"Attempting to load: {full_path}")
            
            # Check if file exists
            if not os.path.exists(full_path):
                st.error(f"File not found: {full_path}")
                continue
            
            try:
                model = tf.keras.models.load_model(full_path, compile=False)
                model.compile(optimizer='adam', loss='mse')
                models[param] = model
                st.write(f"Successfully loaded: {file}")
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
    
    return models

@st.cache_resource
def load_scalers():
    scaler_X = joblib.load('scaler_X.save')
    scalers_y = {
        'Eur_ref': joblib.load('scaler_y_Eur_ref.save'),
        'E50_ref': joblib.load('scaler_y_E50_ref.save'),
        'Eoed_ref': joblib.load('scaler_y_Eoed_ref.save'),
        'm_Exponent': joblib.load('scaler_y_m_Exponent.save'),
        'Cohesion': joblib.load('scaler_y_Cohesion.save'),
        'InternalFrictionAngle': joblib.load('scaler_y_InternalFrictionAngle.save'),
        'Rf': joblib.load('scaler_y_Rf.save')
    }
    return scaler_X, scalers_y

def calculate_confidence_intervals(predictions, std_errors, variable_names, selected_ci):
    z_values = {
        '80%': norm.ppf(0.90),
        '90%': norm.ppf(0.95),
        '95%': norm.ppf(0.975)
    }
    
    results = []
    z = z_values[selected_ci]
    
    for name, pred_value, std_error in zip(variable_names, predictions, std_errors):
        ci_lower = pred_value - z * std_error
        ci_upper = pred_value + z * std_error

        results.append({
            'Variable': name,
            'Predicted Value': pred_value,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper
        })
    
    return pd.DataFrame(results)

# Data loading section
if data_source == "Use test data":
    test_data_choice = st.selectbox(
        "Select test dataset:",
        ["Test Data 1", "Test Data 2", "Test Data 3", "Test Data 4"]
    )
    
    try:
        df = pd.read_excel(f"{test_data_choice}.xlsx")
        st.success(f"Successfully loaded {test_data_choice}")
        
        # Show data preview
        st.header('Data Preview')
        with st.expander("Click to view/hide data"):
            num_rows = st.number_input("Number of rows to display", min_value=5, max_value=102, value=10)
            st.dataframe(df.head(num_rows))
            st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            if st.checkbox("Show data statistics"):
                st.write("Basic statistics:")
                st.write(df.describe())
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        st.stop()

else:
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Show data preview
            st.header('Data Preview')
            with st.expander("Click to view/hide data"):
                num_rows = st.number_input("Number of rows to display", min_value=5, max_value=102, value=10)
                st.dataframe(df.head(num_rows))
                st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                if st.checkbox("Show data statistics"):
                    st.write("Basic statistics:")
                    st.write(df.describe())
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")
            st.stop()

# Prediction section
if 'df' in locals():
    if len(df) != 102 or len(df.columns) != 6:
        st.error("Input data must have exactly 102 rows and 6 columns.")
        st.stop()
        
    if st.button('Predict Parameters'):
        input_data = df.values
        additional_columns = np.array([[poisson_ratio]] * 102)
        X = np.hstack((input_data, additional_columns))
        
        # Load models and scalers
        models = load_models()
        scaler_X, scalers_y = load_scalers()
        
        with st.spinner('Making predictions...'):
            # Normalize input data
            X_normalized = scaler_X.transform(X)
            X_reshaped = X_normalized.reshape(1, 102, 7)
            
            # Make predictions for all parameters
            predictions = {}
            for param, model in models.items():
                pred_norm = model.predict(X_reshaped, verbose=0)
                pred = scalers_y[param].inverse_transform(pred_norm)
                predictions[param] = float(pred[0][0])

        # Define standard errors for each parameter
        std_errors = {
            'Eur_ref': 1000,
            'E50_ref': 1000,
            'Eoed_ref': 1000,
            'm_Exponent': 0.05,
            'Cohesion': 2,
            'InternalFrictionAngle': 1,
            'Rf': 0.02
        }

        results_df = calculate_confidence_intervals(
            list(predictions.values()),
            list(std_errors.values()),
            list(predictions.keys()),
            confidence_level
        )
        
        # Display results
        st.header('Basic Parameters')
        col1, col2 = st.columns(2)
        with col1:
            st.write('**Initial void ratio:**', void_ratio)
            st.write('**Unit weight:**', f"{unit_weight} kN/m³")
        with col2:
            st.write('**Response type:**', response_type)
            st.write('**O.C. ratio:**', ocr)

        st.header(f'Predicted Parameters with {confidence_level} Confidence Interval')
        
        # Stiffness Parameters
        st.subheader('Stiffness Parameters')
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Effective Poisson's Ratio:** {poisson_ratio:.4f}")
            
            for param in ['Eur_ref', 'E50_ref', 'Eoed_ref']:
                result = results_df.loc[results_df['Variable'] == param].iloc[0]
                st.write(f"**{param}:** {result['Predicted Value']:.0f} kPa")
                st.write(f"{confidence_level} CI: ({result['CI Lower']:.0f}, {result['CI Upper']:.0f}) kPa")
        
        with col2:
            result = results_df.loc[results_df['Variable'] == 'm_Exponent'].iloc[0]
            st.write(f"**m Exponent:** {result['Predicted Value']:.4f}")
            st.write(f"{confidence_level} CI: ({result['CI Lower']:.4f}, {result['CI Upper']:.4f})")
            
            st.write("**Reference Stress:** 100 kPa")

        # Strength Parameters
        st.subheader('Strength Parameters')
        col1, col2 = st.columns(2)
        
        with col1:
            for param in ['Cohesion', 'InternalFrictionAngle']:
                result = results_df.loc[results_df['Variable'] == param].iloc[0]
                st.write(f"**{param}:** {result['Predicted Value']:.2f}")
                st.write(f"{confidence_level} CI: ({result['CI Lower']:.2f}, {result['CI Upper']:.2f})")
        
        with col2:
            result = results_df.loc[results_df['Variable'] == 'Rf'].iloc[0]
            st.write(f"**Rf:** {result['Predicted Value']:.4f}")
            st.write(f"{confidence_level} CI: ({result['CI Lower']:.4f}, {result['CI Upper']:.4f})")
else:
    if data_source == "Upload your own data":
        st.write("Please upload an Excel file to begin.")