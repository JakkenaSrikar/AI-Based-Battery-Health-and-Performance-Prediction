import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import LeakyReLU

model = tf.keras.models.load_model(
    "battery_prediction_model.h5",
    custom_objects={"LeakyReLU": LeakyReLU},
    compile=False
)

# Streamlit UI
st.title("EV Battery Performance Estimator")

# User-defined parameters
max_voltage = st.number_input("Maximum Battery Voltage (V)", min_value=0.0, value=58.8, format="%.2f")
min_voltage = st.number_input("Minimum Battery Voltage (V)", min_value=0.0, value=38.5, format="%.2f")
total_capacity = st.number_input("Full Charge Capacity (kWh)", min_value=0.0, value=3.183, format="%.3f")
charge_capacity = st.number_input("Charge Current Capacity (Ah)", min_value=0.0, value=54.5, format="%.1f")
range_factor = st.number_input("Range Factor (km)", min_value=0.0, value=150.0, format="%.1f")

# Input type selection
input_mode = st.radio("Select Input Mode", ('Manual Entry', 'Upload CSV'))

if input_mode == 'Manual Entry':
    # User-provided values
    voltage = st.number_input("Battery Voltage (V)", min_value=0.0, format="%.2f")
    current = st.number_input("Battery Current (A)", min_value=0.0, format="%.2f")
    resistance = st.number_input("Internal Resistance (Ω)", min_value=0.0, format="%.6f")
    
    if st.button("Estimate Performance"):
        # Format input data
        input_features = np.array([[voltage, current, resistance]])
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(input_features)
        
        # Model prediction
        estimated_values = model.predict(normalized_data)[0]
        soc_est, soh_est, duration_est, speed_est = estimated_values
        
        # Calculations
        open_circuit_voltage = voltage + (current * resistance)
        soc_actual = 100 * (open_circuit_voltage - min_voltage) / (max_voltage - min_voltage)
        available_energy = (soc_actual / 100) * total_capacity
        soh_actual = 100 * available_energy / total_capacity
        charge_duration = charge_capacity / current
        speed = range_factor / charge_duration
        
        # Display results
        st.subheader("Predicted Performance Metrics")
        st.write(f"State of Charge (SoC): {soc_actual:.2f}%")
        st.write(f"State of Health (SoH): {soh_actual:.2f}%")
        st.write(f"Estimated Duration: {charge_duration:.2f} hrs")
        st.write(f"Speed Estimate: {speed:.2f} km/hr")

elif input_mode == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload Battery Data (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_columns = {'Battery Voltage (V)', 'Battery Current (A)', 'Internal Resistance (Ω)'}
        
        if required_columns.issubset(df.columns):
            scaler = StandardScaler()
            scaled_inputs = scaler.fit_transform(df[list(required_columns)])
            predictions = model.predict(scaled_inputs)
            results = []
            
            for index, row in df.iterrows():
                voltage, current, resistance = row['Battery Voltage (V)'], row['Battery Current (A)'], row['Internal Resistance (Ω)']
                open_circuit_voltage = voltage + (current * resistance)
                soc_actual = 100 * (open_circuit_voltage - min_voltage) / (max_voltage - min_voltage)
                available_energy = (soc_actual / 100) * total_capacity
                soh_actual = 100 * available_energy / total_capacity
                charge_duration = charge_capacity / current
                speed = range_factor / charge_duration
                
                results.append({
                    "Battery Voltage (V)": voltage,
                    "Battery Current (A)": current,
                    "Internal Resistance (Ω)": resistance,
                    "SoC (%)": soc_actual,
                    "SoH (%)": soh_actual,
                    "Duration (hrs)": charge_duration,
                    "Speed (km/hr)": speed
                })
            
            results_df = pd.DataFrame(results)
            st.subheader("Predicted Data")
            st.dataframe(results_df)
            
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_data, "battery_estimates.csv", "text/csv")
        else:
            st.error("CSV must contain required columns: 'Battery Voltage (V)', 'Battery Current (A)', and 'Internal Resistance (Ω)'.")

st.write("To run on another device, execute:")
st.code("streamlit run battery_app.py --server.address 0.0.0.0 --server.port 8501", language="bash")


