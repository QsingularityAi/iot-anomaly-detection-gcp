import streamlit as st
import pandas as pd
from google.cloud import bigquery
import json
import os
import subprocess

# Set up BigQuery client
# Ensure you have authenticated to GCP, e.g., by running `gcloud auth application-default login`
client = bigquery.Client()

# --- Data Visualization ---
st.title("IoT Anomaly Detection Dashboard")

st.header("Sensor Data Visualization")

# Query to get sensor data (example: last 24 hours)
# Replace `your_project_id.iot_anomaly_detection.sensor_data` with your actual table path
# Get project ID from environment or use current gcloud config
import os
import subprocess

try:
    PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT') or subprocess.check_output(['gcloud', 'config', 'get-value', 'project'], text=True).strip()
except:
    PROJECT_ID = 'zeltask-staging'  # fallback

query_sensor_data = f"""
SELECT
    timestamp,
    device_id,
    temperature,
    humidity,
    pressure,
    vibration_level,
    power_consumption,
    is_anomaly
FROM
    `{PROJECT_ID}.iot_anomaly_detection.sensor_data`
WHERE
    timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
ORDER BY
    timestamp DESC
"""

@st.cache_data
def load_sensor_data():
    query_job = client.query(query_sensor_data)
    return query_job.to_dataframe()

try:
    sensor_data_df = load_sensor_data()
    st.write("Recent Sensor Data:")
    st.dataframe(sensor_data_df)

    # Example visualization: Temperature over time for a specific device
    device_id_to_plot = st.selectbox("Select Device ID to Visualize", sensor_data_df['device_id'].unique())
    if device_id_to_plot:
        device_data_df = sensor_data_df[sensor_data_df['device_id'] == device_id_to_plot]
        st.line_chart(device_data_df.set_index('timestamp')[['temperature', 'humidity', 'pressure']])

except Exception as e:
    st.error(f"Error loading sensor data: {e}")
    st.warning(f"Error connecting to BigQuery table {PROJECT_ID}.iot_anomaly_detection.sensor_data. Please ensure the table exists and you have proper permissions.")


# --- Anomaly Alerts ---
st.header("Anomaly Alerts")

# Query to get recent anomaly alerts (example: last 7 days)
# Replace `your_project_id.iot_anomaly_detection.anomaly_alerts` with your actual table path
query_anomaly_alerts = f"""
SELECT
    alert_timestamp,
    device_id,
    anomaly_type,
    severity,
    anomaly_score,
    alert_status
FROM
    `{PROJECT_ID}.iot_anomaly_detection.anomaly_alerts`
WHERE
    alert_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY
    alert_timestamp DESC
"""

@st.cache_data
def load_anomaly_alerts():
    query_job = client.query(query_anomaly_alerts)
    return query_job.to_dataframe()

try:
    anomaly_alerts_df = load_anomaly_alerts()
    st.write("Recent Anomaly Alerts:")
    st.dataframe(anomaly_alerts_df)

except Exception as e:
    st.error(f"Error loading anomaly alerts: {e}")
    st.warning(f"Error connecting to BigQuery table {PROJECT_ID}.iot_anomaly_detection.anomaly_alerts. Please ensure the table exists and you have proper permissions.")

st.markdown("""
---
**Note:** This is a basic dashboard. For real-time alerts, you would typically set up a mechanism to listen to the `anomaly-alerts` Pub/Sub topic and update the dashboard dynamically. This might involve using Streamlit's `st.experimental_rerun()` or a separate background process.
""")
