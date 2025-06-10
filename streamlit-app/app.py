import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os
import subprocess
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="IoT Anomaly Detection Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b6b;
    }
    .anomaly-alert {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4757;
    }
</style>
""", unsafe_allow_html=True)

# Initialize BigQuery client
@st.cache_resource
def init_bigquery_client():
    try:
        return bigquery.Client()
    except Exception as e:
        st.error(f"Failed to initialize BigQuery client: {e}")
        return None

client = init_bigquery_client()

# Get project ID
@st.cache_data
def get_project_id():
    try:
        return os.getenv('GOOGLE_CLOUD_PROJECT') or \
               os.getenv('PROJECT_ID') or \
               subprocess.check_output(['gcloud', 'config', 'get-value', 'project'], text=True).strip()
    except:
        return 'zeltask-staging'  # fallback

PROJECT_ID = get_project_id()

# Main title
st.title("🌡️ IoT Temperature & Vibration Anomaly Detection Dashboard")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)

# Convert time range to hours
time_mapping = {
    "Last 1 Hour": 1,
    "Last 6 Hours": 6,
    "Last 24 Hours": 24,
    "Last 7 Days": 168,
    "Last 30 Days": 720
}
hours = time_mapping[time_range]

# Device filter
device_filter = st.sidebar.text_input("Filter by Device ID (optional)", "")

# Updated query for temperature and vibration data only
query_sensor_data = f"""
SELECT
    timestamp,
    device_id,
    device_type,
    temperature,
    vibration_level as vibration,
    is_anomaly,
    -- Calculate z-scores for anomaly detection
    (temperature - AVG(temperature) OVER (PARTITION BY device_id ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)) / 
    NULLIF(STDDEV(temperature) OVER (PARTITION BY device_id ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND CURRENT ROW), 0) as temp_z_score,
    (vibration_level - AVG(vibration_level) OVER (PARTITION BY device_id ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)) / 
    NULLIF(STDDEV(vibration_level) OVER (PARTITION BY device_id ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND CURRENT ROW), 0) as vibration_z_score
FROM
    `{PROJECT_ID}.iot_anomaly_detection.sensor_data`
WHERE
    timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    {f"AND device_id LIKE '%{device_filter}%'" if device_filter else ""}
ORDER BY
    timestamp DESC
"""

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sensor_data():
    if client is None:
        return pd.DataFrame()
    
    try:
        query_job = client.query(query_sensor_data)
        df = query_job.to_dataframe()
        
        if df.empty:
            st.warning("No data found for the selected time range.")
            return pd.DataFrame()
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except NotFound:
        st.error(f"Table `{PROJECT_ID}.iot_anomaly_detection.sensor_data` not found. Please ensure the table exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load and display data
try:
    sensor_data_df = load_sensor_data()
    
    if not sensor_data_df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_readings = len(sensor_data_df)
            st.metric("📊 Total Readings", total_readings)
        
        with col2:
            anomaly_count = sensor_data_df['is_anomaly'].sum() if 'is_anomaly' in sensor_data_df.columns else 0
            anomaly_rate = (anomaly_count / total_readings * 100) if total_readings > 0 else 0
            st.metric("🚨 Anomalies", f"{anomaly_count} ({anomaly_rate:.1f}%)")
        
        with col3:
            unique_devices = sensor_data_df['device_id'].nunique() if 'device_id' in sensor_data_df.columns else 0
            st.metric("🔧 Active Devices", unique_devices)
        
        with col4:
            avg_temp = sensor_data_df['temperature'].mean() if 'temperature' in sensor_data_df.columns else 0
            st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C")
        
        st.markdown("---")
        
        # Time series visualization
        st.header("📈 Time Series Analysis")
        
        # Create subplots for temperature and vibration
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Over Time', 'Vibration Over Time'),
            vertical_spacing=0.1
        )
        
        # Temperature plot
        normal_temp = sensor_data_df[sensor_data_df['is_anomaly'] == False] if 'is_anomaly' in sensor_data_df.columns else sensor_data_df
        anomaly_temp = sensor_data_df[sensor_data_df['is_anomaly'] == True] if 'is_anomaly' in sensor_data_df.columns else pd.DataFrame()
        
        # Add normal temperature data
        fig.add_trace(
            go.Scatter(
                x=normal_temp['timestamp'],
                y=normal_temp['temperature'],
                mode='lines+markers',
                name='Normal Temperature',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Add anomaly temperature data
        if not anomaly_temp.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_temp['timestamp'],
                    y=anomaly_temp['temperature'],
                    mode='markers',
                    name='Temperature Anomalies',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=1, col=1
            )
        
        # Vibration plot
        normal_vib = sensor_data_df[sensor_data_df['is_anomaly'] == False] if 'is_anomaly' in sensor_data_df.columns else sensor_data_df
        anomaly_vib = sensor_data_df[sensor_data_df['is_anomaly'] == True] if 'is_anomaly' in sensor_data_df.columns else pd.DataFrame()
        
        # Add normal vibration data
        fig.add_trace(
            go.Scatter(
                x=normal_vib['timestamp'],
                y=normal_vib['vibration'],
                mode='lines+markers',
                name='Normal Vibration',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Add anomaly vibration data
        if not anomaly_vib.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_vib['timestamp'],
                    y=anomaly_vib['vibration'],
                    mode='markers',
                    name='Vibration Anomalies',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Vibration Level", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution analysis
        st.header("📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature distribution
            fig_temp = px.histogram(
                sensor_data_df, 
                x='temperature', 
                color='is_anomaly' if 'is_anomaly' in sensor_data_df.columns else None,
                title='Temperature Distribution',
                nbins=30,
                color_discrete_map={True: 'red', False: 'blue'}
            )
            fig_temp.update_layout(height=400)
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            # Vibration distribution
            fig_vib = px.histogram(
                sensor_data_df, 
                x='vibration', 
                color='is_anomaly' if 'is_anomaly' in sensor_data_df.columns else None,
                title='Vibration Distribution',
                nbins=30,
                color_discrete_map={True: 'red', False: 'blue'}
            )
            fig_vib.update_layout(height=400)
            st.plotly_chart(fig_vib, use_container_width=True)
        
        # Correlation analysis
        st.header("🔗 Correlation Analysis")
        
        # Temperature vs Vibration scatter plot
        fig_scatter = px.scatter(
            sensor_data_df,
            x='temperature',
            y='vibration',
            color='is_anomaly' if 'is_anomaly' in sensor_data_df.columns else None,
            title='Temperature vs Vibration Correlation',
            color_discrete_map={True: 'red', False: 'blue'},
            hover_data=['device_id', 'timestamp']
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Z-score analysis if available
        if 'temp_z_score' in sensor_data_df.columns and 'vibration_z_score' in sensor_data_df.columns:
            st.header("📏 Z-Score Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_z_temp = px.histogram(
                    sensor_data_df,
                    x='temp_z_score',
                    title='Temperature Z-Score Distribution',
                    nbins=30
                )
                fig_z_temp.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="Threshold (+2)")
                fig_z_temp.add_vline(x=-2, line_dash="dash", line_color="red", annotation_text="Threshold (-2)")
                st.plotly_chart(fig_z_temp, use_container_width=True)
            
            with col2:
                fig_z_vib = px.histogram(
                    sensor_data_df,
                    x='vibration_z_score',
                    title='Vibration Z-Score Distribution',
                    nbins=30
                )
                fig_z_vib.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="Threshold (+2)")
                fig_z_vib.add_vline(x=-2, line_dash="dash", line_color="red", annotation_text="Threshold (-2)")
                st.plotly_chart(fig_z_vib, use_container_width=True)
        
        # Device-wise analysis
        if 'device_id' in sensor_data_df.columns:
            st.header("🔧 Device-wise Analysis")
            
            device_stats = sensor_data_df.groupby('device_id').agg({
                'temperature': ['mean', 'std', 'min', 'max'],
                'vibration': ['mean', 'std', 'min', 'max'],
                'is_anomaly': 'sum' if 'is_anomaly' in sensor_data_df.columns else 'count'
            }).round(2)
            
            device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns.values]
            st.dataframe(device_stats, use_container_width=True)
        
        # Recent anomalies
        if 'is_anomaly' in sensor_data_df.columns:
            recent_anomalies = sensor_data_df[sensor_data_df['is_anomaly'] == True].head(10)
            if not recent_anomalies.empty:
                st.header("🚨 Recent Anomalies")
                st.dataframe(recent_anomalies[['timestamp', 'device_id', 'temperature', 'vibration']], use_container_width=True)
        
        # Raw data table
        st.header("📋 Raw Data")
        st.dataframe(sensor_data_df.head(100), use_container_width=True)
        
        # Download data
        csv = sensor_data_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'iot_sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
    
    else:
        st.info("No data available for the selected time range. Please check your BigQuery table or adjust the time range.")

except Exception as e:
    st.error(f"An error occurred while loading the dashboard: {e}")
    st.info("Please ensure that:")
    st.markdown("""
    - BigQuery table `{PROJECT_ID}.iot_anomaly_detection.sensor_data` exists
    - You have proper authentication set up
    - The table contains data for the selected time range
    """)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Information")
st.sidebar.info(f"**Project ID:** {PROJECT_ID}")
st.sidebar.info(f"**Time Range:** {time_range}")
if device_filter:
    st.sidebar.info(f"**Device Filter:** {device_filter}")

# Auto-refresh option
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)")
if auto_refresh:
    import time
    time.sleep(30)
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
### 📝 Dashboard Information
- **Real-time Data**: Data is refreshed every 5 minutes (cached)
- **Anomaly Detection**: Based on statistical analysis and ML models
- **Data Source**: BigQuery table `{PROJECT_ID}.iot_anomaly_detection.sensor_data`
- **Sensors**: Temperature and Vibration monitoring

For technical support or questions, please contact your system administrator.
""".format(PROJECT_ID=PROJECT_ID))
