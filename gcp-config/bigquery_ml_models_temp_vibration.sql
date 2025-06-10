-- BigQuery ML Models for Temperature & Vibration Anomaly Detection
-- Updated schema for temperature and vibration data only

-- Create dataset if it doesn't exist
CREATE SCHEMA IF NOT EXISTS `iot_anomaly_detection`
OPTIONS(
  description="IoT Anomaly Detection for Temperature and Vibration Data",
  location="US"
);

-- Note: The main sensor_data table is created by the Terraform configuration
-- This table structure should match what's defined in main.tf

-- Create training data view for ML models
CREATE OR REPLACE VIEW `iot_anomaly_detection.training_data_temp_vibration` AS
SELECT 
  device_id,
  timestamp,
  device_type,
  temperature,
  vibration_level as vibration,
  -- Feature engineering
  temperature - LAG(temperature, 1) OVER (PARTITION BY device_id ORDER BY timestamp) as temp_change,
  vibration_level - LAG(vibration_level, 1) OVER (PARTITION BY device_id ORDER BY timestamp) as vibration_change,
  -- Rolling averages (5-minute window)
  AVG(temperature) OVER (
    PARTITION BY device_id 
    ORDER BY timestamp 
    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
  ) as temp_avg_5min,
  AVG(vibration_level) OVER (
    PARTITION BY device_id 
    ORDER BY timestamp 
    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
  ) as vibration_avg_5min,
  -- Standard deviations
  STDDEV(temperature) OVER (
    PARTITION BY device_id 
    ORDER BY timestamp 
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) as temp_stddev_10min,
  STDDEV(vibration_level) OVER (
    PARTITION BY device_id 
    ORDER BY timestamp 
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) as vibration_stddev_10min,
  -- Target variable
  is_anomaly as label
FROM `iot_anomaly_detection.sensor_data`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);

-- 1. Linear Regression Model for Temperature Prediction
CREATE OR REPLACE MODEL `iot_anomaly_detection.temperature_prediction_model`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['temperature'],
  auto_class_weights=true
) AS
SELECT
  device_type,
  EXTRACT(HOUR FROM timestamp) as hour_of_day,
  EXTRACT(DAYOFWEEK FROM timestamp) as day_of_week,
  vibration_level as vibration,
  temp_avg_5min,
  vibration_avg_5min,
  temperature
FROM `iot_anomaly_detection.training_data_temp_vibration`
WHERE 
  temp_avg_5min IS NOT NULL 
  AND vibration_avg_5min IS NOT NULL
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY);

-- 2. Linear Regression Model for Vibration Prediction
CREATE OR REPLACE MODEL `iot_anomaly_detection.vibration_prediction_model`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['vibration'],
  auto_class_weights=true
) AS
SELECT
  device_type,
  EXTRACT(HOUR FROM timestamp) as hour_of_day,
  EXTRACT(DAYOFWEEK FROM timestamp) as day_of_week,
  temperature,
  temp_avg_5min,
  vibration_avg_5min,
  vibration_level as vibration
FROM `iot_anomaly_detection.training_data_temp_vibration`
WHERE 
  temp_avg_5min IS NOT NULL 
  AND vibration_avg_5min IS NOT NULL
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY);

-- 3. Logistic Regression Model for Anomaly Classification
CREATE OR REPLACE MODEL `iot_anomaly_detection.anomaly_classification_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label'],
  auto_class_weights=true
) AS
SELECT
  device_type,
  temperature,
  vibration_level as vibration,
  temp_change,
  vibration_change,
  temp_avg_5min,
  vibration_avg_5min,
  temp_stddev_10min,
  vibration_stddev_10min,
  -- Z-scores for anomaly detection
  ABS(temperature - temp_avg_5min) / NULLIF(temp_stddev_10min, 0) as temp_z_score,
  ABS(vibration - vibration_avg_5min) / NULLIF(vibration_stddev_10min, 0) as vibration_z_score,
  label
FROM `iot_anomaly_detection.training_data_temp_vibration`
WHERE 
  temp_avg_5min IS NOT NULL 
  AND vibration_avg_5min IS NOT NULL
  AND temp_stddev_10min IS NOT NULL
  AND vibration_stddev_10min IS NOT NULL
  AND temp_stddev_10min > 0
  AND vibration_stddev_10min > 0;

-- 4. K-means Clustering for Device Behavior Patterns
CREATE OR REPLACE MODEL `iot_anomaly_detection.device_clustering_model`
OPTIONS(
  model_type='KMEANS',
  num_clusters=5
) AS
SELECT
  AVG(temperature) as avg_temperature,
  AVG(vibration_level) as avg_vibration,
  STDDEV(temperature) as temp_variability,
  STDDEV(vibration_level) as vibration_variability,
  COUNT(*) as reading_count
FROM `iot_anomaly_detection.sensor_data`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY device_id, device_type;

-- 5. Time Series Forecasting for Temperature
CREATE OR REPLACE MODEL `iot_anomaly_detection.temperature_forecast_model`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='timestamp',
  time_series_data_col='temperature',
  time_series_id_col='device_id',
  auto_arima=true,
  data_frequency='AUTO_FREQUENCY'
) AS
SELECT
  device_id,
  timestamp,
  temperature
FROM `iot_anomaly_detection.sensor_data`
WHERE 
  timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
  AND device_id IN (
    SELECT device_id 
    FROM `iot_anomaly_detection.sensor_data_temp_vibration` 
    GROUP BY device_id 
    HAVING COUNT(*) > 100
  );

-- Model evaluation queries
-- Evaluate anomaly classification model
SELECT
  *
FROM
  ML.EVALUATE(MODEL `iot_anomaly_detection.anomaly_classification_model`,
    (
      SELECT
        device_type,
        temperature,
        vibration_level as vibration,
        temp_change,
        vibration_change,
        temp_avg_5min,
        vibration_avg_5min,
        temp_stddev_10min,
        vibration_stddev_10min,
        ABS(temperature - temp_avg_5min) / NULLIF(temp_stddev_10min, 0) as temp_z_score,
        ABS(vibration - vibration_avg_5min) / NULLIF(vibration_stddev_10min, 0) as vibration_z_score,
        label
      FROM `iot_anomaly_detection.training_data_temp_vibration`
      WHERE 
        temp_avg_5min IS NOT NULL 
        AND vibration_avg_5min IS NOT NULL
        AND temp_stddev_10min IS NOT NULL
        AND vibration_stddev_10min IS NOT NULL
        AND temp_stddev_10min > 0
        AND vibration_stddev_10min > 0
        AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    )
  );

-- Feature importance for anomaly classification
SELECT
  *
FROM
  ML.FEATURE_IMPORTANCE(MODEL `iot_anomaly_detection.anomaly_classification_model`);

-- Global explanation for the model
SELECT
  *
FROM
  ML.GLOBAL_EXPLAIN(MODEL `iot_anomaly_detection.anomaly_classification_model`);

