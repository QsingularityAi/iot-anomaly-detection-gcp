-- BigQuery ML Anomaly Detection Queries
-- This file contains SQL queries for performing anomaly detection using trained models

-- 1. Detect anomalies using the autoencoder model
-- This query identifies data points with high reconstruction error
CREATE OR REPLACE TABLE `iot_anomaly_detection.autoencoder_anomalies` AS
SELECT
  device_id,
  timestamp,
  temperature,
  humidity,
  pressure,
  vibration_level,
  power_consumption,
  anomaly_score,
  CASE 
    WHEN anomaly_score > 0.8 THEN 'HIGH'
    WHEN anomaly_score > 0.6 THEN 'MEDIUM'
    WHEN anomaly_score > 0.4 THEN 'LOW'
    ELSE 'NORMAL'
  END as anomaly_severity,
  'autoencoder' as detection_method
FROM
  ML.DETECT_ANOMALIES(
    MODEL `iot_anomaly_detection.autoencoder_anomaly_model`,
    (
      SELECT
        device_id,
        timestamp,
        temperature,
        humidity,
        pressure,
        COALESCE(vibration_level, 0) as vibration_level,
        COALESCE(power_consumption, 0) as power_consumption,
        heat_index,
        hour_of_day,
        CASE 
          WHEN device_type = 'environmental_sensor' THEN 1
          WHEN device_type = 'industrial_sensor' THEN 2
          WHEN device_type = 'smart_meter' THEN 3
          ELSE 0
        END as device_type_encoded,
        CASE 
          WHEN zone = 'A' THEN 1
          WHEN zone = 'B' THEN 2
          WHEN zone = 'C' THEN 3
          ELSE 0
        END as zone_encoded,
        CASE 
          WHEN criticality = 'low' THEN 1
          WHEN criticality = 'medium' THEN 2
          WHEN criticality = 'high' THEN 3
          ELSE 2
        END as criticality_encoded
      FROM `iot_anomaly_detection.sensor_data`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    ),
    STRUCT(0.02 as contamination)  -- Expect 2% of data to be anomalies
  );

-- 2. Detect anomalies using K-means clustering
-- Points far from cluster centroids are considered anomalies
CREATE OR REPLACE TABLE `iot_anomaly_detection.kmeans_anomalies` AS
WITH cluster_predictions AS (
  SELECT
    device_id,
    timestamp,
    temperature,
    humidity,
    pressure,
    vibration_level,
    power_consumption,
    predicted_cluster_id,
    distance_to_centroid
  FROM
    ML.PREDICT(
      MODEL `iot_anomaly_detection.kmeans_anomaly_model`,
      (
        SELECT
          device_id,
          timestamp,
          temperature,
          humidity,
          pressure,
          COALESCE(vibration_level, 0) as vibration_level,
          COALESCE(power_consumption, 0) as power_consumption,
          heat_index,
          hour_of_day,
          CASE 
            WHEN device_type = 'environmental_sensor' THEN 1
            WHEN device_type = 'industrial_sensor' THEN 2
            WHEN device_type = 'smart_meter' THEN 3
            ELSE 0
          END as device_type_encoded
        FROM `iot_anomaly_detection.sensor_data`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
      )
    )
),
distance_stats AS (
  SELECT
    APPROX_QUANTILES(distance_to_centroid, 100)[OFFSET(95)] as p95_distance,
    APPROX_QUANTILES(distance_to_centroid, 100)[OFFSET(99)] as p99_distance
  FROM cluster_predictions
)
SELECT
  cp.*,
  CASE 
    WHEN cp.distance_to_centroid > ds.p99_distance THEN 0.9
    WHEN cp.distance_to_centroid > ds.p95_distance THEN 0.7
    ELSE 0.3
  END as anomaly_score,
  CASE 
    WHEN cp.distance_to_centroid > ds.p99_distance THEN 'HIGH'
    WHEN cp.distance_to_centroid > ds.p95_distance THEN 'MEDIUM'
    ELSE 'LOW'
  END as anomaly_severity,
  'kmeans' as detection_method
FROM cluster_predictions cp
CROSS JOIN distance_stats ds
WHERE cp.distance_to_centroid > ds.p95_distance;

-- 3. Detect anomalies using time series forecasting
-- Compare actual values with predicted values
CREATE OR REPLACE TABLE `iot_anomaly_detection.forecast_anomalies` AS
WITH forecasts AS (
  SELECT
    device_id,
    forecast_timestamp as timestamp,
    forecast_value as predicted_temperature,
    prediction_interval_lower_bound,
    prediction_interval_upper_bound
  FROM
    ML.FORECAST(
      MODEL `iot_anomaly_detection.temperature_forecast_model`,
      STRUCT(24 as horizon, 0.95 as confidence_level)  -- 24 hour forecast with 95% confidence
    )
),
actual_data AS (
  SELECT
    device_id,
    timestamp,
    temperature as actual_temperature
  FROM `iot_anomaly_detection.sensor_data`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
)
SELECT
  a.device_id,
  a.timestamp,
  a.actual_temperature,
  f.predicted_temperature,
  f.prediction_interval_lower_bound,
  f.prediction_interval_upper_bound,
  ABS(a.actual_temperature - f.predicted_temperature) as prediction_error,
  CASE 
    WHEN a.actual_temperature < f.prediction_interval_lower_bound 
         OR a.actual_temperature > f.prediction_interval_upper_bound THEN 0.8
    WHEN ABS(a.actual_temperature - f.predicted_temperature) > 5 THEN 0.6
    WHEN ABS(a.actual_temperature - f.predicted_temperature) > 3 THEN 0.4
    ELSE 0.2
  END as anomaly_score,
  CASE 
    WHEN a.actual_temperature < f.prediction_interval_lower_bound 
         OR a.actual_temperature > f.prediction_interval_upper_bound THEN 'HIGH'
    WHEN ABS(a.actual_temperature - f.predicted_temperature) > 5 THEN 'MEDIUM'
    WHEN ABS(a.actual_temperature - f.predicted_temperature) > 3 THEN 'LOW'
    ELSE 'NORMAL'
  END as anomaly_severity,
  'time_series_forecast' as detection_method
FROM actual_data a
JOIN forecasts f
  ON a.device_id = f.device_id 
  AND a.timestamp = f.timestamp
WHERE ABS(a.actual_temperature - f.predicted_temperature) > 2;

-- 4. Use supervised model for anomaly prediction
CREATE OR REPLACE TABLE `iot_anomaly_detection.supervised_anomalies` AS
SELECT
  device_id,
  timestamp,
  temperature,
  humidity,
  pressure,
  vibration_level,
  power_consumption,
  predicted_is_anomaly,
  predicted_is_anomaly_probs[OFFSET(1)].prob as anomaly_probability,
  CASE 
    WHEN predicted_is_anomaly_probs[OFFSET(1)].prob > 0.8 THEN 'HIGH'
    WHEN predicted_is_anomaly_probs[OFFSET(1)].prob > 0.6 THEN 'MEDIUM'
    WHEN predicted_is_anomaly_probs[OFFSET(1)].prob > 0.4 THEN 'LOW'
    ELSE 'NORMAL'
  END as anomaly_severity,
  'supervised_classification' as detection_method
FROM
  ML.PREDICT(
    MODEL `iot_anomaly_detection.supervised_anomaly_model`,
    (
      SELECT
        device_id,
        timestamp,
        temperature,
        humidity,
        pressure,
        COALESCE(vibration_level, 0) as vibration_level,
        COALESCE(power_consumption, 0) as power_consumption,
        heat_index,
        hour_of_day,
        CASE 
          WHEN device_type = 'environmental_sensor' THEN 1
          WHEN device_type = 'industrial_sensor' THEN 2
          WHEN device_type = 'smart_meter' THEN 3
          ELSE 0
        END as device_type_encoded,
        CASE 
          WHEN zone = 'A' THEN 1
          WHEN zone = 'B' THEN 2
          WHEN zone = 'C' THEN 3
          ELSE 0
        END as zone_encoded,
        CASE 
          WHEN criticality = 'low' THEN 1
          WHEN criticality = 'medium' THEN 2
          WHEN criticality = 'high' THEN 3
          ELSE 2
        END as criticality_encoded
      FROM `iot_anomaly_detection.sensor_data`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    )
  )
WHERE predicted_is_anomaly = TRUE;

-- 5. Combine all anomaly detection methods into a unified view
CREATE OR REPLACE VIEW `iot_anomaly_detection.unified_anomalies` AS
SELECT * FROM `iot_anomaly_detection.autoencoder_anomalies`
UNION ALL
SELECT * FROM `iot_anomaly_detection.kmeans_anomalies`
UNION ALL
SELECT * FROM `iot_anomaly_detection.forecast_anomalies`
UNION ALL
SELECT * FROM `iot_anomaly_detection.supervised_anomalies`;

-- 6. Create a summary view of anomalies by device and method
CREATE OR REPLACE VIEW `iot_anomaly_detection.anomaly_summary` AS
SELECT
  device_id,
  detection_method,
  COUNT(*) as anomaly_count,
  AVG(anomaly_score) as avg_anomaly_score,
  MAX(anomaly_score) as max_anomaly_score,
  COUNTIF(anomaly_severity = 'HIGH') as high_severity_count,
  COUNTIF(anomaly_severity = 'MEDIUM') as medium_severity_count,
  COUNTIF(anomaly_severity = 'LOW') as low_severity_count,
  MIN(timestamp) as first_anomaly,
  MAX(timestamp) as last_anomaly
FROM `iot_anomaly_detection.unified_anomalies`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
GROUP BY device_id, detection_method
ORDER BY anomaly_count DESC, avg_anomaly_score DESC;

