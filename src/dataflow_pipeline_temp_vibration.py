#!/usr/bin/env python3
"""
Apache Beam Dataflow Pipeline - Temperature & Vibration Anomaly Detection
Processes IoT sensor data for temperature and vibration anomaly detection.
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window
import json
import logging
from datetime import datetime
import numpy as np

class ParseIoTMessage(beam.DoFn):
    """Parse incoming IoT messages"""
    
    def process(self, element):
        try:
            # Parse the Pub/Sub message
            message = json.loads(element.decode('utf-8'))
            
            # Extract relevant fields for temperature and vibration
            sensor_data = message.get('sensor_data', {})
            parsed_data = {
                'device_id': message.get('device_id'),
                'timestamp': message.get('timestamp'),
                'location': message.get('location', {}),
                'device_type': message.get('device_type'),
                'temperature': sensor_data.get('temperature'),
                'vibration': sensor_data.get('vibration'),
                'sensor_data': sensor_data,  # Keep full sensor data for processing
                'is_anomaly': message.get('is_anomaly', False)
            }
            
            # Validate required fields
            if all(key in parsed_data and parsed_data[key] is not None 
                   for key in ['device_id', 'timestamp', 'temperature', 'vibration']):
                yield parsed_data
            else:
                logging.warning(f"Invalid message format: {message}")
                
        except Exception as e:
            logging.error(f"Error parsing message: {e}")

class TemperatureAnomalyDetector(beam.DoFn):
    """Detect temperature anomalies using statistical methods"""
    
    def __init__(self):
        self.temp_thresholds = {
            'industrial_sensor': {'min': 10, 'max': 35},
            'hvac_monitor': {'min': 15, 'max': 30},
            'machinery_sensor': {'min': 12, 'max': 40}
        }
    
    def process(self, element):
        try:
            device_type = element.get('device_type', 'industrial_sensor')
            temperature = element.get('temperature')
            
            # Get thresholds for device type
            thresholds = self.temp_thresholds.get(device_type, self.temp_thresholds['industrial_sensor'])
            
            # Detect anomalies
            temp_anomaly = False
            anomaly_reason = []
            
            if temperature < thresholds['min']:
                temp_anomaly = True
                anomaly_reason.append(f"Temperature too low: {temperature}°C < {thresholds['min']}°C")
            elif temperature > thresholds['max']:
                temp_anomaly = True
                anomaly_reason.append(f"Temperature too high: {temperature}°C > {thresholds['max']}°C")
            
            # Add anomaly detection results
            element['temperature_anomaly'] = temp_anomaly
            element['temperature_anomaly_reason'] = anomaly_reason
            
            yield element
            
        except Exception as e:
            logging.error(f"Error in temperature anomaly detection: {e}")
            yield element

class VibrationAnomalyDetector(beam.DoFn):
    """Detect vibration anomalies using statistical methods"""
    
    def __init__(self):
        self.vibration_thresholds = {
            'industrial_sensor': {'max': 5.0},
            'hvac_monitor': {'max': 3.0},
            'machinery_sensor': {'max': 8.0}
        }
    
    def process(self, element):
        try:
            device_type = element.get('device_type', 'industrial_sensor')
            vibration = element.get('vibration')
            
            # Get thresholds for device type
            thresholds = self.vibration_thresholds.get(device_type, self.vibration_thresholds['industrial_sensor'])
            
            # Detect anomalies
            vibration_anomaly = False
            anomaly_reason = []
            
            if vibration > thresholds['max']:
                vibration_anomaly = True
                anomaly_reason.append(f"Vibration too high: {vibration} mm/s > {thresholds['max']} mm/s")
            
            # Add anomaly detection results
            element['vibration_anomaly'] = vibration_anomaly
            element['vibration_anomaly_reason'] = anomaly_reason
            
            yield element
            
        except Exception as e:
            logging.error(f"Error in vibration anomaly detection: {e}")
            yield element

class CombinedAnomalyDetector(beam.DoFn):
    """Combine temperature and vibration anomaly detection results"""
    
    def process(self, element):
        try:
            temp_anomaly = element.get('temperature_anomaly', False)
            vibration_anomaly = element.get('vibration_anomaly', False)
            
            # Combine anomaly flags
            combined_anomaly = temp_anomaly or vibration_anomaly
            
            # Combine reasons
            all_reasons = []
            all_reasons.extend(element.get('temperature_anomaly_reason', []))
            all_reasons.extend(element.get('vibration_anomaly_reason', []))
            
            # Determine severity
            severity = 'normal'
            if temp_anomaly and vibration_anomaly:
                severity = 'critical'
            elif temp_anomaly or vibration_anomaly:
                severity = 'warning'
            
            # Add combined results
            element['anomaly_detected'] = combined_anomaly
            element['anomaly_severity'] = severity
            element['anomaly_reasons'] = all_reasons
            element['processed_timestamp'] = datetime.now().isoformat()
            
            yield element
            
        except Exception as e:
            logging.error(f"Error in combined anomaly detection: {e}")
            yield element

class FormatForBigQuery(beam.DoFn):
    """Format data for BigQuery insertion"""

    def process(self, element):
        try:
            # Calculate z-scores for temperature and vibration
            temp_z_score = self._calculate_z_score(element.get('temperature'), 'temperature')
            vibration_z_score = self._calculate_z_score(element.get('vibration'), 'vibration')
            
            # Format for BigQuery schema (temperature and vibration only)
            bq_row = {
                'device_id': element.get('device_id'),
                'device_type': element.get('device_type'),
                'location': json.dumps(element.get('location', {})),  # Store location as JSON string
                'timestamp': element.get('timestamp'),
                'processed_at': element.get('processed_timestamp'),
                'temperature': element.get('temperature'),
                'vibration_level': element.get('vibration'),
                'is_anomaly': element.get('anomaly_detected', False),
                'anomaly_type': element.get('anomaly_severity', 'normal'),
                'anomaly_score': self._calculate_anomaly_score(element),
                'anomaly_reasons': json.dumps(element.get('anomaly_reasons', [])),
                'anomaly_severity': element.get('anomaly_severity', 'normal'),
                'temp_anomaly': element.get('temperature_anomaly', False),
                'vibration_anomaly': element.get('vibration_anomaly', False),
                'temp_z_score': temp_z_score,
                'vibration_z_score': vibration_z_score
            }

            # Ensure required fields are not None
            if bq_row['device_id'] is None or bq_row['timestamp'] is None or bq_row['temperature'] is None or bq_row['vibration_level'] is None:
                logging.warning(f"Skipping row with missing required fields: {element}")
                return

            yield bq_row

        except Exception as e:
            logging.error(f"Error formatting for BigQuery: {e}")
    
    def _calculate_z_score(self, value, sensor_type):
        """Calculate simple z-score (placeholder - in production, use historical data)"""
        if value is None:
            return None
        
        # Simple z-score calculation with default means and stds
        defaults = {
            'temperature': {'mean': 22.0, 'std': 5.0},
            'vibration': {'mean': 2.0, 'std': 1.0}
        }
        
        if sensor_type in defaults:
            mean = defaults[sensor_type]['mean']
            std = defaults[sensor_type]['std']
            return (value - mean) / std if std > 0 else 0.0
        
        return 0.0
    
    def _calculate_anomaly_score(self, element):
        """Calculate a simple anomaly score based on detected anomalies"""
        score = 0.0
        
        if element.get('temperature_anomaly', False):
            score += 0.5
        if element.get('vibration_anomaly', False):
            score += 0.5
        
        # Adjust based on severity
        severity = element.get('anomaly_severity', 'normal')
        if severity == 'critical':
            score = min(1.0, score * 1.5)
        elif severity == 'warning':
            score = min(1.0, score * 1.2)
        
        return score

class FormatAlert(beam.DoFn):
    """Format anomaly alerts for notification"""
    
    def process(self, element):
        try:
            # Only process if anomaly detected
            if element.get('anomaly_detected', False):
                alert = {
                    'alert_id': f"{element.get('device_id')}_{element.get('timestamp')}",
                    'device_id': element.get('device_id'),
                    'timestamp': element.get('timestamp'),
                    'severity': element.get('anomaly_severity', 'warning'),
                    'message': f"Anomaly detected on device {element.get('device_id')}",
                    'details': {
                        'temperature': element.get('temperature'),
                        'vibration': element.get('vibration'),
                        'reasons': element.get('anomaly_reasons', []),
                        'location': element.get('location', {})
                    }
                }
                yield alert
                
        except Exception as e:
            logging.error(f"Error formatting alert: {e}")

def run_pipeline(pipeline_options):
    """Run the Dataflow pipeline"""
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        # Read from Pub/Sub
        messages = (
            pipeline
            | 'Read from Pub/Sub' >> beam.io.ReadFromPubSub(
                subscription=pipeline_options.input_subscription if hasattr(pipeline_options, 'input_subscription') else None,
                topic=pipeline_options.input_topic if not hasattr(pipeline_options, 'input_subscription') else None,
                with_attributes=False
            )
        )
        
        # Parse and process messages
        processed_data = (
            messages
            | 'Parse Messages' >> beam.ParDo(ParseIoTMessage())
            | 'Detect Temperature Anomalies' >> beam.ParDo(TemperatureAnomalyDetector())
            | 'Detect Vibration Anomalies' >> beam.ParDo(VibrationAnomalyDetector())
            | 'Combine Anomaly Results' >> beam.ParDo(CombinedAnomalyDetector())
        )
        
        # Write to BigQuery
        (
            processed_data
            | 'Format for BigQuery' >> beam.ParDo(FormatForBigQuery())
            | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                table=pipeline_options.output_table,
                schema={
                    'fields': [
                        {'name': 'device_id', 'type': 'STRING', 'mode': 'REQUIRED'},
                        {'name': 'device_type', 'type': 'STRING', 'mode': 'REQUIRED'},
                        {'name': 'location', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
                        {'name': 'processed_at', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
                        {'name': 'temperature', 'type': 'FLOAT', 'mode': 'REQUIRED'},
                        {'name': 'vibration_level', 'type': 'FLOAT', 'mode': 'REQUIRED'},
                        {'name': 'is_anomaly', 'type': 'BOOLEAN', 'mode': 'REQUIRED'},
                        {'name': 'anomaly_type', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'anomaly_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                        {'name': 'anomaly_reasons', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'anomaly_severity', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'temp_anomaly', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
                        {'name': 'vibration_anomaly', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
                        {'name': 'temp_z_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                        {'name': 'vibration_z_score', 'type': 'FLOAT', 'mode': 'NULLABLE'}
                    ]
                },
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )
        
        # Generate alerts for anomalies
        (
            processed_data
            | 'Format Alerts' >> beam.ParDo(FormatAlert())
            | 'Format Alerts as JSON' >> beam.Map(json.dumps)
            | 'Write Alerts to Pub/Sub' >> beam.io.WriteToPubSub(
                topic=pipeline_options.alert_topic
            )
        )

class CustomPipelineOptions(PipelineOptions):
    """Custom pipeline options"""
    
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            '--input_topic',
            help='Pub/Sub topic to read from'
        )
        parser.add_argument(
            '--input_subscription',
            help='Pub/Sub subscription to read from'
        )
        parser.add_argument(
            '--output_table',
            required=True,
            help='BigQuery table to write to (format: project:dataset.table)'
        )
        parser.add_argument(
            '--alert_topic',
            default='iot-temp-vibration-alerts',
            help='Pub/Sub topic to write alerts to'
        )

def main():
    """Main function"""
    pipeline_options = CustomPipelineOptions()
    
    logging.info("Starting Temperature & Vibration Anomaly Detection Pipeline")
    if hasattr(pipeline_options, 'input_subscription') and pipeline_options.input_subscription:
        logging.info(f"Input subscription: {pipeline_options.input_subscription}")
    else:
        logging.info(f"Input topic: {pipeline_options.input_topic}")
    logging.info(f"Output table: {pipeline_options.output_table}")
    logging.info(f"Alert topic: {pipeline_options.alert_topic}")
    
    run_pipeline(pipeline_options)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
