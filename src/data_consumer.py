#!/usr/bin/env python3
"""
Real-time Data Consumer

This script consumes data from Pub/Sub and demonstrates real-time processing
before implementing the full Dataflow pipeline.
"""

import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from google.cloud import pubsub_v1
from google.cloud import bigquery
import argparse
from typing import Dict, Any
import datetime

class DataConsumer:
    """Consumes and processes IoT data from Pub/Sub."""
    
    def __init__(self, project_id: str, subscription_name: str, bigquery_dataset: str, bigquery_table: str):
        self.project_id = project_id
        self.subscription_name = subscription_name
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(project_id, subscription_name)
        
        # BigQuery client for data storage
        self.bq_client = bigquery.Client(project=project_id)
        self.dataset_id = bigquery_dataset
        self.table_id = bigquery_table
        self.table_ref = self.bq_client.dataset(bigquery_dataset).table(bigquery_table)
        
        self.running = True
        
    def process_message(self, message) -> None:
        """Process a single message from Pub/Sub."""
        try:
            # Parse the message data
            data = json.loads(message.data.decode('utf-8'))
            
            # Add processing timestamp
            data['processed_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
            
            # Simple anomaly detection logic (rule-based)
            anomaly_score = self.calculate_anomaly_score(data)
            data['anomaly_score'] = anomaly_score
            data['rule_based_anomaly'] = anomaly_score > 0.7
            
            # Log the processed data
            if data.get('is_anomaly') or data.get('rule_based_anomaly'):
                print(f"🚨 ANOMALY DETECTED: {data['device_id']} - Score: {anomaly_score:.2f}")
            else:
                print(f"📊 Processing: {data['device_id']} - Score: {anomaly_score:.2f}")
            
            # Store in BigQuery (in a real implementation, you'd batch these)
            self.store_in_bigquery(data)
            
            # Acknowledge the message
            message.ack()
            
        except Exception as e:
            print(f"Error processing message: {e}")
            message.nack()
    
    def calculate_anomaly_score(self, data: Dict[str, Any]) -> float:
        """Calculate a simple rule-based anomaly score."""
        score = 0.0
        
        # Temperature anomaly detection
        temp = data.get('temperature', 0)
        if temp < -10 or temp > 40:
            score += 0.3
        if temp < -50 or temp > 60:
            score += 0.5
        
        # Humidity anomaly detection
        humidity = data.get('humidity', 50)
        if humidity < 10 or humidity > 90:
            score += 0.2
        if humidity < 0 or humidity > 100:
            score += 0.4
        
        # Pressure anomaly detection
        pressure = data.get('pressure', 1013)
        if pressure < 950 or pressure > 1050:
            score += 0.2
        if pressure < 800 or pressure > 1200:
            score += 0.4
        
        # Sensor failure detection
        if temp == -999.0 or humidity == -1.0 or pressure == 0.0:
            score = 1.0
        
        # Vibration anomaly (for industrial sensors)
        vibration = data.get('vibration_level', 0)
        if vibration > 2.0:
            score += 0.3
        
        return min(1.0, score)
    
    def store_in_bigquery(self, data: Dict[str, Any]) -> None:
        """Store processed data in BigQuery."""
        try:
            # In a real implementation, you'd batch these inserts for efficiency
            rows_to_insert = [data]
            errors = self.bq_client.insert_rows_json(self.table_ref, rows_to_insert)
            
            if errors:
                print(f"BigQuery insert errors: {errors}")
        except Exception as e:
            print(f"Error storing data in BigQuery: {e}")
    
    def start_consuming(self) -> None:
        """Start consuming messages from Pub/Sub."""
        print(f"Starting to consume messages from {self.subscription_path}")
        
        # Configure the subscriber
        flow_control = pubsub_v1.types.FlowControl(max_messages=100)
        
        def signal_handler(signum, frame):
            print("Received interrupt signal. Shutting down...")
            self.running = False
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start pulling messages
        with ThreadPoolExecutor(max_workers=10) as executor:
            streaming_pull_future = self.subscriber.pull(
                request={"subscription": self.subscription_path, "max_messages": 10},
                timeout=300.0
            )
            
            try:
                while self.running:
                    try:
                        response = self.subscriber.pull(
                            request={"subscription": self.subscription_path, "max_messages": 10},
                            timeout=10.0
                        )
                        
                        if response.received_messages:
                            for received_message in response.received_messages:
                                executor.submit(self.process_message, received_message)
                    
                    except Exception as e:
                        print(f"Error in message pulling: {e}")
                        
            except KeyboardInterrupt:
                print("Shutting down...")

def main():
    """Main function to start the data consumer."""
    parser = argparse.ArgumentParser(description='IoT Data Consumer')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--subscription-name', default='dataflow-subscription', help='Pub/Sub subscription name')
    parser.add_argument('--bigquery-dataset', default='iot_anomaly_detection', help='BigQuery dataset name')
    parser.add_argument('--bigquery-table', default='sensor_data', help='BigQuery table name')
    
    args = parser.parse_args()
    
    consumer = DataConsumer(
        args.project_id,
        args.subscription_name,
        args.bigquery_dataset,
        args.bigquery_table
    )
    
    consumer.start_consuming()

if __name__ == "__main__":
    main()

