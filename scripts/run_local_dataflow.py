#!/usr/bin/env python3
"""
Local Dataflow Pipeline Runner

This script runs the Dataflow pipeline locally for testing and development.
"""

import subprocess
import sys
import argparse

def run_local_pipeline(project_id: str, subscription: str, output_table: str):
    """Run the Dataflow pipeline locally using DirectRunner."""
    
    cmd = [
        sys.executable, 'dataflow_pipeline.py',
        '--runner', 'DirectRunner',
        '--project', project_id,
        '--input_subscription', subscription,
        '--output_table', output_table,
        '--window_size', '30',  # Shorter window for testing
        '--streaming'
    ]
    
    print("Running Dataflow pipeline locally...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")

def main():
    parser = argparse.ArgumentParser(description='Run Dataflow pipeline locally')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--subscription', help='Pub/Sub subscription name')
    parser.add_argument('--output-table', help='BigQuery output table')
    
    args = parser.parse_args()
    
    # Set defaults
    subscription = args.subscription or f"projects/{args.project_id}/subscriptions/dataflow-subscription"
    output_table = args.output_table or f"{args.project_id}:iot_anomaly_detection.sensor_data"
    
    run_local_pipeline(args.project_id, subscription, output_table)

if __name__ == "__main__":
    main()

