#!/usr/bin/env python3
"""
IoT Device Data Simulator - Temperature & Vibration Only
Generates realistic temperature and vibration sensor data for anomaly detection testing.
"""

import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from google.cloud import pubsub_v1
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IoTDeviceSimulator:
    def __init__(self, project_id, topic_name, num_devices=50):
        self.project_id = project_id
        self.topic_name = topic_name
        self.num_devices = num_devices
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_name)
        self.devices = self._initialize_devices()
        self.running = False
        
    def _initialize_devices(self):
        """Initialize device configurations with baseline values"""
        devices = []
        for i in range(self.num_devices):
            device = {
                'device_id': f'sensor_{i:04d}',
                'location': {
                    'building': random.choice(['Building_A', 'Building_B', 'Building_C']),
                    'floor': random.randint(1, 10),
                    'room': f'Room_{random.randint(100, 999)}'
                },
                'device_type': random.choice(['industrial_sensor', 'hvac_monitor', 'machinery_sensor']),
                # Temperature baseline (Celsius)
                'temp_baseline': random.uniform(18.0, 25.0),
                'temp_variance': random.uniform(0.5, 2.0),
                # Vibration baseline (mm/s RMS)
                'vibration_baseline': random.uniform(0.5, 3.0),
                'vibration_variance': random.uniform(0.1, 0.5),
                # Anomaly probability
                'anomaly_probability': 0.02,  # 2% chance of anomaly
                'last_anomaly': None
            }
            devices.append(device)
        return devices
    
    def _generate_normal_reading(self, device):
        """Generate normal sensor readings"""
        # Temperature with daily cycle and random variation
        hour = datetime.now().hour
        daily_temp_variation = 2 * np.sin(2 * np.pi * hour / 24)  # Daily temperature cycle
        
        temperature = (device['temp_baseline'] + 
                      daily_temp_variation + 
                      np.random.normal(0, device['temp_variance']))
        
        # Vibration with some correlation to temperature (machinery heating up)
        temp_factor = max(0, (temperature - device['temp_baseline']) / 10)
        vibration = (device['vibration_baseline'] + 
                    temp_factor * 0.5 +  # Higher temp = slightly more vibration
                    np.random.normal(0, device['vibration_variance']))
        
        return {
            'temperature': round(temperature, 2),
            'vibration': round(max(0, vibration), 3)  # Vibration can't be negative
        }
    
    def _generate_anomaly_reading(self, device):
        """Generate anomalous sensor readings"""
        anomaly_type = random.choice([
            'temperature_spike', 'temperature_drop', 
            'vibration_spike', 'combined_anomaly'
        ])
        
        normal_reading = self._generate_normal_reading(device)
        
        if anomaly_type == 'temperature_spike':
            # Temperature spike (overheating)
            normal_reading['temperature'] += random.uniform(8, 15)
            # Vibration might increase slightly due to thermal expansion
            normal_reading['vibration'] *= random.uniform(1.2, 1.8)
            
        elif anomaly_type == 'temperature_drop':
            # Temperature drop (cooling system malfunction)
            normal_reading['temperature'] -= random.uniform(5, 10)
            
        elif anomaly_type == 'vibration_spike':
            # Vibration spike (mechanical issue)
            normal_reading['vibration'] *= random.uniform(3, 8)
            # Temperature might increase due to friction
            normal_reading['temperature'] += random.uniform(2, 5)
            
        elif anomaly_type == 'combined_anomaly':
            # Combined anomaly (major equipment failure)
            normal_reading['temperature'] += random.uniform(10, 20)
            normal_reading['vibration'] *= random.uniform(5, 12)
        
        return {
            'temperature': round(normal_reading['temperature'], 2),
            'vibration': round(max(0, normal_reading['vibration']), 3),
            'anomaly_type': anomaly_type
        }
    
    def _generate_device_reading(self, device):
        """Generate a single reading for a device"""
        # Determine if this should be an anomaly
        is_anomaly = random.random() < device['anomaly_probability']
        
        if is_anomaly:
            sensor_data = self._generate_anomaly_reading(device)
            device['last_anomaly'] = datetime.now()
        else:
            sensor_data = self._generate_normal_reading(device)
        
        # Create the complete message
        message = {
            'device_id': device['device_id'],
            'timestamp': datetime.now().isoformat(),
            'location': device['location'],
            'device_type': device['device_type'],
            'sensor_data': sensor_data,
            'is_anomaly': is_anomaly
        }
        
        return message
    
    def _publish_message(self, message):
        """Publish message to Pub/Sub"""
        try:
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            
            # Add message attributes
            attributes = {
                'device_id': message['device_id'],
                'device_type': message['device_type'],
                'timestamp': message['timestamp']
            }
            
            future = self.publisher.publish(
                self.topic_path, 
                message_bytes, 
                **attributes
            )
            
            # Don't wait for the result to avoid blocking
            logger.debug(f"Published message for device {message['device_id']}")
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
    
    def simulate_device(self, device, interval=5):
        """Simulate a single device continuously"""
        while self.running:
            try:
                message = self._generate_device_reading(device)
                self._publish_message(message)
                
                # Add some jitter to avoid synchronized publishing
                jitter = random.uniform(-0.5, 0.5)
                time.sleep(interval + jitter)
                
            except Exception as e:
                logger.error(f"Error in device simulation for {device['device_id']}: {e}")
                time.sleep(interval)
    
    def start_simulation(self, interval=5):
        """Start simulating all devices"""
        logger.info(f"Starting simulation for {self.num_devices} devices")
        logger.info(f"Publishing to topic: {self.topic_path}")
        logger.info("Data types: Temperature (°C) and Vibration (mm/s RMS)")
        
        self.running = True
        threads = []
        
        # Start a thread for each device
        for device in self.devices:
            thread = threading.Thread(
                target=self.simulate_device, 
                args=(device, interval),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping simulation...")
            self.running = False
            
        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=1)
        
        logger.info("Simulation stopped")
    
    def generate_sample_data(self, num_samples=100):
        """Generate sample data for testing (without publishing)"""
        samples = []
        for _ in range(num_samples):
            device = random.choice(self.devices)
            message = self._generate_device_reading(device)
            samples.append(message)
        return samples

def main():
    # Configuration
    PROJECT_ID = "zeltask-staging"  # Replace with your GCP project ID
    TOPIC_NAME = "iot-temp-vibration-data"
    NUM_DEVICES = 50
    PUBLISH_INTERVAL = 5  # seconds
    
    # Create simulator
    simulator = IoTDeviceSimulator(
        project_id=PROJECT_ID,
        topic_name=TOPIC_NAME,
        num_devices=NUM_DEVICES
    )
    
    # Generate sample data for inspection
    print("Sample data structure:")
    samples = simulator.generate_sample_data(3)
    for sample in samples:
        print(json.dumps(sample, indent=2))
    
    print(f"\nStarting simulation with {NUM_DEVICES} devices...")
    print("Data includes only Temperature and Vibration sensors")
    print("Press Ctrl+C to stop")
    
    # Start simulation
    simulator.start_simulation(interval=PUBLISH_INTERVAL)

if __name__ == "__main__":
    main()

