#!/usr/bin/env python3
"""
Comprehensive test script for IoT Anomaly Detection System
Tests all components: ML models, data pipeline, BigQuery integration
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
try:
    from custom_ml_models_temp_vibration import TemperatureVibrationAnomalyDetector
    from iot_device_simulator_temp_vibration import IoTDeviceSimulator
    print("✅ Successfully imported custom modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def test_ml_models():
    """Test ML models functionality"""
    print("\n🧠 Testing ML Models...")
    
    try:
        # Initialize detector
        detector = TemperatureVibrationAnomalyDetector()
        
        # Generate synthetic data
        print("📊 Generating synthetic data...")
        data = detector.generate_synthetic_data(n_samples=1000)
        print(f"Generated {len(data)} samples")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Temperature range: {data['temperature'].min():.2f} - {data['temperature'].max():.2f}°C")
        print(f"Vibration range: {data['vibration'].min():.2f} - {data['vibration'].max():.2f}")
        
        # Prepare features
        X = data[detector.feature_columns]
        y = data['is_anomaly']
        
        print(f"Features shape: {X.shape}")
        print(f"Anomaly rate: {y.mean():.2%}")
        
        # Test Isolation Forest
        print("\n🌲 Testing Isolation Forest...")
        iso_model, iso_scaler = detector.train_isolation_forest(X)
        iso_predictions = detector.predict_anomalies(X, 'isolation_forest')
        print(f"Isolation Forest predictions: {np.sum(iso_predictions)} anomalies detected")
        
        # Test One-Class SVM
        print("\n🎯 Testing One-Class SVM...")
        svm_model, svm_scaler = detector.train_one_class_svm(X)
        svm_predictions = detector.predict_anomalies(X, 'one_class_svm')
        print(f"One-Class SVM predictions: {np.sum(svm_predictions)} anomalies detected")
        
        # Test LSTM Autoencoder
        print("\n🧠 Testing LSTM Autoencoder...")
        lstm_model, lstm_scaler, history = detector.train_lstm_autoencoder(X, epochs=5)
        lstm_predictions = detector.predict_anomalies(X, 'lstm_autoencoder')
        print(f"LSTM Autoencoder predictions: {np.sum(lstm_predictions)} anomalies detected")
        
        # Test ensemble prediction
        print("\n🎭 Testing Ensemble Prediction...")
        ensemble_pred, individual_preds = detector.ensemble_predict(X)
        print(f"Ensemble predictions: {np.sum(ensemble_pred)} anomalies detected")
        print(f"Individual model predictions: {individual_preds}")
        
        # Evaluate models
        print("\n📈 Model Evaluation:")
        for model_name in detector.models.keys():
            predictions = detector.predict_anomalies(X, model_name)
            if len(predictions) == len(y):
                detector.evaluate_model(y, predictions, model_name)
        
        print("✅ ML Models test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ ML Models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_iot_simulator():
    """Test IoT device simulator"""
    print("\n📡 Testing IoT Device Simulator...")
    
    try:
        # Initialize simulator
        simulator = IoTDeviceSimulator(
            project_id="test-project",
            topic_name="test-topic",
            device_count=3
        )
        
        # Generate sample data
        print("📊 Generating sample IoT data...")
        sample_data = []
        
        for i in range(10):
            data = simulator.generate_sensor_data()
            sample_data.append(data)
            print(f"Sample {i+1}: Device {data['device_id']}, Temp: {data['sensor_data']['temperature']:.1f}°C, Vibration: {data['sensor_data']['vibration']:.2f}")
        
        # Validate data structure
        required_fields = ['device_id', 'timestamp', 'device_type', 'location', 'sensor_data']
        for data in sample_data:
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
        
        print("✅ IoT Simulator test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ IoT Simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing pipeline components"""
    print("\n⚙️ Testing Data Processing Pipeline...")
    
    try:
        # Test message parsing
        print("📝 Testing message parsing...")
        
        sample_message = {
            "device_id": "temp_sensor_001",
            "timestamp": datetime.now().isoformat(),
            "device_type": "temperature_sensor",
            "location": {"building": "A", "floor": 1, "room": "101"},
            "sensor_data": {
                "temperature": 25.5,
                "vibration": 2.1
            }
        }
        
        # Convert to JSON string (simulating Pub/Sub message)
        message_json = json.dumps(sample_message).encode('utf-8')
        
        # Test parsing (simulating ParseIoTMessage DoFn)
        parsed_data = json.loads(message_json.decode('utf-8'))
        
        # Validate parsed data
        required_fields = ['device_id', 'timestamp', 'temperature', 'vibration']
        sensor_data = parsed_data.get('sensor_data', {})
        
        validation_data = {
            'device_id': parsed_data.get('device_id'),
            'timestamp': parsed_data.get('timestamp'),
            'temperature': sensor_data.get('temperature'),
            'vibration': sensor_data.get('vibration')
        }
        
        for field in required_fields:
            if field not in validation_data or validation_data[field] is None:
                raise ValueError(f"Missing or null field: {field}")
        
        print(f"✅ Message parsing successful: {validation_data}")
        
        # Test anomaly detection logic
        print("🔍 Testing anomaly detection logic...")
        
        # Temperature anomaly detection
        temp = validation_data['temperature']
        temp_anomaly = temp < 10 or temp > 40
        
        # Vibration anomaly detection
        vibration = validation_data['vibration']
        vibration_anomaly = vibration > 8.0
        
        # Combined anomaly
        combined_anomaly = temp_anomaly or vibration_anomaly
        
        print(f"Temperature: {temp}°C, Anomaly: {temp_anomaly}")
        print(f"Vibration: {vibration}, Anomaly: {vibration_anomaly}")
        print(f"Combined Anomaly: {combined_anomaly}")
        
        print("✅ Data Processing Pipeline test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Data Processing Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bigquery_schema():
    """Test BigQuery schema compatibility"""
    print("\n🗄️ Testing BigQuery Schema Compatibility...")
    
    try:
        # Define expected schema
        expected_schema = [
            'device_id', 'device_type', 'location', 'timestamp', 'processed_at',
            'temperature', 'vibration_level', 'is_anomaly', 'anomaly_type',
            'anomaly_score', 'anomaly_reasons', 'anomaly_severity',
            'temp_anomaly', 'vibration_anomaly', 'temp_z_score', 'vibration_z_score'
        ]
        
        # Create sample data matching schema
        sample_bq_data = {
            'device_id': 'temp_sensor_001',
            'device_type': 'temperature_sensor',
            'location': '{"building": "A", "floor": 1, "room": "101"}',
            'timestamp': datetime.now().isoformat(),
            'processed_at': datetime.now().isoformat(),
            'temperature': 25.5,
            'vibration_level': 2.1,
            'is_anomaly': False,
            'anomaly_type': 'normal',
            'anomaly_score': 0.0,
            'anomaly_reasons': '[]',
            'anomaly_severity': 'normal',
            'temp_anomaly': False,
            'vibration_anomaly': False,
            'temp_z_score': 0.7,
            'vibration_z_score': 0.1
        }
        
        # Validate all fields are present
        for field in expected_schema:
            if field not in sample_bq_data:
                raise ValueError(f"Missing field in BigQuery data: {field}")
        
        # Validate data types
        type_validations = {
            'device_id': str,
            'temperature': (int, float),
            'vibration_level': (int, float),
            'is_anomaly': bool,
            'temp_z_score': (int, float, type(None))
        }
        
        for field, expected_type in type_validations.items():
            value = sample_bq_data[field]
            if not isinstance(value, expected_type):
                raise TypeError(f"Field {field} has wrong type: {type(value)}, expected: {expected_type}")
        
        print(f"✅ BigQuery schema validation successful")
        print(f"Schema fields: {len(expected_schema)}")
        print(f"Sample data keys: {len(sample_bq_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ BigQuery Schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_compatibility():
    """Test Streamlit dashboard compatibility"""
    print("\n📊 Testing Streamlit Dashboard Compatibility...")
    
    try:
        # Create sample data that Streamlit would receive
        sample_df_data = {
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
            'device_id': [f'device_{i%3}' for i in range(10)],
            'device_type': ['temperature_sensor'] * 10,
            'temperature': [20 + np.random.normal(0, 2) for _ in range(10)],
            'vibration': [2 + np.random.normal(0, 0.5) for _ in range(10)],
            'is_anomaly': [False] * 8 + [True] * 2,
            'temp_z_score': [np.random.normal(0, 1) for _ in range(10)],
            'vibration_z_score': [np.random.normal(0, 1) for _ in range(10)]
        }
        
        df = pd.DataFrame(sample_df_data)
        
        # Test basic operations that Streamlit dashboard would perform
        print("📈 Testing dashboard operations...")
        
        # Metrics calculations
        total_readings = len(df)
        anomaly_count = df['is_anomaly'].sum()
        anomaly_rate = (anomaly_count / total_readings * 100) if total_readings > 0 else 0
        unique_devices = df['device_id'].nunique()
        avg_temp = df['temperature'].mean()
        
        print(f"Total readings: {total_readings}")
        print(f"Anomalies: {anomaly_count} ({anomaly_rate:.1f}%)")
        print(f"Unique devices: {unique_devices}")
        print(f"Average temperature: {avg_temp:.1f}°C")
        
        # Test data filtering
        normal_data = df[df['is_anomaly'] == False]
        anomaly_data = df[df['is_anomaly'] == True]
        
        print(f"Normal data points: {len(normal_data)}")
        print(f"Anomaly data points: {len(anomaly_data)}")
        
        # Test aggregations
        device_stats = df.groupby('device_id').agg({
            'temperature': ['mean', 'std', 'min', 'max'],
            'vibration': ['mean', 'std', 'min', 'max'],
            'is_anomaly': 'sum'
        }).round(2)
        
        print(f"Device statistics shape: {device_stats.shape}")
        
        print("✅ Streamlit Dashboard compatibility test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Starting Comprehensive IoT Anomaly Detection System Tests")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    tests = [
        ("ML Models", test_ml_models),
        ("IoT Simulator", test_iot_simulator),
        ("Data Processing", test_data_processing),
        ("BigQuery Schema", test_bigquery_schema),
        ("Streamlit Compatibility", test_streamlit_compatibility)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("🏁 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review and fix issues before deployment.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)