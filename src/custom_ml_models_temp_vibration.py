#!/usr/bin/env python3
"""
Custom ML Models for Temperature & Vibration Anomaly Detection
Focused on temperature and vibration sensor data only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TemperatureVibrationAnomalyDetector:
    """
    Comprehensive anomaly detection system for temperature and vibration data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = ['temperature', 'vibration']
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic temperature and vibration data"""
        
        # Time-based features
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='h')
        hours = timestamps.hour
        days = timestamps.dayofyear
        
        # Base patterns
        daily_temp_cycle = 3 * np.sin(2 * np.pi * hours / 24)  # Daily temperature variation
        seasonal_temp_trend = 5 * np.sin(2 * np.pi * days / 365)  # Seasonal variation
        
        # Normal temperature data (18-25°C base range)
        base_temp = 21.5 + daily_temp_cycle + seasonal_temp_trend
        temp_noise = np.random.normal(0, 1.5, n_samples)
        temperature = base_temp + temp_noise
        
        # Vibration correlated with temperature (machinery heating up)
        base_vibration = 1.5 + 0.1 * (temperature - 20)  # Slight correlation with temperature
        vibration_noise = np.random.normal(0, 0.3, n_samples)
        vibration = np.maximum(0, base_vibration + vibration_noise)  # Vibration can't be negative
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'vibration': vibration,
            'hour': hours,
            'day_of_year': days,
            'is_anomaly': False
        })
        
        # Inject anomalies (5% of data)
        n_anomalies = int(0.05 * n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['temp_spike', 'temp_drop', 'vibration_spike', 'combined'])
            
            if anomaly_type == 'temp_spike':
                df.loc[idx, 'temperature'] += np.random.uniform(8, 15)
                df.loc[idx, 'vibration'] *= np.random.uniform(1.2, 2.0)
            elif anomaly_type == 'temp_drop':
                df.loc[idx, 'temperature'] -= np.random.uniform(5, 10)
            elif anomaly_type == 'vibration_spike':
                df.loc[idx, 'vibration'] *= np.random.uniform(3, 8)
                df.loc[idx, 'temperature'] += np.random.uniform(2, 5)
            elif anomaly_type == 'combined':
                df.loc[idx, 'temperature'] += np.random.uniform(10, 20)
                df.loc[idx, 'vibration'] *= np.random.uniform(5, 12)
            
            df.loc[idx, 'is_anomaly'] = True
        
        return df
    
    def create_features(self, df):
        """Create additional features for anomaly detection"""
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Rolling statistics (5-point window)
        df['temp_rolling_mean'] = df['temperature'].rolling(window=5, min_periods=1).mean()
        df['temp_rolling_std'] = df['temperature'].rolling(window=5, min_periods=1).std()
        df['vibration_rolling_mean'] = df['vibration'].rolling(window=5, min_periods=1).mean()
        df['vibration_rolling_std'] = df['vibration'].rolling(window=5, min_periods=1).std()
        
        # Rate of change
        df['temp_change'] = df['temperature'].diff()
        df['vibration_change'] = df['vibration'].diff()
        
        # Z-scores
        df['temp_z_score'] = np.abs((df['temperature'] - df['temp_rolling_mean']) / 
                                   (df['temp_rolling_std'] + 1e-8))
        df['vibration_z_score'] = np.abs((df['vibration'] - df['vibration_rolling_mean']) / 
                                        (df['vibration_rolling_std'] + 1e-8))
        
        # Temperature-vibration interaction
        df['temp_vibration_ratio'] = df['temperature'] / (df['vibration'] + 1e-8)
        df['temp_vibration_product'] = df['temperature'] * df['vibration']
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def train_isolation_forest(self, X, contamination=0.05):
        """Train Isolation Forest model"""
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        # Store model and scaler
        self.models['isolation_forest'] = model
        self.scalers['isolation_forest'] = scaler
        
        return model, scaler
    
    def train_one_class_svm(self, X, nu=0.05):
        """Train One-Class SVM model"""
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        model.fit(X_scaled)
        
        # Store model and scaler
        self.models['one_class_svm'] = model
        self.scalers['one_class_svm'] = scaler
        
        return model, scaler
    
    def create_lstm_autoencoder(self, sequence_length, n_features):
        """Create LSTM Autoencoder for sequence anomaly detection"""
        
        # Encoder
        input_layer = Input(shape=(sequence_length, n_features))
        encoder = LSTM(64, activation='relu', return_sequences=True)(input_layer)
        encoder = LSTM(32, activation='relu', return_sequences=False)(encoder)
        encoder = Dropout(0.2)(encoder)
        
        # Decoder
        decoder = RepeatVector(sequence_length)(encoder)
        decoder = LSTM(32, activation='relu', return_sequences=True)(decoder)
        decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(n_features))(decoder)
        
        # Create model
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def prepare_sequences(self, data, sequence_length=10):
        """Prepare sequences for LSTM autoencoder"""
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def train_lstm_autoencoder(self, X, sequence_length=10, epochs=50):
        """Train LSTM Autoencoder"""
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Prepare sequences
        sequences = self.prepare_sequences(X_scaled, sequence_length)
        
        # Split data
        train_size = int(0.8 * len(sequences))
        X_train = sequences[:train_size]
        X_val = sequences[train_size:]
        
        # Create and train model
        model = self.create_lstm_autoencoder(sequence_length, X.shape[1])
        
        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, X_val),
            verbose=0,
            shuffle=True
        )
        
        # Store model and scaler
        self.models['lstm_autoencoder'] = model
        self.scalers['lstm_autoencoder'] = scaler
        
        return model, scaler, history
    
    def predict_anomalies(self, X, model_name='isolation_forest'):
        """Predict anomalies using specified model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        if model_name == 'lstm_autoencoder':
            # For LSTM autoencoder, calculate reconstruction error
            X_scaled = scaler.transform(X)
            sequences = self.prepare_sequences(X_scaled, 10)
            
            if len(sequences) == 0:
                return np.array([])
            
            reconstructed = model.predict(sequences, verbose=0)
            mse = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
            
            # Use threshold based on training data distribution
            threshold = np.percentile(mse, 95)  # Top 5% as anomalies
            predictions = (mse > threshold).astype(int)
            
            # Pad predictions to match original length
            padded_predictions = np.zeros(len(X))
            padded_predictions[10-1:] = predictions
            
            return padded_predictions
        else:
            # For other models
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            return (predictions == -1).astype(int)  # Convert to 0/1
    
    def ensemble_predict(self, X):
        """Ensemble prediction using all trained models"""
        
        predictions = {}
        for model_name in self.models.keys():
            try:
                pred = self.predict_anomalies(X, model_name)
                if len(pred) == len(X):
                    predictions[model_name] = pred
            except Exception as e:
                print(f"Error with {model_name}: {e}")
        
        if not predictions:
            return np.zeros(len(X))
        
        # Majority voting
        pred_array = np.array(list(predictions.values()))
        ensemble_pred = (np.mean(pred_array, axis=0) > 0.5).astype(int)
        
        return ensemble_pred, predictions
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        
        print(f"\n=== {model_name} Performance ===")
        print(f"Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            print(f"AUC-ROC: {auc:.3f}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def visualize_results(self, df, predictions, model_name):
        """Visualize anomaly detection results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Temperature & Vibration Anomaly Detection - {model_name}', fontsize=16)
        
        # Temperature time series
        axes[0, 0].plot(df['timestamp'], df['temperature'], alpha=0.7, label='Temperature')
        anomaly_mask = predictions == 1
        if np.any(anomaly_mask):
            axes[0, 0].scatter(df.loc[anomaly_mask, 'timestamp'], 
                              df.loc[anomaly_mask, 'temperature'], 
                              color='red', s=20, label='Detected Anomalies')
        axes[0, 0].set_title('Temperature Over Time')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Vibration time series
        axes[0, 1].plot(df['timestamp'], df['vibration'], alpha=0.7, label='Vibration', color='green')
        if np.any(anomaly_mask):
            axes[0, 1].scatter(df.loc[anomaly_mask, 'timestamp'], 
                              df.loc[anomaly_mask, 'vibration'], 
                              color='red', s=20, label='Detected Anomalies')
        axes[0, 1].set_title('Vibration Over Time')
        axes[0, 1].set_ylabel('Vibration (mm/s)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Temperature vs Vibration scatter plot
        normal_mask = predictions == 0
        axes[1, 0].scatter(df.loc[normal_mask, 'temperature'], 
                          df.loc[normal_mask, 'vibration'], 
                          alpha=0.6, label='Normal', s=10)
        if np.any(anomaly_mask):
            axes[1, 0].scatter(df.loc[anomaly_mask, 'temperature'], 
                              df.loc[anomaly_mask, 'vibration'], 
                              color='red', alpha=0.8, label='Anomalies', s=20)
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Vibration (mm/s)')
        axes[1, 0].set_title('Temperature vs Vibration')
        axes[1, 0].legend()
        
        # Distribution of anomaly scores
        if model_name == 'isolation_forest' and 'isolation_forest' in self.models:
            model = self.models['isolation_forest']
            scaler = self.scalers['isolation_forest']
            feature_cols = ['temperature', 'vibration', 'temp_z_score', 'vibration_z_score']
            X_scaled = scaler.transform(df[feature_cols])
            scores = model.decision_function(X_scaled)
            
            axes[1, 1].hist(scores[normal_mask], bins=50, alpha=0.7, label='Normal', density=True)
            if np.any(anomaly_mask):
                axes[1, 1].hist(scores[anomaly_mask], bins=50, alpha=0.7, label='Anomalies', density=True)
            axes[1, 1].set_xlabel('Anomaly Score')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Anomaly Score Distribution')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Score distribution\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(f'anomaly_detection_{model_name.lower()}_temp_vibration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, filepath_prefix='temp_vibration_anomaly_models'):
        """Save trained models"""
        
        for model_name, model in self.models.items():
            if model_name == 'lstm_autoencoder':
                model.save(f'{filepath_prefix}_{model_name}.h5')
            else:
                joblib.dump(model, f'{filepath_prefix}_{model_name}.pkl')
        
        # Save scalers
        joblib.dump(self.scalers, f'{filepath_prefix}_scalers.pkl')
        
        print(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix='temp_vibration_anomaly_models'):
        """Load trained models"""
        
        try:
            # Load scalers
            self.scalers = joblib.load(f'{filepath_prefix}_scalers.pkl')
            
            # Load models
            for model_name in ['isolation_forest', 'one_class_svm']:
                try:
                    self.models[model_name] = joblib.load(f'{filepath_prefix}_{model_name}.pkl')
                except FileNotFoundError:
                    print(f"Model {model_name} not found")
            
            # Load LSTM autoencoder
            try:
                self.models['lstm_autoencoder'] = tf.keras.models.load_model(f'{filepath_prefix}_lstm_autoencoder.h5')
            except:
                print("LSTM autoencoder not found")
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")

def main():
    """Main function to demonstrate the anomaly detection system"""
    
    print("Temperature & Vibration Anomaly Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = TemperatureVibrationAnomalyDetector()
    
    # Generate synthetic data
    print("Generating synthetic temperature and vibration data...")
    df = detector.generate_synthetic_data(n_samples=5000)
    
    # Create features
    print("Creating features...")
    df = detector.create_features(df)
    
    # Prepare feature matrix
    feature_cols = ['temperature', 'vibration', 'temp_z_score', 'vibration_z_score', 
                   'temp_change', 'vibration_change', 'temp_vibration_ratio']
    X = df[feature_cols].fillna(0)
    y = df['is_anomaly'].astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Anomaly rate: {y.mean():.3f}")
    
    # Train models
    print("\nTraining Isolation Forest...")
    detector.train_isolation_forest(X, contamination=0.05)
    
    print("Training One-Class SVM...")
    detector.train_one_class_svm(X, nu=0.05)
    
    print("Training LSTM Autoencoder...")
    lstm_features = ['temperature', 'vibration']
    detector.train_lstm_autoencoder(df[lstm_features], sequence_length=10, epochs=30)
    
    # Evaluate models
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    results = {}
    
    for model_name in detector.models.keys():
        print(f"\nEvaluating {model_name}...")
        
        if model_name == 'lstm_autoencoder':
            predictions = detector.predict_anomalies(df[lstm_features], model_name)
        else:
            predictions = detector.predict_anomalies(X, model_name)
        
        if len(predictions) == len(y):
            results[model_name] = detector.evaluate_model(y, predictions, model_name)
            detector.visualize_results(df, predictions, model_name)
    
    # Ensemble prediction
    print(f"\nEvaluating Ensemble Model...")
    ensemble_pred, individual_preds = detector.ensemble_predict(X)
    if len(ensemble_pred) == len(y):
        results['ensemble'] = detector.evaluate_model(y, ensemble_pred, 'Ensemble')
        detector.visualize_results(df, ensemble_pred, 'Ensemble')
    
    # Save models
    detector.save_models('temp_vibration_models')
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:20} - F1: {metrics['f1_score']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}")
    
    print("\nTemperature & Vibration Anomaly Detection System Training Complete!")
    print("Models saved and ready for deployment.")

if __name__ == "__main__":
    main()

