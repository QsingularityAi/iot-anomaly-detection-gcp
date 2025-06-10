#!/usr/bin/env python3
"""
Model Performance Evaluation Script

This script evaluates the performance of different anomaly detection models
and stores the results in BigQuery for monitoring and comparison.
"""

import argparse
import json
import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelPerformanceEvaluator:
    """Evaluate and track model performance metrics."""
    
    def __init__(self, project_id: str, dataset_id: str = "iot_anomaly_detection"):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bq_client = bigquery.Client(project=project_id)
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics."""
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'false_positive_rate': self._calculate_fpr(y_true, y_pred),
            'false_negative_rate': self._calculate_fnr(y_true, y_pred),
        }
        
        # Add AUC-ROC if scores are provided
        if y_scores is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = None
            
        return metrics
    
    def _calculate_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate False Positive Rate."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def _calculate_fnr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate False Negative Rate."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    def store_performance_metrics(self, model_name: str, model_version: str, 
                                metrics: Dict[str, float], training_data_size: int = None,
                                evaluation_data_size: int = None) -> None:
        """Store performance metrics in BigQuery."""
        
        table_id = f"{self.project_id}.{self.dataset_id}.model_performance"
        
        row = {
            'model_name': model_name,
            'model_version': model_version,
            'evaluation_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'accuracy': metrics.get('accuracy'),
            'auc_roc': metrics.get('auc_roc'),
            'false_positive_rate': metrics.get('false_positive_rate'),
            'false_negative_rate': metrics.get('false_negative_rate'),
            'training_data_size': training_data_size,
            'evaluation_data_size': evaluation_data_size,
        }
        
        # Insert row into BigQuery
        errors = self.bq_client.insert_rows_json(table_id, [row])
        
        if errors:
            print(f"Error inserting performance metrics: {errors}")
        else:
            print(f"Performance metrics stored for {model_name} v{model_version}")
    
    def compare_models(self, days_back: int = 30) -> pd.DataFrame:
        """Compare performance of different models over time."""
        
        query = f"""
        SELECT 
            model_name,
            model_version,
            evaluation_timestamp,
            precision,
            recall,
            f1_score,
            accuracy,
            auc_roc,
            false_positive_rate,
            false_negative_rate
        FROM `{self.project_id}.{self.dataset_id}.model_performance`
        WHERE evaluation_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
        ORDER BY evaluation_timestamp DESC
        """
        
        return self.bq_client.query(query).to_dataframe()
    
    def get_best_model(self, metric: str = 'f1_score', days_back: int = 30) -> Dict[str, Any]:
        """Get the best performing model based on a specific metric."""
        
        query = f"""
        SELECT 
            model_name,
            model_version,
            evaluation_timestamp,
            {metric}
        FROM `{self.project_id}.{self.dataset_id}.model_performance`
        WHERE evaluation_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
            AND {metric} IS NOT NULL
        ORDER BY {metric} DESC
        LIMIT 1
        """
        
        result = self.bq_client.query(query).to_dataframe()
        
        if not result.empty:
            return result.iloc[0].to_dict()
        else:
            return {}
    
    def create_performance_visualization(self, df: pd.DataFrame, save_path: str = None) -> None:
        """Create visualizations for model performance comparison."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Precision comparison
        axes[0, 0].bar(df['model_name'], df['precision'])
        axes[0, 0].set_title('Precision by Model')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        axes[0, 1].bar(df['model_name'], df['recall'])
        axes[0, 1].set_title('Recall by Model')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[1, 0].bar(df['model_name'], df['f1_score'])
        axes[1, 0].set_title('F1-Score by Model')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # AUC-ROC comparison (if available)
        if 'auc_roc' in df.columns and df['auc_roc'].notna().any():
            axes[1, 1].bar(df['model_name'], df['auc_roc'])
            axes[1, 1].set_title('AUC-ROC by Model')
            axes[1, 1].set_ylabel('AUC-ROC')
        else:
            axes[1, 1].bar(df['model_name'], df['accuracy'])
            axes[1, 1].set_title('Accuracy by Model')
            axes[1, 1].set_ylabel('Accuracy')
        
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance visualization saved to {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, days_back: int = 30) -> str:
        """Generate a comprehensive performance report."""
        
        df = self.compare_models(days_back)
        
        if df.empty:
            return "No performance data available for the specified time period."
        
        # Get latest performance for each model
        latest_performance = df.groupby('model_name').first().reset_index()
        
        report = f"""
# Model Performance Report
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Time Period: Last {days_back} days

## Summary Statistics

Total Models Evaluated: {len(latest_performance)}
Total Evaluations: {len(df)}

## Model Performance Comparison

"""
        
        for _, row in latest_performance.iterrows():
            report += f"""
### {row['model_name']} (v{row['model_version']})
- **Precision**: {row['precision']:.3f}
- **Recall**: {row['recall']:.3f}
- **F1-Score**: {row['f1_score']:.3f}
- **Accuracy**: {row['accuracy']:.3f}
- **AUC-ROC**: {row['auc_roc']:.3f if pd.notna(row['auc_roc']) else 'N/A'}
- **False Positive Rate**: {row['false_positive_rate']:.3f}
- **False Negative Rate**: {row['false_negative_rate']:.3f}
- **Last Evaluated**: {row['evaluation_timestamp']}

"""
        
        # Best performing models
        best_f1 = self.get_best_model('f1_score', days_back)
        best_precision = self.get_best_model('precision', days_back)
        best_recall = self.get_best_model('recall', days_back)
        
        report += f"""
## Best Performing Models

- **Best F1-Score**: {best_f1.get('model_name', 'N/A')} ({best_f1.get('f1_score', 0):.3f})
- **Best Precision**: {best_precision.get('model_name', 'N/A')} ({best_precision.get('precision', 0):.3f})
- **Best Recall**: {best_recall.get('model_name', 'N/A')} ({best_recall.get('recall', 0):.3f})

## Recommendations

"""
        
        # Add recommendations based on performance
        if latest_performance['f1_score'].max() < 0.7:
            report += "- Consider retraining models with more data or different hyperparameters.\n"
        
        if latest_performance['false_positive_rate'].max() > 0.1:
            report += "- High false positive rate detected. Consider adjusting decision thresholds.\n"
        
        if latest_performance['false_negative_rate'].max() > 0.2:
            report += "- High false negative rate detected. This could lead to missed anomalies.\n"
        
        return report


def simulate_model_evaluation():
    """Simulate model evaluation with synthetic data for demonstration."""
    
    # Generate synthetic evaluation data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (10% anomalies)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Simulate different model predictions
    models = {
        'bigquery_ml_autoencoder': {
            'predictions': np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
            'scores': np.random.beta(2, 5, n_samples)
        },
        'lstm_autoencoder': {
            'predictions': np.random.choice([0, 1], size=n_samples, p=[0.91, 0.09]),
            'scores': np.random.beta(2, 4, n_samples)
        },
        'ensemble_detector': {
            'predictions': np.random.choice([0, 1], size=n_samples, p=[0.89, 0.11]),
            'scores': np.random.beta(3, 4, n_samples)
        },
        'isolation_forest': {
            'predictions': np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12]),
            'scores': np.random.beta(2, 3, n_samples)
        }
    }
    
    # Adjust predictions to be more realistic (correlated with true labels)
    for model_name, model_data in models.items():
        # Make predictions more accurate for true anomalies
        anomaly_indices = np.where(y_true == 1)[0]
        normal_indices = np.where(y_true == 0)[0]
        
        # Increase detection rate for true anomalies
        model_data['predictions'][anomaly_indices] = np.random.choice([0, 1], 
                                                                     size=len(anomaly_indices), 
                                                                     p=[0.3, 0.7])
        
        # Decrease false positive rate for normal data
        model_data['predictions'][normal_indices] = np.random.choice([0, 1], 
                                                                    size=len(normal_indices), 
                                                                    p=[0.95, 0.05])
        
        # Adjust scores accordingly
        model_data['scores'][anomaly_indices] += 0.3
        model_data['scores'][normal_indices] -= 0.2
        model_data['scores'] = np.clip(model_data['scores'], 0, 1)
    
    return y_true, models


def main():
    """Main function to evaluate model performance."""
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--dataset-id', default='iot_anomaly_detection', help='BigQuery dataset ID')
    parser.add_argument('--simulate', action='store_true', help='Run simulation with synthetic data')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--visualize', action='store_true', help='Create performance visualizations')
    
    args = parser.parse_args()
    
    evaluator = ModelPerformanceEvaluator(args.project_id, args.dataset_id)
    
    if args.simulate:
        print("Running simulation with synthetic data...")
        y_true, models = simulate_model_evaluation()
        
        # Evaluate each model
        for model_name, model_data in models.items():
            metrics = evaluator.evaluate_model(
                y_true, 
                model_data['predictions'], 
                model_data['scores']
            )
            
            print(f"\n{model_name} Performance:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.3f}")
            
            # Store metrics in BigQuery
            evaluator.store_performance_metrics(
                model_name=model_name,
                model_version="1.0",
                metrics=metrics,
                training_data_size=8000,
                evaluation_data_size=1000
            )
    
    if args.report:
        print("\nGenerating performance report...")
        report = evaluator.generate_performance_report()
        
        # Save report to file
        with open('model_performance_report.md', 'w') as f:
            f.write(report)
        
        print("Performance report saved to model_performance_report.md")
        print(report)
    
    if args.visualize:
        print("\nCreating performance visualizations...")
        df = evaluator.compare_models()
        
        if not df.empty:
            # Get latest performance for each model
            latest_performance = df.groupby('model_name').first().reset_index()
            evaluator.create_performance_visualization(
                latest_performance, 
                'model_performance_comparison.png'
            )
        else:
            print("No performance data available for visualization.")

if __name__ == "__main__":
    main()

