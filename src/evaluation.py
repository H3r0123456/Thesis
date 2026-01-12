# src/evaluation.py
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

class AlgorithmEvaluator:
    def __init__(self, algorithm, dataset):
        self.algorithm = algorithm
        self.dataset = dataset
        self.results = []
        
    def run_evaluation(self):
        """Run complete evaluation on dataset"""
        print(f"Evaluating {len(self.dataset)} samples...")
        
        for idx, row in self.dataset.iterrows():
            # Process each sample
            result = self.algorithm.process_sample(
                row['temperature'],
                row['gas']
            )
            
            # Map algorithm state to label for comparison
            predicted_label = self._map_state_to_label(result['state'])
            
            self.results.append({
                'timestamp': row.get('timestamp', datetime.now()),
                'temperature': row['temperature'],
                'gas': row['gas'],
                'true_label': row.get('label', 'unknown'),
                'predicted_state': result['state'],
                'predicted_label': predicted_label,
                'trigger': result.get('trigger', 'none')
            })
        
        print("✅ Evaluation complete")
        return pd.DataFrame(self.results)
    
    def _map_state_to_label(self, state: str) -> str:
        """Map algorithm state to ground truth label"""
        if state == "FIRE_ALERT":
            return "fire"
        elif state in ["WARNING", "WATCH"]:
            return "warning"
        else:
            return "normal"
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        # Simple accuracy calculation
        correct = 0
        for _, row in results_df.iterrows():
            if row['true_label'] == row['predicted_label']:
                correct += 1
        
        accuracy = correct / len(results_df) if len(results_df) > 0 else 0
        
        # Count alerts
        fire_alerts = sum(1 for r in self.results if r['predicted_state'] == 'FIRE_ALERT')
        warning_alerts = sum(1 for r in self.results if r['predicted_state'] == 'WARNING')
        
        # Count true events
        true_fires = sum(1 for r in self.results if r['true_label'] == 'fire')
        true_warnings = sum(1 for r in self.results if r['true_label'] == 'warning')
        
        return {
            'total_samples': len(results_df),
            'true_fires': true_fires,
            'true_warnings': true_warnings,
            'detected_fires': fire_alerts,
            'detected_warnings': warning_alerts,
            'accuracy': accuracy,
            'detection_rate': fire_alerts / true_fires if true_fires > 0 else 0
        }
    
    def generate_report(self, save_path="results/metrics/evaluation_report.json"):
        """Generate evaluation report"""
        metrics = self.calculate_metrics()
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'algorithm_config': self.algorithm.config.to_dict(),
            'dataset_info': {
                'total_samples': len(self.dataset),
                'columns': list(self.dataset.columns)
            },
            'performance_metrics': metrics,
            'sample_results': self.results[:10]  # First 10 samples
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {save_path}")
        return report

if __name__ == "__main__":
    # Test the evaluator
    print("Algorithm Evaluator Test")
    print("=" * 40)
    
    # Create mock data
    mock_data = pd.DataFrame({
        'temperature': [25, 30, 65, 70, 28, 75],
        'gas': [0.1, 0.2, 0.8, 0.9, 0.15, 1.0],
        'label': ['normal', 'normal', 'fire', 'fire', 'normal', 'fire']
    })
    
    # Create mock algorithm
    from algorithm import FireDetectionAlgorithm, ThresholdConfig
    
    config = ThresholdConfig()
    algorithm = FireDetectionAlgorithm(config)
    evaluator = AlgorithmEvaluator(algorithm, mock_data)
    
    results = evaluator.run_evaluation()
    metrics = evaluator.calculate_metrics()
    
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")