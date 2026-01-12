# src/algorithm.py - Clean version
import numpy as np
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class ThresholdConfig:
    temp_threshold: float = 60.0
    temp_warning: float = 45.0
    gas_threshold: float = 0.7
    gas_warning: float = 0.4
    verification_window: int = 5
    persistence_samples: int = 3
    
    def to_dict(self):
        return {
            'temp_threshold': self.temp_threshold,
            'temp_warning': self.temp_warning,
            'gas_threshold': self.gas_threshold,
            'gas_warning': self.gas_warning,
            'verification_window': self.verification_window,
            'persistence_samples': self.persistence_samples
        }

class FireDetectionAlgorithm:
    def __init__(self, config: ThresholdConfig = None):
        self.config = config or ThresholdConfig()
        self.state = "NORMAL"
        self.alert_history = []
        self.verification_buffer = []
        self.warning_count = 0
        self.fire_count = 0
        
    def process_sample(self, temperature: float, gas: float) -> dict:
        """Process single sensor reading"""
        # Add to verification buffer
        self.verification_buffer.append({
            'timestamp': datetime.now(),
            'temperature': temperature,
            'gas': gas,
            'state': self._evaluate_sample(temperature, gas)
        })
        
        # Keep only recent samples
        if len(self.verification_buffer) > self.config.verification_window:
            self.verification_buffer.pop(0)
        
        # Apply time-series verification
        result = self._apply_time_series_verification()
        
        # Update state if changed
        if result['state'] != self.state:
            alert_entry = {
                'timestamp': datetime.now(),
                'old_state': self.state,
                'new_state': result['state'],
                'temperature': temperature,
                'gas': gas,
                'trigger': result['trigger']
            }
            self.alert_history.append(alert_entry)
            self.state = result['state']
            
            # Count alerts
            if result['state'] == "FIRE_ALERT":
                self.fire_count += 1
            elif result['state'] == "WARNING":
                self.warning_count += 1
        
        result['current_state'] = self.state
        return result
    
    def _evaluate_sample(self, temperature: float, gas: float) -> str:
        """Evaluate single sample without time verification"""
        if temperature >= self.config.temp_threshold and gas >= self.config.gas_threshold:
            return "FIRE_CANDIDATE"
        elif temperature >= self.config.temp_threshold or gas >= self.config.gas_threshold:
            return "WARNING_CANDIDATE"
        elif temperature >= self.config.temp_warning or gas >= self.config.gas_warning:
            return "WATCH"
        else:
            return "NORMAL"
    
    def _apply_time_series_verification(self) -> dict:
        """Apply time-series verification to buffer"""
        if len(self.verification_buffer) < self.config.persistence_samples:
            return {'state': 'NORMAL', 'trigger': 'insufficient_data'}
        
        # Count recent states
        recent_states = [s['state'] for s in self.verification_buffer[-self.config.persistence_samples:]]
        
        # Check for sustained fire condition
        fire_candidates = recent_states.count("FIRE_CANDIDATE")
        if fire_candidates >= self.config.persistence_samples:
            return {'state': 'FIRE_ALERT', 'trigger': 'temperature_and_gas_threshold'}
        
        # Check for sustained warning
        warning_candidates = recent_states.count("WARNING_CANDIDATE")
        if warning_candidates >= self.config.persistence_samples:
            return {'state': 'WARNING', 'trigger': 'temperature_or_gas_threshold'}
        
        return {'state': 'NORMAL', 'trigger': 'below_threshold'}
    
    def get_performance_metrics(self) -> dict:
        """Calculate algorithm performance metrics"""
        return {
            'total_alerts': len(self.alert_history),
            'fire_alerts': self.fire_count,
            'warning_alerts': self.warning_count,
            'current_state': self.state,
            'config': self.config.to_dict()
        }
    
    def reset(self):
        """Reset algorithm state"""
        self.state = "NORMAL"
        self.verification_buffer = []
        self.warning_count = 0
        self.fire_count = 0
        self.alert_history = []

# Test function
def test_algorithm():
    print("=" * 60)
    print("TESTING FIRE DETECTION ALGORITHM")
    print("=" * 60)
    
    config = ThresholdConfig(
        temp_threshold=60.0,
        temp_warning=45.0,
        gas_threshold=0.7,
        gas_warning=0.4
    )
    
    algorithm = FireDetectionAlgorithm(config)
    
    # Test sequence
    test_samples = [
        (25, 0.1),   # Normal
        (30, 0.2),   # Normal
        (50, 0.3),   # Warning
        (65, 0.8),   # Fire candidate
        (70, 0.9),   # Fire candidate
        (75, 1.0),   # Fire candidate
    ]
    
    print("Processing test samples:")
    for i, (temp, gas) in enumerate(test_samples):
        result = algorithm.process_sample(temp, gas)
        print(f"  Sample {i+1}: Temp={temp}°C, Gas={gas} → State: {result['state']}")
    
    metrics = algorithm.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2))
    
    print("\n✅ Algorithm test complete!")

if __name__ == "__main__":
    test_algorithm()