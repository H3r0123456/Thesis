# data/generate_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class DataSimulator:
    def __init__(self, duration_hours=2, sample_rate_min=2):
        self.duration = duration_hours
        self.sample_rate = sample_rate_min
        self.total_samples = duration_hours * 60 // sample_rate_min
        
    def generate_normal_data(self, base_temp=25, base_gas=0.1):
        """Generate normal household conditions"""
        time_index = pd.date_range(
            start=datetime.now(),
            periods=self.total_samples,
            freq=f'{self.sample_rate}T'
        )
        
        # Normal temperature variations
        temp = base_temp + np.random.normal(0, 1.5, self.total_samples)
        
        # Normal gas levels (low)
        gas = base_gas + np.abs(np.random.normal(0, 0.05, self.total_samples))
        
        return pd.DataFrame({
            'timestamp': time_index,
            'temperature': np.clip(temp, 20, 40),
            'gas': np.clip(gas, 0, 0.5),
            'label': 'normal'
        })
    
    def generate_fire_scenario(self, base_data, start_idx, duration_samples=30):
        """Add fire event to normal data"""
        df = base_data.copy()
        fire_samples = min(duration_samples, len(df) - start_idx)
        
        if fire_samples > 0:
            time_factor = np.linspace(0, 1, fire_samples)
            
            for i in range(fire_samples):
                idx = start_idx + i
                df.loc[idx, 'temperature'] = 60 + 40 * time_factor[i] + np.random.normal(0, 2)
                df.loc[idx, 'gas'] = 0.7 + 0.3 * time_factor[i] + np.random.normal(0, 0.1)
                df.loc[idx, 'label'] = 'fire'
        
        return df
    
    def create_test_dataset(self, scenario_name="test_scenario"):
        """Create complete test dataset"""
        print(f"Generating {self.total_samples} samples...")
        
        # Generate normal data
        base = self.generate_normal_data()
        
        # Add fire event (start at 25% of dataset)
        fire_start = self.total_samples // 4
        base = self.generate_fire_scenario(base, fire_start, 30)
        
        # Add warning event (start at 60% of dataset)
        warning_start = int(self.total_samples * 0.6)
        for i in range(10):
            if warning_start + i < len(base):
                base.loc[warning_start + i, 'temperature'] = 50 + np.random.normal(0, 3)
                base.loc[warning_start + i, 'label'] = 'warning'
        
        # Ensure datasets folder exists
        os.makedirs("data/datasets", exist_ok=True)
        
        # Save dataset
        filename = f"data/datasets/{scenario_name}.csv"
        base.to_csv(filename, index=False)
        print(f"✅ Dataset saved: {filename}")
        print(f"   Samples: {len(base)}")
        print(f"   Normal samples: {(base['label'] == 'normal').sum()}")
        print(f"   Fire samples: {(base['label'] == 'fire').sum()}")
        print(f"   Warning samples: {(base['label'] == 'warning').sum()}")
        
        return base

if __name__ == "__main__":
    simulator = DataSimulator(duration_hours=2, sample_rate_min=2)
    dataset = simulator.create_test_dataset("demo_dataset")
    print(f"\nFirst 5 rows:")
    print(dataset.head())