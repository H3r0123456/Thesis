# src/notification.py
import json
from datetime import datetime

class AlertSimulator:
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.sent_alerts = []
        
    def send_fire_alert(self, temperature: float, gas: float, location: str = "Test Area"):
        """Send fire alert notification"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'FIRE_ALERT',
            'temperature': temperature,
            'gas_level': gas,
            'location': location,
            'message': f"🚨 FIRE DETECTED! Temperature: {temperature:.1f}°C, Gas: {gas:.2f}"
        }
        
        print(f"[FIRE ALERT] {alert_data['message']}")
        self._log_alert(alert_data)
        return True
    
    def send_warning_alert(self, temperature: float, gas: float, location: str = "Test Area"):
        """Send warning alert"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'WARNING',
            'temperature': temperature,
            'gas_level': gas,
            'location': location,
            'message': f"⚠️ WARNING: Temperature: {temperature:.1f}°C, Gas: {gas:.2f}"
        }
        
        print(f"[WARNING] {alert_data['message']}")
        self._log_alert(alert_data)
        return True
    
    def _log_alert(self, alert_data):
        """Log alert to file"""
        self.sent_alerts.append(alert_data)
        
        # Ensure logs directory exists
        import os
        os.makedirs("results/logs", exist_ok=True)
        
        # Save to log file
        log_file = "results/logs/alerts.json"
        try:
            with open(log_file, 'r') as f:
                existing_logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_logs = []
        
        existing_logs.append(alert_data)
        
        with open(log_file, 'w') as f:
            json.dump(existing_logs, f, indent=2, default=str)
    
    def get_alert_statistics(self):
        """Get statistics about sent alerts"""
        return {
            'total_alerts': len(self.sent_alerts),
            'fire_alerts': sum(1 for a in self.sent_alerts if a['type'] == 'FIRE_ALERT'),
            'warning_alerts': sum(1 for a in self.sent_alerts if a['type'] == 'WARNING'),
            'latest_alert': self.sent_alerts[-1] if self.sent_alerts else None
        }

if __name__ == "__main__":
    # Test the notification system
    alert_system = AlertSimulator()
    
    # Send test alerts
    alert_system.send_warning_alert(50.5, 0.45)
    alert_system.send_fire_alert(70.2, 0.85)
    
    stats = alert_system.get_alert_statistics()
    print(f"\nAlert Statistics: {stats}")