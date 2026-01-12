import os

required_files = [
    'data/generate_data.py',
    'src/algorithm.py',
    'src/evaluation.py',
    'src/notification.py',
    'tests/test_scenarios.py',
    'main_simulation.py',
    'requirements.txt'
]

print("Checking project structure...")
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} (MISSING)")

print(f"\nCurrent directory: {os.getcwd()}")