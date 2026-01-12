import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

# Test imports
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    import seaborn as sns
    print("✅ All libraries imported successfully!")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")