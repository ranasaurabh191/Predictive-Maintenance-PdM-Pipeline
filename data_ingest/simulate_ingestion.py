import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
dates = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(10000)]
data = {
    'timestamp': dates,
    'asset_id': ['A001'] * 10000,
    'temperature': np.random.normal(70, 5, 10000),
    'vibration': np.random.normal(0.5, 0.1, 10000),
    'rpm': np.random.normal(1200, 100, 10000),
    'pressure': np.random.normal(100, 10, 10000)
}
df = pd.DataFrame(data)

anomaly_indices = np.random.choice(10000, size=500, replace=False)
df.loc[anomaly_indices, 'temperature'] *= 1.5
df.loc[anomaly_indices, 'vibration'] *= 2.0
df.to_csv("data/raw/synthetic_sensor_data.csv", index=False)
print("Synthetic data with anomalies saved")    