import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import json

# Configure logging
logging.basicConfig(filename='monitoring/custom_monitor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
df = pd.read_csv("data/processed/features.csv")
total_rows = len(df)

# Dynamically split data
if total_rows < 100:
    raise ValueError("Not enough data for monitoring (need at least 100 rows)")
split_point = min(1000, int(total_rows * 0.7))  # 70% for reference, rest for current
reference_data = df.iloc[:split_point].copy()
current_data = df.iloc[split_point:].copy()

# Compute basic statistics
ref_stats = reference_data[['temperature', 'vibration']].describe()
curr_stats = current_data[['temperature', 'vibration']].describe()

# Detect drift (simple mean/std comparison)
drift_detected = False
for col in ['temperature', 'vibration']:
    mean_diff = abs(ref_stats.loc['mean', col] - curr_stats.loc['mean', col])
    std_diff = abs(ref_stats.loc['std', col] - curr_stats.loc['std', col])
    if not np.isnan(mean_diff) and not np.isnan(std_diff):  # Check for NaN
        if mean_diff > 0.5 or std_diff > 0.5:  # Threshold for drift
            drift_detected = True
            logging.warning(f"Drift detected in {col}: Mean diff {mean_diff:.3f}, Std diff {std_diff:.3f}")

# Log metrics
logging.info(f"Total rows: {total_rows}, Split point: {split_point}")
logging.info(f"Drift detected: {drift_detected}")
logging.info(f"Reference mean: {ref_stats.loc['mean'].to_dict()}")
logging.info(f"Current mean: {curr_stats.loc['mean'].to_dict()}")
logging.info(f"Reference std: {ref_stats.loc['std'].to_dict()}")
logging.info(f"Current std: {curr_stats.loc['std'].to_dict()}")

# Generate simple plot
plt.figure(figsize=(10, 5))
plt.plot(reference_data['timestamp'], reference_data['temperature'], label='Reference Temp')
plt.plot(current_data['timestamp'], current_data['temperature'], label='Current Temp')
plt.title("Temperature Trend with Drift Check")
plt.xlabel("Timestamp")
plt.ylabel("Temperature")
plt.legend()
plt.savefig('monitoring/temp_trend.png')
plt.close()

# Save stats to JSON for later analysis
stats = {
    "drift_detected": drift_detected,
    "total_rows": total_rows,
    "split_point": split_point,
    "reference_stats": ref_stats.to_dict(),
    "current_stats": curr_stats.to_dict()
}
with open('monitoring/monitor_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Monitoring complete. Check custom_monitor.log, temp_trend.png, and monitor_stats.json.")