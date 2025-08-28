import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from lstm_ae import LSTMAutoencoder

# ------------------------
# Load sequences and model
# ------------------------
sequences = np.load("data/processed/sequences.npy").astype(np.float32)
test_seq = sequences[-1000:]  # last 1000 sequences for evaluation

n_features = test_seq.shape[2]
model = LSTMAutoencoder(n_features=n_features)
model.load_state_dict(torch.load("lstm_ae_model.pth"))
model.eval()

# ------------------------
# Compute reconstruction errors
# ------------------------
errors = []
with torch.no_grad():
    for seq in test_seq:
        input_tensor = torch.tensor(seq).float().unsqueeze(0)  # shape [1, seq_len, n_features]
        recon = model(input_tensor)
        error = torch.mean((recon - input_tensor)**2).item()
        errors.append(error)

errors_np = np.array(errors)

# ------------------------
# Compute dynamic threshold
# ------------------------
threshold = errors_np.mean() + 2 * errors_np.std()  # 2-sigma threshold
# Optional: percentile-based threshold
# threshold = np.percentile(errors_np, 95)

# ------------------------
# Detect anomalies
# ------------------------
anomaly_indices = np.where(errors_np > threshold)[0]
num_anomalies = len(anomaly_indices)

# ------------------------
# Print summary
# ------------------------
print(f"Mean Error: {errors_np.mean():.4f}")
print(f"Std Error: {errors_np.std():.4f}")
print(f"Anomalies Detected: {num_anomalies}/{len(test_seq)} ({num_anomalies/len(test_seq)*100:.2f}%)")
print(f"Threshold: {threshold:.4f}")
print(f"Anomaly indices: {anomaly_indices}")

# ------------------------
# Plot error distribution
# ------------------------
plt.figure(figsize=(10,6))
plt.hist(errors_np, bins=50, density=True, alpha=0.7, color="skyblue")
plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')

# Highlight anomalies on histogram
plt.scatter(errors_np[anomaly_indices], np.zeros_like(anomaly_indices), color='red', marker='x', label='Anomalies')

plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.show()

# ------------------------
# Save threshold to file
# ------------------------
with open("threshold.txt", "w") as f:
    f.write(f"{threshold:.4f}")
