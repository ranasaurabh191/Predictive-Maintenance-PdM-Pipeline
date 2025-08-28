import requests
import numpy as np
import json

# Create a sample sequence (shape: 1, 128, 13)
sample_seq = np.random.randn(1, 128, 13).tolist()  # 13 features
payload = {"data": sample_seq}
print("Payload shape:", np.array(payload["data"]).shape)  # Debug shape

try:
    response = requests.post("http://localhost:8000/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")