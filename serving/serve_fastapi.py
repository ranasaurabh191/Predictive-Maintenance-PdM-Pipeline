
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
from lstm_ae import LSTMAutoencoder
import joblib

app = FastAPI()

class PredictInput(BaseModel):
    data: list

model = LSTMAutoencoder(n_features=13)
model.load_state_dict(torch.load("lstm_ae_model.pth"))
model.eval()
threshold = float(open("threshold.txt").read())
scaler = joblib.load("data/processed/scaler.joblib")

@app.post("/predict")
def predict(input_data: PredictInput):
    try:
        data = np.array(input_data.data)
        data = scaler.transform(data.reshape(-1, 13)).reshape(data.shape)
        input_tensor = torch.tensor(data).float()
        expected_shape = (128, 13)
        if len(input_tensor.shape) != 3 or input_tensor.shape[1:] != expected_shape:
            raise HTTPException(status_code=400, detail=f"Expected shape (batch, {expected_shape[0]}, {expected_shape[1]}), got {input_tensor.shape}")
        with torch.no_grad():
            recon = model(input_tensor)
            error = torch.mean((recon - input_tensor)**2).item()
        is_anomaly = error > threshold
        return {"error": error, "is_anomaly": is_anomaly}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)