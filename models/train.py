import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from lstm_ae import LSTMAutoencoder

sequences = np.load("data/processed/sequences.npy").astype(np.float32)
n_features = sequences.shape[-1]
train_dataset = TensorDataset(torch.tensor(sequences))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = LSTMAutoencoder(n_features=13)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
best_loss = float('inf')
patience = 10
trigger_times = 0

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        batch = batch[0]
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss {avg_loss:.6f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0
        torch.save(model.state_dict(), "lstm_ae_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

model.eval()
train_errors = []
with torch.no_grad():
    for batch in train_loader:
        batch = batch[0]
        recon = model(batch)
        error = torch.mean((recon - batch)**2, dim=(1, 2)).numpy()
        train_errors.extend(error)
threshold = np.percentile(train_errors, 90)
print(f"Anomaly Threshold: {threshold}")

with open("threshold.txt", "w") as f:
    f.write(str(threshold))

example_input = torch.randn(1, 128, n_features)
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("Model exported to model.onnx")