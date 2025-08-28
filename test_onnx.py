import onnxruntime as ort
import numpy as np

# Load sequences to get n_features
sequences = np.load("data/processed/sequences.npy").astype(np.float32)
n_features = sequences.shape[-1]

# Test ONNX model
ort_session = ort.InferenceSession("model.onnx")
input_name = ort_session.get_inputs()[0].name
example = np.random.randn(1, 128, n_features).astype(np.float32)
recon = ort_session.run(None, {input_name: example})[0]
print("ONNX Recon shape:", recon.shape)