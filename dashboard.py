import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import joblib
import plotly.graph_objects as go

# ------------------------
# Title and setup
# ------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("⚙️ Predictive Maintenance Dashboard")

# Load data & scaler
df = pd.read_csv("data/processed/features.csv")
feature_cols = [col for col in df.columns if col not in ['timestamp', 'asset_id']]
st.write(f"Using features: {feature_cols}")
scaler = joblib.load("data/processed/scaler.joblib")

placeholder = st.empty()
anomaly_indices = []

# ------------------------
# Streaming simulation loop
# ------------------------
for i in range(0, len(df) - 128, 100):
    batch = df[feature_cols].iloc[i:i+128].values
    if batch.shape[0] != 128:
        continue

    batch_df = df[feature_cols].iloc[i:i+128]  # preserve columns
    batch_scaled = scaler.transform(batch_df)  # ✅ uses feature names
    seq = np.array([batch_scaled])

    try:
        payload = {"data": seq.tolist()}
        response = requests.post("http://localhost:8000/predict", json=payload)
        response.raise_for_status()
        response_data = response.json()

        with placeholder.container():
            col1, col2 = st.columns([2,1])

            # ------------------------
            # Left: Plot (Plotly instead of st.line_chart)
            # ------------------------
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df['temperature'].iloc[i:i+128],
                mode='lines',
                name='Temperature',
                line=dict(color="orange", width=2)
            ))
            fig.add_trace(go.Scatter(
                y=df['vibration'].iloc[i:i+128],
                mode='lines',
                name='Vibration',
                line=dict(color="royalblue", width=2)
            ))

            # Highlight anomaly point if detected
            if response_data['is_anomaly']:
                anomaly_point = i + 64
                fig.add_trace(go.Scatter(
                    x=[anomaly_point - i],
                    y=[df['temperature'].iloc[anomaly_point]],
                    mode="markers+text",
                    text=["⚠️ Anomaly"],
                    textposition="top center",
                    marker=dict(size=12, color="red", symbol="x"),
                    name="Anomaly"
                ))

            fig.update_layout(
                title="Sensor Readings (Temperature & Vibration)",
                xaxis_title="Time (sequence index)",
                yaxis_title="Sensor Value",
                template="plotly_dark",
                legend=dict(orientation="h", y=-0.2),
                height=400
            )
            col1.plotly_chart(fig, use_container_width=True)

            # ------------------------
            # Right: Status Card
            # ------------------------
            if response_data['is_anomaly']:
                col2.error("🚨 **Anomaly Detected!**")
                anomaly_indices.append(i + 64)
            else:
                col2.success("✅ Normal Operation")

            col2.metric("Reconstruction Error", f"{response_data['error']:.4f}")

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")

    st.empty()

# ------------------------
# Show anomaly summary
# ------------------------
st.write("📍 Anomaly indices detected:", anomaly_indices)

if anomaly_indices:
    summary_fig = go.Figure()
    summary_fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['temperature'],
        mode="lines", name="Temperature", line=dict(color="orange")
    ))
    summary_fig.add_trace(go.Scatter(
        x=df['timestamp'].iloc[anomaly_indices],
        y=df['temperature'].iloc[anomaly_indices],
        mode="markers", name="Anomalies",
        marker=dict(size=10, color="red", symbol="x")
    ))
    st.plotly_chart(summary_fig, use_container_width=True)