# 🔧 Predictive Maintenance (PdM) Pipeline

## Overview
Welcome to the **Predictive Maintenance (PdM) Pipeline**! This project leverages an **LSTM Autoencoder** to detect anomalies in synthetic sensor data (temperature, vibration, RPM, pressure) from industrial assets.  

The pipeline includes:  
- Data ingestion  
- Model training  
- RESTful API with **FastAPI**  
- Real-time dashboard with **Streamlit**  
- Custom monitoring for data drift  
- MQTT integration for IoT streaming  
- Docker containerization for deployment  

Designed for **scalability and extensibility**, this project serves as a foundation for industrial asset health management.

---

## ✨ Features
- **Data Ingestion:** Generates and processes synthetic sensor data with injected anomalies.  
- **Model Training:** Trains an LSTM Autoencoder to identify deviations from normal behavior.  
- **API Serving:** Exposes a FastAPI endpoint for real-time anomaly predictions.  
- **Dashboard:** Interactive Streamlit interface to visualize trends and anomalies.  
- **Monitoring:** Custom drift detection with logs, plots, and JSON statistics.  
- **IoT Integration:** Supports MQTT for real-time sensor streaming.  
- **Containerization:** Dockerized for seamless deployment across environments.  

---

## ⚙️ Prerequisites
Before setting up the project, ensure you have:

- 🐍 **Python 3.10 or 3.11**  
- 📦 **pip** (latest version)  
- 🐙 **Git** (optional)  
- 🐳 **Docker** (optional)  
- 📡 **Mosquitto** (optional, for MQTT streaming)  

---


## Project Structure

textpdmp_project/
│
├── data/
│   ├── raw/                # Raw synthetic sensor data (e.g., synthetic_sensor_data.csv)
│   └── processed/          # Processed data (e.g., features.csv, sequences.npy, scaler.joblib)
│
├── models/                 # Model definitions and training scripts
│   ├── lstm_ae.py          # LSTM Autoencoder model architecture
│   ├── train.py            # Model training and threshold calculation
│
├── data_ingest/            # Data generation and ingestion
│   ├── simulate_ingestion.py  # Script to generate synthetic data
│
├── serving/                # API serving
│   ├── serve_fastapi.py    # FastAPI server implementation
│
├── dashboard.py            # Streamlit dashboard for visualization
├── test_api.py             # Script to test the API endpoint
├── monitoring/             # Custom monitoring scripts and outputs
│   ├── custom_monitor.py   # Drift detection and logging
│   ├── custom_monitor.log  # Monitoring logs
│   ├── temp_trend.png      # Temperature trend visualization
│   ├── monitor_stats.json  # Statistical monitoring data
│
├── notebooks/              # Jupyter notebooks for exploratory data analysis
│   ├── EDA_and_baselines.ipynb  # Data preprocessing and baseline analysis
│
├── Dockerfile              # Docker configuration file
├── requirements.txt        # List of Python dependencies
└── README.md               # This file

## ⚙️ Prerequisites
Before setting up the project, ensure you have the following installed:

🐍 Python 3.10 or 3.11 (recommended for compatibility)
📦 pip (latest version, included with Python)
🐙 Git (optional, for version control and cloning the repository)
🐳 Docker (optional, for containerized deployment)
📡 Mosquitto (optional, for MQTT-based IoT streaming)

## 📥 Installation
1. Clone the Repository
Clone this repository to your local machine:
bashgit clone https://github.com/your-username/pdmp_project.git
cd pdmp_project
Note: Replace your-username with your GitHub username or the repository URL.
2. Set Up a Virtual Environment
Create and activate a virtual environment to isolate dependencies:
bashpython -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
3. Install Dependencies
Install the required Python packages listed in requirements.txt:
bashpip install -r requirements.txt
The requirements.txt includes:

numpy==2.2.6
pandas==2.3.2
torch==2.8.0
scikit-learn==1.7.1
joblib==1.5.2
fastapi==0.116.1
uvicorn==0.35.0
streamlit==1.49.0
matplotlib==3.10.5
paho-mqtt==2.1.0
shap==0.48.0
requests==2.32.5
pillow==11.3.0
typing-extensions==4.15.0

4. (Optional) Install Mosquitto for MQTT
To enable MQTT streaming, install Mosquitto and verify it’s running:
bashmosquitto -v
Download Mosquitto from mosquitto.org if not already installed.
🚀 Usage
1. Generate and Process Data
Generate synthetic sensor data and process it for model training:
bashpython data_ingest/simulate_ingestion.py
jupyter notebook notebooks/EDA_and_baselines.ipynb
Run all cells in the notebook to create features.csv and sequences.npy.
2. Train the Model
Train the LSTM Autoencoder and set the anomaly threshold:
bashpython models/train.py
This generates lstm_ae_model.pth, threshold.txt, and model.onnx.
3. Evaluate the Model
Assess the model’s performance on the test dataset:
bashpython evaluate_model.py
Expected Output:
yamlMean Error: ~0.3156
Std Error: ~0.2103
Anomalies Detected: ~7.24%
Threshold: ~0.7363
Anomaly indices: [756 757 ... 814]
4. Serve Predictions (API)
Start the FastAPI server and test the endpoint:
bashpython serving/serve_fastapi.py
python test_api.py
Expected: Status Code: 200 from the test script.
5. Launch Dashboard
Run the Streamlit dashboard to visualize results:
bashstreamlit run dashboard.py
👉 Visit http://localhost:8501 to interact with the dashboard.
6. Run Monitoring
Monitor data drift and generate visualizations:
bashpython monitoring/custom_monitor.py
Check the outputs in the monitoring/ directory:

📜 custom_monitor.log: Logs drift detection and statistics.
📈 temp_trend.png: Temperature trend visualization.
📑 monitor_stats.json: Detailed statistical data.

7. IoT Streaming (MQTT)
Enable real-time data streaming with MQTT:
bashmosquitto -v
python data_ingest/simulate_ingestion.py
The dashboard will automatically subscribe and display anomalies.
8. Deploy with Docker
Build and run the Docker container for deployment:
bashdocker build -t pdmp_app .
docker run -p 8000:8000 pdmp_app
Access the API at http://localhost:8000.
📊 Performance Metrics

✅ Mean Error: ~0.3156 (indicates effective reconstruction of normal data).
📉 Std Error: ~0.2103 (reflects consistent model performance).
🚨 Anomaly Detection Rate: ~7.24% (closely matches the injected 5% anomaly rate).
🔑 Threshold: ~0.7363 (90th percentile of training reconstruction errors).

🔮 Future Enhancements

🔗 Real Data Integration: Incorporate live IoT sensor data.
📊 Advanced Monitoring: Add statistical tests (e.g., Kolmogorov-Smirnov, Population Stability Index).
⚙️ Model Optimization: Tune hyperparameters or explore alternative models (e.g., GRU).
☁️ Cloud Deployment: Extend to Kubernetes or cloud platforms.
🔀 Multi-Asset Support: Implement load balancing for multiple assets.

🤝 Contributing
We welcome contributions, issues, and feature requests! Follow these steps:

Fork the repository.
Create a feature branch: git checkout -b feature/new-feature.
Commit your changes: git commit -m 'Add new feature'.
Push to the branch: git push origin feature/new-feature.
Open a Pull Request.

## Inspired by real-world PdM applications in industrial asset management.
