# ChemSense Monitor
An AI-driven industrial monitoring system for anomaly detection and fault prediction in chemical manufacturing processes.

## Overview
This repository contains an end-to-end Machine Learning pipeline that analyzes real-time sensor data from a chemical process (reactors, pumps, heat exchangers). It automatically detects invisible anomalies using unsupervised learning (Isolation Forest) and predicts the exact mechanical fault occurring using supervised models (XGBoost & Random Forest) before a catastrophic failure happens.

## Project Structure
- `data/`: Contains raw and cleaned CSV datasets.
- `models/`: Stores the trained `.pkl` models (Isolation Forest, Random Forest, XGBoost).
- `outputs/`: Generated visualizations and graphs like confusion matrices and feature importance.
- `src/`: Core Python modules for data cleaning, EDA, anomaly detection, and fault prediction.
- `app.py`: An interactive web dashboard built with Streamlit to monitor plant status in real-time.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
"# ChemSense-AI" 
