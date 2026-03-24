"""
Streamlit Dashboard for ChemSense Chemical Process Monitoring
Real-time anomaly detection and fault prediction interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from pathlib import Path

# Import our modules
from src.config import (
    MODEL_PATHS, FEATURE_COLS, FAULT_LABELS, NORMAL_RANGES
)
from src.utils import load_model, validate_input_features, validate_feature_bounds, setup_logger

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logger = setup_logger(__name__)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChemSense Monitor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS (Professional Dark Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #0a0d12;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #0f1319;
    border-right: 1px solid #1e2530;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}

[data-testid="metric-container"] {
    background: #0f1319;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 12px 16px;
}

.stButton > button[kind="primary"] {
    background: #00e5a0 !important;
    color: #0a0d12 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
}

.stButton > button[kind="primary"]:hover {
    background: #00ffb3 !important;
    box-shadow: 0 8px 24px rgba(0,229,160,0.3) !important;
}

.stSuccess {
    background: rgba(0,229,160,0.1) !important;
    border: 1px solid rgba(0,229,160,0.3) !important;
    border-radius: 10px !important;
    color: #00e5a0 !important;
}

.stError {
    background: rgba(255,77,109,0.1) !important;
    border: 1px solid rgba(255,77,109,0.3) !important;
    border-radius: 10px !important;
    color: #ff4d6d !important;
}

.stWarning {
    background: rgba(255,194,48,0.1) !important;
    border: 1px solid rgba(255,194,48,0.3) !important;
    border-radius: 10px !important;
    color: #ffc230 !important;
}

.metric-card {
    background: #0f1319;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}

# ─────────────────────────────────────────────
#  MODEL LOADING WITH IMPROVED ERROR HANDLING
# ─────────────────────────────────────────────
@st.cache_resource
def load_all_models(model_dir: str = "models") -> tuple:
    """Load all trained models with comprehensive error handling"""
    
    loaded = {
        'isolation_forest': None,
        'random_forest': None,
        'xgboost': None,
        'scaler_anomaly': None,
        'scaler_fault': None,
    }
    
    status_messages = []
    
    # Try loading each model
    for model_name, path in MODEL_PATHS.items():
        # Clean up path by taking just the filename, so it respects model_dir
        filename = os.path.basename(path)
        model_path = os.path.join(model_dir, filename)
        
        try:
            # First check if reconstructed path exists, otherwise use fallback logic from load_model
            if os.path.exists(model_path):
                model = load_model(model_path)
            else:
                model = load_model(path) # Use dictionary path which triggers fallback mapping

            if model is None:
                status_messages.append((model_name, "❌ Load failed", False))
            else:
                loaded[model_name] = model
                status_messages.append((model_name, "✅ Loaded", True))
        except Exception as e:
            status_messages.append((model_name, f"❌ {str(e)[:30]}", False))
            logger.error(f"Failed to load {model_name}: {e}")
    
    return loaded, status_messages

# ─────────────────────────────────────────────
#  SIDEBAR - MODEL LOADING
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ ChemSense")
    st.markdown('<p style="font-size:0.9rem; color:#888;">Advanced Process Monitoring</p>', unsafe_allow_html=True)
    st.divider()
    
    # Model directory selection
    st.markdown("**📁 Model Directory**")
    model_dir = st.text_input(
        "Path to .pkl models",
        value="models",
        label_visibility="collapsed",
        placeholder="e.g., ./models"
    )
    
    if st.button("🔄 Load Models", use_container_width=True):
        with st.spinner("Loading models..."):
            models, status = load_all_models(model_dir)
            st.session_state.models = models
            
            # Show status
            st.divider()
            st.markdown("**Load Status**")
            for model_name, message, success in status:
                color = "green" if success else "red"
                st.markdown(f'<p style="color:{color}">{model_name}: {message}</p>', unsafe_allow_html=True)
            
            loaded_count = sum(1 for v in models.values() if v is not None)
            if loaded_count >= 3:
                st.success(f"✅ {loaded_count}/5 models ready!")
            else:
                st.warning(f"⚠️ Only {loaded_count}/5 models loaded")
    
    # Model status indicator
    st.divider()
    st.markdown("**Model Status**")
    for key, model in st.session_state.models.items():
        status = "🟢" if model is not None else "🔴"
        st.markdown(f"{status} {key}", help="Green = Ready, Red = Not loaded")
    
    st.divider()
    st.markdown(f"**Predictions**: {len(st.session_state.history)}")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
col_h1, col_h2, col_h3, col_h4 = st.columns(4)

total_preds = len(st.session_state.history)
anomalies = sum(1 for h in st.session_state.history if h.get('is_anomaly'))
faults = sum(1 for h in st.session_state.history if h.get('is_fault'))
normals = total_preds - anomalies - faults

col_h1.metric("Total Predictions", total_preds)
col_h2.metric("🚨 Anomalies", anomalies)
col_h3.metric("⚠️ Faults", faults)
col_h4.metric("✅ Normal", normals)

st.divider()

# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Real-Time Prediction", "📊 Batch Analysis", "📋 History"])

# ══════════════════════════════════════════════
#  TAB 1: REAL-TIME PREDICTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Live Sensor Input")
    st.markdown("Enter current sensor readings for real-time prediction")
    
    # Input columns
    c1, c2, c3 = st.columns(3)
    
    input_data = {}
    
    with c1:
        st.markdown("**Temperature & Pressure**")
        input_data['ambient_temp_effect'] = st.number_input("Ambient Temp Effect (°C)", value=0.0, step=0.1)
        input_data['reactor_temp'] = st.number_input("Reactor Temp (°C)", value=85.0, step=0.1)
        input_data['reactor_pressure'] = st.number_input("Reactor Pressure (bar)", value=2.1, step=0.01)
        input_data['temp_setpoint'] = st.number_input("Temp Setpoint (°C)", value=85.0, step=0.5)
        input_data['pressure_setpoint'] = st.number_input("Pressure Setpoint (bar)", value=2.1, step=0.01)
    
    with c2:
        st.markdown("**Flow & Speed**")
        input_data['feed_flow_rate'] = st.number_input("Feed Flow (L/h)", value=120.0, step=1.0)
        input_data['coolant_flow_rate'] = st.number_input("Coolant Flow (L/h)", value=45.0, step=0.5)
        input_data['agitator_speed_rpm'] = st.number_input("Agitator Speed (RPM)", value=250.0, step=5.0)
        input_data['vibration_rms'] = st.number_input("Vibration RMS (mm/s)", value=1.2, step=0.01)
        input_data['motor_current'] = st.number_input("Motor Current (A)", value=20.0, step=0.5)
    
    with c3:
        st.markdown("**Reaction Metrics**")
        input_data['reaction_rate'] = st.number_input("Reaction Rate", value=0.82, step=0.01)
        input_data['conversion_rate'] = st.number_input("Conversion Rate", value=0.85, step=0.01)
        input_data['selectivity'] = st.number_input("Selectivity", value=0.90, step=0.01)
        input_data['yield_pct'] = st.number_input("Yield (%)", value=80.0, step=0.5)
        input_data['power_consumption_kw'] = st.number_input("Power (kW)", value=10.0, step=0.5)
    
    st.markdown("")
    
    # Action buttons
    col_run, col_validate, col_clear = st.columns([2, 2, 1])
    
    with col_run:
        run_btn = st.button("▶ Run Prediction", type="primary", use_container_width=True)
    
    with col_validate:
        validate_btn = st.button("✓ Validate Inputs", use_container_width=True)
    
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    # ── Validate Inputs ──
    if validate_btn:
        violations = validate_feature_bounds(pd.DataFrame([input_data]), NORMAL_RANGES)
        out_of_bounds = [col for col in violations.columns if violations[col].any()]
        
        if not out_of_bounds:
            st.success("✅ All sensors within normal ranges!")
        else:
            st.warning(f"⚠️ {len(out_of_bounds)} sensors out of normal range:")
            for sensor in out_of_bounds:
                lo, hi, unit = NORMAL_RANGES.get(sensor, (None, None, ''))
                value = input_data.get(sensor, 'N/A')
                st.write(f"  • {sensor}: {value} {unit} (expected {lo}–{hi})")
    
    # ── Run Prediction ──
    if run_btn:
        if st.session_state.models['isolation_forest'] is None:
            st.error("❌ Please load models from the sidebar first!")
        else:
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Select available features
            feature_cols_available = [c for c in FEATURE_COLS if c in input_df.columns]
            for col in feature_cols_available:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            
            X = input_df[feature_cols_available].values.astype(float)
            
            # ── Anomaly Detection ──
            with st.spinner("Running inference..."):
                iso_model = st.session_state.models['isolation_forest']
                scaler = st.session_state.models['scaler_anomaly']
                
                try:
                    if scaler:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X
                    
                    anomaly_pred = iso_model.predict(X_scaled)[0]
                    anomaly_score = iso_model.decision_function(X_scaled)[0]
                    is_anomaly = anomaly_pred == -1
                    
                    # ── Fault Prediction ──
                    rf_model = st.session_state.models.get('random_forest')
                    xgb_model = st.session_state.models.get('xgboost')
                    scaler_fault = st.session_state.models.get('scaler_fault')
                    
                    is_fault = False
                    fault_label = "None"
                    
                    if rf_model and is_anomaly:
                        try:
                            X_fault_scaled = scaler_fault.transform(X) if scaler_fault else X
                            fault_pred = rf_model.predict(X_fault_scaled)[0]
                            fault_prob = rf_model.predict_proba(X_fault_scaled)[0] if hasattr(rf_model, 'predict_proba') else None
                            
                            try:
                                # Try if it's an integer index
                                fault_index = int(fault_pred)
                                is_fault = fault_index != 0
                                fault_label = FAULT_LABELS[fault_index] if fault_index < len(FAULT_LABELS) else str(fault_index)
                            except ValueError:
                                # It's already a string label
                                fault_label = str(fault_pred)
                                is_fault = fault_label not in ["Normal Operation", "None", "0", "0.0"]
                            
                            
                        except Exception as e:
                            logger.error(f"Fault prediction error: {e}")

                    # Results
                    st.divider()
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        st.markdown("**ANOMALY DETECTION**")
                        if is_anomaly:
                            st.toast('🚨 Anomaly detected! Check dashboard.', icon='🚩')
                            
                            st.error("🚨 **CRITICAL WARNING** - Anomaly Detected!")
                            
                            if is_fault:
                                st.warning(f"🔧 **DIAGNOSIS:** {fault_label}")
                                if fault_prob is not None:
                                    max_prob = max(fault_prob) * 100
                                    st.write(f"*Confidence: {max_prob:.1f}%*")
                            
                            # Add an engaging risk percentage metric based on score
                            risk_percentage = max(0, min(100, int((0.5 - anomaly_score) * 100)))
                            st.metric(label="System Health", value="UNSTABLE", delta=f"-{risk_percentage}% Critical", delta_color="inverse")
                            st.progress(risk_percentage, text=f"Risk Level: {risk_percentage}%")
                            
                            # ✨ Root Cause Analysis
                            st.markdown("#### 🔍 Expected Root Causes")
                            deviations = []
                            for k, v in input_data.items():
                                if k in NORMAL_RANGES:
                                    lo, hi, unit = NORMAL_RANGES[k]
                                    if v < lo or v > hi:
                                        direction = "HIGH ⬆️" if v > hi else "LOW ⬇️"
                                        deviations.append(f"**{k.replace('_', ' ').title()}**: {v:.2f} {unit} ({direction})")
                            
                            if deviations:
                                st.markdown("\n".join([f"- {d}" for d in deviations]))
                            else:
                                st.info("Process metrics are near boundaries but compounding into an anomaly.")
                            
                        else:
                            st.toast('✅ Process normal.', icon='🧪')
                            
                            st.success("✅ **SYSTEM STABLE** - Normal operations")
                            st.metric("Decision Score", f"{anomaly_score:.4f}", delta="Optimal", delta_color="normal")
                            st.balloons()
                    
                    with res_col2:
                        st.markdown("**Sensor Status**")
                        status_data = []
                        for k, v in input_data.items():
                            if k in NORMAL_RANGES:
                                lo, hi, unit = NORMAL_RANGES[k]
                                in_range = lo <= v <= hi
                                status_data.append({
                                    'Sensor': k.replace('_', ' ').title(),
                                    'Value': f"{v:.2f} {unit}",
                                    'Range': f"{lo}–{hi}",
                                    'Status': '✅' if in_range else '⚠️'
                                })
                        if status_data:
                            st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)
                    
                    # Store result
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'model': 'Isolation Forest',
                        'result': 'ANOMALY' if is_anomaly else 'NORMAL',
                        'fault_type': fault_label if is_anomaly else 'None',
                        'score': float(anomaly_score),
                        'confidence': max(fault_prob)*100 if is_anomaly and 'fault_prob' in locals() and fault_prob is not None else None,
                        'is_anomaly': is_anomaly,
                        'is_fault': is_fault if 'is_fault' in locals() else False,
                    })
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    logger.error(f"Prediction error: {e}", exc_info=True)

# ══════════════════════════════════════════════
#  TAB 2: BATCH ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Prediction from CSV")
    st.markdown("Upload a CSV file for bulk analysis")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.info(f"📊 Loaded {len(df_upload)} rows × {len(df_upload.columns)} columns")
        st.dataframe(df_upload.head(), use_container_width=True)
        
        if st.button("▶ Run Batch Predictions", type="primary", use_container_width=True):
            iso_model = st.session_state.models['isolation_forest']
            scaler = st.session_state.models['scaler_anomaly']
            
            if iso_model is None:
                st.error("❌ Load models first!")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Prepare features
                        feature_cols_avail = [c for c in FEATURE_COLS if c in df_upload.columns]
                        for col in feature_cols_avail:
                            if col not in df_upload.columns:
                                df_upload[col] = 0.0
                        
                        X_batch = df_upload[feature_cols_avail].fillna(df_upload[feature_cols_avail].mean()).values.astype(float)
                        
                        if scaler:
                            X_batch = scaler.transform(X_batch)

                        # Predict
                        preds = iso_model.predict(X_batch)
                        scores = iso_model.decision_function(X_batch)
                        
                        df_upload['prediction'] = preds
                        df_upload['anomaly_score'] = scores
                        df_upload['result'] = df_upload['prediction'].map({1: 'NORMAL', -1: 'ANOMALY'})
                        
                        # Show metrics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Total Rows", len(df_upload))
                        col_m2.metric("Anomalies Found", int((preds == -1).sum()))
                        col_m3.metric("Anomaly Rate", f"{(preds == -1).mean()*100:.1f}%")
                        
                        # Results
                        st.markdown("**Prediction Results**")
                        st.dataframe(df_upload, use_container_width=True)
                        
                        # Download
                        csv_out = df_upload.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Results",
                            data=csv_out,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Batch processing failed: {e}")
                        logger.error(f"Batch error: {e}", exc_info=True)

# ══════════════════════════════════════════════
#  TAB 3: PREDICTION HISTORY
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Prediction History")
    
    if not st.session_state.history:
        st.info("📭 No predictions yet. Go to the 'Real-Time Prediction' tab to start.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        if len(st.session_state.history) >= 2:
            st.markdown("**Result Distribution**")
            results = [h['result'] for h in st.session_state.history]
            result_counts = pd.Series(results).value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0a0d12')
            ax.set_facecolor('#0f1319')
            
            colors = ['#2ecc71' if 'NORMAL' in r else '#e74c3c' for r in result_counts.index]
            result_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='none', width=0.6)
            ax.set_xlabel('Status', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.tick_params(colors='#888', labelsize=9)
            ax.spines['left'].set_color('#1e2530')
            ax.spines['bottom'].set_color('#1e2530')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()
