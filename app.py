import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

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
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0d12;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1319;
    border-right: 1px solid #1e2530;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}

/* Metric boxes */
[data-testid="metric-container"] {
    background: #0f1319;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 12px 16px;
}

[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    color: #4a5568 !important;
    text-transform: uppercase;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* Inputs */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background-color: #0f1319 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}

.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #00e5a0 !important;
    box-shadow: 0 0 0 1px #00e5a0 !important;
}

/* Labels */
.stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    color: #4a5568 !important;
    text-transform: uppercase;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #00e5a0 !important;
    color: #0a0d12 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.9rem !important;
    padding: 12px 28px !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton > button[kind="primary"]:hover {
    background: #00ffb3 !important;
    box-shadow: 0 8px 24px rgba(0,229,160,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Secondary button */
.stButton > button {
    background: #0f1319 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Radio buttons */
.stRadio label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #e2e8f0 !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1e2530 !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: #1e2530 !important;
}

/* Success / error / warning boxes */
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

/* Info */
.stInfo {
    background: rgba(99,179,237,0.1) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 10px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1319 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5568 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
}

.stTabs [aria-selected="true"] {
    background: #1e2530 !important;
    color: #00e5a0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f1319 !important;
    border: 1px dashed #1e2530 !important;
    border-radius: 10px !important;
}

/* Custom card class */
.result-card {
    background: #0f1319;
    border: 1px solid #1e2530;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}

.badge-normal  { background: rgba(0,229,160,0.15); color: #00e5a0; }
.badge-anomaly { background: rgba(255,77,109,0.15); color: #ff4d6d; }
.badge-fault   { background: rgba(255,194,48,0.15);  color: #ffc230; }

.mono { font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.muted { color: #4a5568; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
MODEL_FILES = {
    "Isolation Forest (Anomaly Detection)": "models/isolation_forest.pkl",
    "Random Forest (Fault Classification)": "models/random_forest_model.pkl",
    "XGBoost (Fault Classification)":       "models/xgboost_model.pkl",
}

SCALER_FILE = "models/scaler.pkl"

FEATURE_COLS = [
    'ambient_temp_effect',
    'reactor_temp',
    'reactor_pressure',
    'feed_flow_rate',
    'coolant_flow_rate',
    'agitator_speed_rpm',
    'reaction_rate',
    'conversion_rate',
    'selectivity',
    'yield_pct',
    'vibration_rms',
    'motor_current',
    'power_consumption_kw',
    'temp_setpoint',
    'pressure_setpoint',
]

FAULT_LABELS = [
    'Normal Operation', 'Sensor Drift', 'Valve Fault',
    'Heat Exchanger Fouling', 'Pump Cavitation',
    'Catalyst Deactivation', 'Pipe Blockage'
]

NORMAL_RANGES = {
    'ambient_temp_effect':  (-5.0,  5.0,   '°C'),
    'reactor_temp':         (60.0,  110.0, '°C'),
    'reactor_pressure':     (1.0,   5.0,   'bar'),
    'feed_flow_rate':       (80.0,  180.0, 'L/h'),
    'coolant_flow_rate':    (20.0,  80.0,  'L/h'),
    'agitator_speed_rpm':   (100.0, 500.0, 'RPM'),
    'reaction_rate':        (0.5,   1.2,   ''),
    'conversion_rate':      (0.6,   0.99,  ''),
    'selectivity':          (0.7,   1.0,   ''),
    'yield_pct':            (60.0,  99.0,  '%'),
    'vibration_rms':        (0.0,   3.0,   'mm/s'),
    'motor_current':        (5.0,   50.0,  'A'),
    'power_consumption_kw': (1.0,   30.0,  'kW'),
    'temp_setpoint':        (60.0,  110.0, '°C'),
    'pressure_setpoint':    (1.0,   5.0,   'bar'),
}

@st.cache_resource
def load_models(model_dir):
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception as e:
                loaded[name] = None
                st.warning(f"Could not load {fname}: {e}")
        else:
            loaded[name] = None
    scaler = None
    scaler_path = os.path.join(model_dir, SCALER_FILE)
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")
    return loaded, scaler

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ ChemSense")
    st.markdown('<p class="mono muted">Chemical Process Monitor</p>', unsafe_allow_html=True)
    st.divider()

    # Model directory
    st.markdown("**📁 Model Directory**")
    model_dir = st.text_input(
        "Path to .pkl files",
        value=os.getcwd(),
        label_visibility="collapsed",
        placeholder="e.g. C:/Users/you/project/"
    )

    if st.button("🔄 Load Models", use_container_width=True):
        with st.spinner("Loading models..."):
            models, scaler = load_models(model_dir)
            st.session_state.models  = models
            st.session_state.scaler  = scaler

        loaded_count = sum(1 for v in models.values() if v is not None)
        if loaded_count > 0:
            st.success(f"✅ {loaded_count}/{len(MODEL_FILES)} models loaded")
        else:
            st.error("No models found. Check the directory path.")

    # Show model status
    st.divider()
    st.markdown("**Model Status**")
    for name, fname in MODEL_FILES.items():
        short = name.split(" (")[0]
        loaded = st.session_state.models.get(name) is not None
        icon = "🟢" if loaded else "🔴"
        st.markdown(f'<p class="mono" style="font-size:0.7rem">{icon} {short}</p>', unsafe_allow_html=True)

    scaler_ok = st.session_state.scaler is not None
    st.markdown(f'<p class="mono" style="font-size:0.7rem">{"🟢" if scaler_ok else "🔴"} Scaler</p>', unsafe_allow_html=True)

    st.divider()

    # Model selection
    st.markdown("**Select Model**")
    available_models = [n for n, v in st.session_state.models.items() if v is not None]
    if available_models:
        selected_model = st.radio(
            "model_select",
            options=available_models,
            label_visibility="collapsed"
        )
    else:
        st.info("Load models first ↑")
        selected_model = None

    st.divider()
    st.markdown('<p class="mono muted" style="font-size:0.65rem">Predictions: ' +
                str(len(st.session_state.history)) + '</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
total = len(st.session_state.history)
anomalies = sum(1 for h in st.session_state.history if h.get('is_anomaly'))
faults    = sum(1 for h in st.session_state.history if h.get('is_fault'))
normals   = total - anomalies - faults

col_h1.metric("Total Predictions", total)
col_h2.metric("Anomalies Detected", anomalies, delta=None)
col_h3.metric("Faults Classified", faults)
col_h4.metric("Normal Readings", normals)

st.divider()

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎛️  MANUAL INPUT", "📂  BATCH CSV", "📋  HISTORY"])

# ════════════════════════════════════════════
#  TAB 1 — MANUAL INPUT
# ════════════════════════════════════════════
with tab1:
    st.markdown("### Sensor Readings")
    st.markdown('<p class="mono muted">Enter real-time sensor values below</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        ambient_temp_effect  = st.number_input("Ambient Temp Effect (°C)",   value=0.0,   step=0.1,  format="%.2f")
        reactor_temp         = st.number_input("Reactor Temp (°C)",           value=85.0,  step=0.1,  format="%.1f")
        reactor_press        = st.number_input("Reactor Pressure (bar)",      value=2.10,  step=0.01, format="%.2f")
        feed_flow_rate       = st.number_input("Feed Flow Rate (L/h)",        value=120.0, step=1.0,  format="%.1f")
        coolant_flow_rate    = st.number_input("Coolant Flow Rate (L/h)",     value=45.0,  step=0.5,  format="%.1f")

    with c2:
        agitator_speed_rpm   = st.number_input("Agitator Speed (RPM)",        value=250.0, step=5.0,  format="%.0f")
        reaction_rate        = st.number_input("Reaction Rate",                value=0.82,  step=0.01, format="%.3f")
        conversion_rate      = st.number_input("Conversion Rate",              value=0.85,  step=0.01, format="%.3f")
        selectivity          = st.number_input("Selectivity",                  value=0.90,  step=0.01, format="%.3f")
        yield_pct            = st.number_input("Yield (%)",                    value=80.0,  step=0.5,  format="%.1f")

    with c3:
        vibration_rms        = st.number_input("Vibration RMS (mm/s)",        value=1.20,  step=0.01, format="%.2f")
        motor_current        = st.number_input("Motor Current (A)",            value=20.0,  step=0.5,  format="%.1f")
        power_consumption_kw = st.number_input("Power Consumption (kW)",       value=10.0,  step=0.5,  format="%.1f")
        temp_setpoint        = st.number_input("Temp Setpoint (°C)",           value=85.0,  step=0.5,  format="%.1f")
        pressure_setpoint    = st.number_input("Pressure Setpoint (bar)",      value=2.10,  step=0.01, format="%.2f")

    st.markdown("")

    run_col, clear_col = st.columns([3, 1])
    with run_col:
        run_btn = st.button("▶  Run Prediction", type="primary", use_container_width=True)
    with clear_col:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # ── RUN PREDICTION ──
    if run_btn:
        if not selected_model:
            st.error("No model selected. Please load models from the sidebar first.")
        else:
            input_values = {
                'ambient_temp_effect':  ambient_temp_effect,
                'reactor_temp':         reactor_temp,
                'reactor_pressure':     reactor_press,
                'feed_flow_rate':       feed_flow_rate,
                'coolant_flow_rate':    coolant_flow_rate,
                'agitator_speed_rpm':   agitator_speed_rpm,
                'reaction_rate':        reaction_rate,
                'conversion_rate':      conversion_rate,
                'selectivity':          selectivity,
                'yield_pct':            yield_pct,
                'vibration_rms':        vibration_rms,
                'motor_current':        motor_current,
                'power_consumption_kw': power_consumption_kw,
                'temp_setpoint':        temp_setpoint,
                'pressure_setpoint':    pressure_setpoint,
            }

            # Build input df
            input_df = pd.DataFrame([input_values])

            # Only use columns the model was trained on
            model_obj = st.session_state.models[selected_model]
            scaler    = st.session_state.scaler

            # Determine feature columns from model if possible
            try:
                n_features = model_obj.n_features_in_
                feat_cols  = FEATURE_COLS[:n_features]
            except AttributeError:
                feat_cols = FEATURE_COLS

            # Fill missing cols with 0
            for col in feat_cols:
                if col not in input_df.columns:
                    input_df[col] = 0.0

            X = input_df[feat_cols].values

            # Scale
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                except Exception as e:
                    st.warning(f"Scaler mismatch ({e}). Running without scaling.")

            # Predict
            with st.spinner("Running inference..."):
                try:
                    prediction = model_obj.predict(X)[0]

                    # Anomaly Detection
                    if "Isolation" in selected_model:
                        score      = model_obj.decision_function(X)[0]
                        is_anomaly = prediction == -1
                        label      = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"
                        badge_cls  = "badge-anomaly" if is_anomaly else "badge-normal"
                        result_dict = {
                            'timestamp':   datetime.now().strftime("%H:%M:%S"),
                            'model':       selected_model,
                            'result':      label,
                            'score':       round(float(score), 4),
                            'confidence':  None,
                            'is_anomaly':  is_anomaly,
                            'is_fault':    False,
                            'inputs':      input_values,
                        }

                        # Display
                        st.divider()
                        res_c1, res_c2 = st.columns([1, 2])
                        with res_c1:
                            st.markdown(f"#### Result")
                            if is_anomaly:
                                st.error(f"## ⚠️ ANOMALY DETECTED")
                            else:
                                st.success(f"## ✅ NORMAL OPERATION")
                            st.markdown(f'<p class="mono">Decision Score: <strong>{score:.4f}</strong></p>', unsafe_allow_html=True)
                            st.markdown('<p class="mono muted" style="font-size:0.7rem">Negative score = anomaly<br>Positive score = normal</p>', unsafe_allow_html=True)

                        with res_c2:
                            st.markdown("#### Sensor Status")
                            status_rows = []
                            for k, v in input_values.items():
                                lo, hi, unit = NORMAL_RANGES.get(k, (None, None, ''))
                                if lo is not None:
                                    in_range = lo <= v <= hi
                                    status_rows.append({
                                        'Sensor': k.replace('_', ' ').title(),
                                        'Value': f"{v} {unit}",
                                        'Normal Range': f"{lo}–{hi} {unit}",
                                        'Status': "✅ OK" if in_range else "⚠️ OUT OF RANGE"
                                    })
                            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

                    # Fault Classification
                    else:
                        # Try to get probabilities
                        try:
                            proba = model_obj.predict_proba(X)[0]
                            classes = model_obj.classes_
                            prob_dict = {str(c): float(p) for c, p in zip(classes, proba)}
                            confidence = max(proba) * 100
                        except Exception:
                            prob_dict  = {}
                            confidence = None

                        fault_label = str(prediction)
                        is_normal   = fault_label in ['0', 'Normal Operation', 'normal']
                        result_dict = {
                            'timestamp':   datetime.now().strftime("%H:%M:%S"),
                            'model':       selected_model,
                            'result':      fault_label,
                            'score':       None,
                            'confidence':  round(confidence, 1) if confidence else None,
                            'is_anomaly':  False,
                            'is_fault':    not is_normal,
                            'inputs':      input_values,
                        }

                        # Display
                        st.divider()
                        res_c1, res_c2 = st.columns([1, 2])
                        with res_c1:
                            st.markdown("#### Fault Prediction")
                            if is_normal:
                                st.success(f"## ✅ Normal Operation")
                            else:
                                st.warning(f"## ⚠️ {fault_label}")

                            if confidence:
                                st.markdown(f'<p class="mono">Confidence: <strong>{confidence:.1f}%</strong></p>', unsafe_allow_html=True)
                                st.progress(confidence / 100)

                        with res_c2:
                            if prob_dict:
                                st.markdown("#### Probability Breakdown")
                                prob_df = pd.DataFrame(
                                    list(prob_dict.items()),
                                    columns=['Fault Type', 'Probability']
                                ).sort_values('Probability', ascending=False)
                                prob_df['Probability'] = (prob_df['Probability'] * 100).round(2)
                                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2f}%")
                                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("#### Sensor Status")
                                status_rows = []
                                for k, v in input_values.items():
                                    lo, hi, unit = NORMAL_RANGES.get(k, (None, None, ''))
                                    if lo is not None:
                                        status_rows.append({
                                            'Sensor': k.replace('_', ' ').title(),
                                            'Value': f"{v} {unit}",
                                            'Status': "✅ OK" if lo <= v <= hi else "⚠️ OUT OF RANGE"
                                        })
                                st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

                    st.session_state.history.append(result_dict)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

# ════════════════════════════════════════════
#  TAB 2 — BATCH CSV
# ════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Prediction from CSV")
    st.markdown('<p class="mono muted">Upload a CSV with sensor columns — predictions run on all rows at once</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f'<p class="mono">📄 Loaded <strong>{len(df_upload)}</strong> rows × <strong>{len(df_upload.columns)}</strong> columns</p>', unsafe_allow_html=True)
        st.dataframe(df_upload.head(5), use_container_width=True)

        if not selected_model:
            st.error("Load and select a model from the sidebar first.")
        else:
            if st.button("▶  Run Batch Prediction", type="primary"):
                model_obj = st.session_state.models[selected_model]
                scaler    = st.session_state.scaler

                try:
                    n_features = model_obj.n_features_in_
                    feat_cols  = FEATURE_COLS[:n_features]
                except AttributeError:
                    feat_cols = FEATURE_COLS

                # Fill missing cols
                for col in feat_cols:
                    if col not in df_upload.columns:
                        df_upload[col] = 0.0

                X_batch = df_upload[feat_cols].values

                if scaler is not None:
                    try:
                        X_batch = scaler.transform(X_batch)
                    except Exception as e:
                        st.warning(f"Scaler issue: {e}")

                with st.spinner(f"Running predictions on {len(df_upload)} rows..."):
                    preds = model_obj.predict(X_batch)
                    df_upload['prediction'] = preds

                    if "Isolation" in selected_model:
                        scores = model_obj.decision_function(X_batch)
                        df_upload['anomaly_score'] = scores
                        df_upload['result'] = df_upload['prediction'].map({1: 'NORMAL', -1: 'ANOMALY'})

                        anomaly_pct = (preds == -1).mean() * 100
                        c1b, c2b, c3b = st.columns(3)
                        c1b.metric("Total Rows", len(df_upload))
                        c2b.metric("Anomalies", int((preds == -1).sum()))
                        c3b.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")

                    else:
                        try:
                            proba = model_obj.predict_proba(X_batch)
                            df_upload['confidence'] = (proba.max(axis=1) * 100).round(2)
                        except Exception:
                            pass

                        fault_counts = pd.Series(preds).value_counts()
                        c1b, c2b = st.columns(2)
                        c1b.metric("Total Rows", len(df_upload))
                        c2b.metric("Unique Fault Types", len(fault_counts))

                st.markdown("#### Prediction Results")
                st.dataframe(df_upload, use_container_width=True)

                # Download
                csv_out = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️  Download Results CSV",
                    data=csv_out,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )

# ════════════════════════════════════════════
#  TAB 3 — HISTORY
# ════════════════════════════════════════════
with tab3:
    st.markdown("### Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet. Run some predictions in the Manual Input tab.")
    else:
        # Summary table
        history_df = pd.DataFrame([{
            'Time':       h['timestamp'],
            'Model':      h['model'].split(' (')[0],
            'Result':     h['result'],
            'Score':      h.get('score', '—'),
            'Confidence': f"{h['confidence']}%" if h.get('confidence') else '—',
        } for h in st.session_state.history])

        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Bar chart
        if len(st.session_state.history) >= 2:
            st.markdown("#### Result Distribution")
            results = [h['result'] for h in st.session_state.history]
            result_counts = pd.Series(results).value_counts()

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0f1319')
            ax.set_facecolor('#0f1319')
            colors = ['#00e5a0' if 'NORMAL' in r or 'Normal' in r else '#ff4d6d' if 'ANOMALY' in r else '#ffc230'
                      for r in result_counts.index]
            result_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='none', width=0.6)
            ax.set_xlabel('')
            ax.set_ylabel('Count', color='#4a5568', fontsize=9)
            ax.tick_params(colors='#4a5568', labelsize=8)
            ax.spines[:].set_color('#1e2530')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if st.button("🗑  Clear All History"):
            st.session_state.history = []
            st.rerun()