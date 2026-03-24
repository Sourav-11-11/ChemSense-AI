"""
Configuration and constants for ChemSense pipeline
Centralized settings to avoid magic numbers scattered across codebase
"""

# ─────────────────────────────────────────────
#  DATA PATHS
# ─────────────────────────────────────────────
RAW_DATA_PATH = "data/chemical_process_timeseries.csv"
CLEANED_DATA_PATH = "data/cleaned_data.csv"
ANOMALY_RESULTS_PATH = "data/anomaly_results.csv"

# ─────────────────────────────────────────────
#  MODEL PATHS
# ─────────────────────────────────────────────
MODELS_DIR = "models"
MODEL_PATHS = {
    "isolation_forest": "models/isolation_forest.pkl",
    "scaler_anomaly": "models/scaler_anomaly.pkl",
    "scaler_fault": "models/scaler_fault.pkl",
    "random_forest": "models/random_forest_model.pkl",
    "xgboost": "models/xgboost_model.pkl",
}

# Fallback paths for models (in case of naming inconsistencies)
MODEL_PATHS_FALLBACK = {
    "scaler_fault": ["models/scaler_fault.pkl", "models/scaler.pkl"],  # Fallback to scaler.pkl if scaler_fault missing
}

# ─────────────────────────────────────────────
#  FEATURE DEFINITIONS
# ─────────────────────────────────────────────
# Numeric sensor columns (used for modeling)
SENSOR_FEATURES = [
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

# Alias for backward compatibility
FEATURE_COLS = SENSOR_FEATURES

# Features to drop before fault prediction to avoid leakage
FAULT_PREDICTION_DROP_COLS = [
    'timestamp', 'fault_type', 'efficiency_loss_pct',
    'time_to_fault_min', 'anomaly_score', 'anomaly_raw',
    'pca1', 'pca2', 'reactor_id', 'operating_regime'
]

# ─────────────────────────────────────────────
#  NORMAL RANGES FOR SENSOR VALUES
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  FAULT LABELS
# ─────────────────────────────────────────────
FAULT_LABELS = [
    'Normal Operation', 'Sensor Drift', 'Valve Fault',
    'Heat Exchanger Fouling', 'Pump Cavitation',
    'Catalyst Deactivation', 'Pipe Blockage'
]

# ─────────────────────────────────────────────
#  ANOMALY DETECTION PARAMS
# ─────────────────────────────────────────────
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.05,      # % of samples as anomalies
    'n_estimators': 250,        # More trees = better
    'random_state': 42,
    'max_samples': 256,         # Better for larger datasets
    'n_jobs': -1,
}

# ─────────────────────────────────────────────
#  FAULT PREDICTION PARAMS
# ─────────────────────────────────────────────
RANDOM_FOREST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced',  # Handle class imbalance
}

XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'scale_pos_weight': 1,
}

# ─────────────────────────────────────────────
#  DATA CLEANING PARAMS
# ─────────────────────────────────────────────
NULL_THRESHOLD = 0.8         # Drop columns with >80% nulls
IQR_MULTIPLIER = 3.0         # 3*IQR for outlier clipping
RANDOM_STATE = 42

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING LAGS & ROLLING
# ─────────────────────────────────────────────
LAG_FEATURES = [1, 2, 3]                        # Hours to lag
ROLLING_WINDOWS = [3, 5, 12]                    # Hours for rolling stats
ROLLING_STATS = ['mean', 'std', 'min', 'max']  # Stats to compute
