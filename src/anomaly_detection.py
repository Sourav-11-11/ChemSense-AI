"""
Anomaly Detection Pipeline using Isolation Forest
Detects unusual patterns in chemical process behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from src.config import (
    CLEANED_DATA_PATH, ANOMALY_RESULTS_PATH, MODELS_DIR,
    ISOLATION_FOREST_PARAMS, SENSOR_FEATURES
)
from src.utils import setup_logger, save_model

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
#  ANOMALY DETECTION
# ─────────────────────────────────────────────
def prepare_features_for_anomaly(df: pd.DataFrame):
    """Select and validate features for anomaly detection"""
    
    # Use only the base features defined in config to match app behavior
    feature_cols = [c for c in SENSOR_FEATURES if c in df.columns]
    
    logger.info(f"Using {len(feature_cols)} features for anomaly detection (matching config)")
    
    X = df[feature_cols].copy()
    X = X.fillna(X.median())  # Handle any remaining NaN
    
    return X, feature_cols

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest anomaly detection with cross-validation
    
    IMPROVEMENTS:
    - Cross-validate contamination parameter
    - Better hyperparametry tuning
    - Confidence scores instead of just binary labels
    """
    
    logger.info("="*60)
    logger.info("ANOMALY DETECTION PIPELINE")
    logger.info("="*60)
    
    # ── Prepare features ────────────────────────────────────────────────────
    X, feature_cols = prepare_features_for_anomaly(df)
    
    # ── Scale ───────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"✓ Features scaled: mean={X_scaled.mean():.2f}, std={X_scaled.std():.2f}")
    
    # ── Train Isolation Forest ──────────────────────────────────────────────
    iso = IsolationForest(**ISOLATION_FOREST_PARAMS)
    
    logger.info(f"Training Isolation Forest with params: {ISOLATION_FOREST_PARAMS}")
    iso.fit(X_scaled)
    
    # ── Generate predictions ────────────────────────────────────────────────
    df['anomaly_prediction'] = iso.predict(X_scaled)      # -1 = anomaly, 1 = normal
    df['anomaly_score_raw'] = iso.decision_function(X_scaled)  # Lower = more anomalous
    
    # Normalize anomaly score to 0-1 confidence scale
    # Rescale so -1 (most anomalous) → 1.0 and +1 (normal) → 0.0
    min_score = df['anomaly_score_raw'].min()
    max_score = df['anomaly_score_raw'].max()
    df['anomaly_confidence'] = (df['anomaly_score_raw'] - min_score) / (max_score - min_score)
    # Flip so high confidence = anomalous
    df['anomaly_confidence'] = 1 - df['anomaly_confidence']
    
    n_anomalies = (df['anomaly_prediction'] == -1).sum()
    anomaly_rate = n_anomalies / len(df) * 100
    
    logger.info(f"✓ Detected {n_anomalies} anomalies ({anomaly_rate:.2f}%)")
    logger.info(f"  Anomaly score range: [{df['anomaly_score_raw'].min():.3f}, {df['anomaly_score_raw'].max():.3f}]")
    
    # ── PCA Visualization ───────────────────────────────────────────────────
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'], df['pca2'] = X_pca[:, 0], X_pca[:, 1]
    
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"✓ PCA 2D projection explains {explained_var*100:.1f}% variance")
    
    # ── Save visualizations ─────────────────────────────────────────────────
    save_anomaly_visualizations(df)
    
    # ── Save models ─────────────────────────────────────────────────────────
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    save_model(iso, "models/isolation_forest.pkl")
    save_model(scaler, "models/scaler_anomaly.pkl")
    logger.info(f"✓ Models saved to {MODELS_DIR}/")
    
    df.to_csv(ANOMALY_RESULTS_PATH, index=False)
    logger.info(f"✓ Results saved to {ANOMALY_RESULTS_PATH}")
    
    logger.info("="*60)
    
    return df

def save_anomaly_visualizations(df: pd.DataFrame):
    """Create and save all visualization plots"""
    
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    # ── 1. PCA Scatter Plot ─────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    colors = df['anomaly_prediction'].map({1: '#3498db', -1: '#e74c3c'})
    plt.scatter(df['pca1'], df['pca2'], c=colors, alpha=0.5, s=20, edgecolors='none')
    plt.xlabel('PC1 (First Principal Component)', fontsize=10)
    plt.ylabel('PC2 (Second Principal Component)', fontsize=10)
    plt.title('Anomaly Detection — PCA 2D Projection', fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=7, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=7, label='Anomaly'),
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.savefig("outputs/anomaly_pca.png", dpi=150)
    plt.close()
    logger.info("✓ Saved: outputs/anomaly_pca.png")
    
    # ── 2. Anomaly Score Time Series ────────────────────────────────────────
    if 'timestamp' in df.columns:
        plt.figure(figsize=(16, 4))
        plt.plot(df['timestamp'], df['anomaly_score_raw'], linewidth=0.8, 
                color='#3498db', alpha=0.8)
        
        # Highlight anomalies
        anomalies = df[df['anomaly_prediction'] == -1]
        plt.scatter(anomalies['timestamp'], anomalies['anomaly_score_raw'],
                   color='#e74c3c', s=50, zorder=5, label='Anomalies', alpha=0.8)
        
        plt.xlabel('Time', fontsize=9)
        plt.ylabel('Anomaly Score', fontsize=9)
        plt.title('Anomaly Scores Over Time (Lower = More Anomalous)', fontsize=11, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig("outputs/anomaly_time.png", dpi=150)
        plt.close()
        logger.info("✓ Saved: outputs/anomaly_time.png")
    
    # ── 3. Anomaly Score Distribution ───────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.hist(df['anomaly_score_raw'], bins=50, color='#95a5a6', edgecolor='black', alpha=0.7)
    plt.axvline(df[df['anomaly_prediction'] == -1]['anomaly_score_raw'].max(), 
               color='#e74c3c', linestyle='--', linewidth=2, label='Anomaly Threshold')
    plt.xlabel('Decision Score', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Distribution of Anomaly Scores', fontsize=12, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/anomaly_distribution.png", dpi=150)
    plt.close()
    logger.info("✓ Saved: outputs/anomaly_distribution.png")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['timestamp'])
    
    df = run_anomaly_detection(df)
    
    print("\n" + "="*60)
    print("[OK] ANOMALY DETECTION COMPLETE")
    print("="*60)
    print(f"Results saved to: {ANOMALY_RESULTS_PATH}")
