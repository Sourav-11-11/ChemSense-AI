"""
Fault Prediction Pipeline using Random Forest + XGBoost
Classifies type of fault occurring in the chemical process
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

from src.config import (
    ANOMALY_RESULTS_PATH, MODELS_DIR, FAULT_PREDICTION_DROP_COLS,
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, RANDOM_STATE
)
from src.utils import setup_logger, save_model

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
#  FEATURE PREPARATION
# ─────────────────────────────────────────────
def prepare_features_for_fault(df: pd.DataFrame):
    """
    Prepare features for fault classification
    
    KEY IMPROVEMENT: Avoid data leakage by NOT using anomaly scores
    Only use raw sensor features and engineered time-series features
    """
    
    # Get numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Remove columns that would cause data leakage
    feature_cols = [c for c in num_cols if c not in FAULT_PREDICTION_DROP_COLS]
    
    logger.info(f"Using {len(feature_cols)} features for fault prediction")
    logger.info(f"Dropped {len(FAULT_PREDICTION_DROP_COLS)} potential leakage columns")
    
    return feature_cols

def run_fault_prediction(df: pd.DataFrame) -> dict:
    """
    Train fault prediction models with cross-validation and comprehensive metrics
    
    IMPROVEMENTS:
    - Cross-validation for robust evaluation
    - Class imbalance handling via class weights
    - Comprehensive metrics (precision, recall, F1)
    - Multiple models for comparison
    - Feature importance analysis
    """
    
    logger.info("="*60)
    logger.info("FAULT PREDICTION PIPELINE")
    logger.info("="*60)
    
    if 'fault_type' not in df.columns:
        logger.warning("No 'fault_type' column found — skipping fault prediction")
        return {}
    
    # ── Prepare features ────────────────────────────────────────────────────
    feature_cols = prepare_features_for_fault(df)
    
    # Handle any missing features gracefully
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    # Encode categorical columns if any
    for col in X.select_dtypes(include='object').columns.tolist():
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    y = df['fault_type'].copy()
    
    # Check class distribution
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    # Compute class weights for imbalance handling
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weight_dict}")
    
    # ── Scale features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"✓ Features scaled: mean={X_scaled.mean():.2f}, std={X_scaled.std():.2f}")
    
    # ── Train / Test Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ── Random Forest ───────────────────────────────────────────────────────
    logger.info("\n" + "─"*40)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("─"*40)
    
    rf = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    rf.fit(X_train, y_train)
    
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_preds),
        'precision': precision_score(y_test, rf_preds, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_preds, average='weighted', zero_division=0),
        'f1': f1_score(y_test, rf_preds, average='weighted', zero_division=0),
    }
    
    logger.info(f"Random Forest Metrics: {rf_metrics}")
    logger.info(f"\nDetailed Report:\n{classification_report(y_test, rf_preds, zero_division=0)}")
    
    # Cross-validation
    cv_results_rf = cross_validate(rf, X_train, y_train, cv=5, scoring='f1_weighted')
    logger.info(f"CV F1 Scores: {cv_results_rf['test_score']}")
    logger.info(f"Mean CV F1: {cv_results_rf['test_score'].mean():.3f} (+/- {cv_results_rf['test_score'].std():.3f})")
    
    # ── XGBoost ─────────────────────────────────────────────────────────────
    logger.info("\n" + "─"*40)
    logger.info("TRAINING XGBOOST")
    logger.info("─"*40)
    
    xgb = XGBClassifier(**XGBOOST_PARAMS)
    xgb.fit(X_train, y_train, eval_metric='mlogloss')
    
    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    
    xgb_metrics = {
        'accuracy': accuracy_score(y_test, xgb_preds),
        'precision': precision_score(y_test, xgb_preds, average='weighted', zero_division=0),
        'recall': recall_score(y_test, xgb_preds, average='weighted', zero_division=0),
        'f1': f1_score(y_test, xgb_preds, average='weighted', zero_division=0),
    }
    
    logger.info(f"XGBoost Metrics: {xgb_metrics}")
    logger.info(f"\nDetailed Report:\n{classification_report(y_test, xgb_preds, zero_division=0)}")
    
    # Cross-validation
    cv_results_xgb = cross_validate(xgb, X_train, y_train, cv=5, scoring='f1_weighted')
    logger.info(f"CV F1 Scores: {cv_results_xgb['test_score']}")
    logger.info(f"Mean CV F1: {cv_results_xgb['test_score'].mean():.3f} (+/- {cv_results_xgb['test_score'].std():.3f})")
    
    # ── Model Comparison ────────────────────────────────────────────────────
    logger.info("\n" + "="*40)
    logger.info("MODEL COMPARISON")
    logger.info("="*40)
    logger.info(f"Random Forest - Test F1: {rf_metrics['f1']:.3f}")
    logger.info(f"XGBoost      - Test F1: {xgb_metrics['f1']:.3f}")
    best_model = 'XGBoost' if xgb_metrics['f1'] > rf_metrics['f1'] else 'Random Forest'
    logger.info(f"Winner: {best_model}")
    
    # ── Visualizations ──────────────────────────────────────────────────────
    save_fault_visualizations(y_test, rf_preds, xgb_preds, rf, feature_cols)
    
    # ── Save models ─────────────────────────────────────────────────────────
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    save_model(rf, f"{MODELS_DIR}/random_forest_model.pkl")
    save_model(xgb, f"{MODELS_DIR}/xgboost_model.pkl")
    save_model(scaler, f"{MODELS_DIR}/scaler_fault.pkl")
    logger.info(f"✓ Models saved to {MODELS_DIR}/")
    
    logger.info("="*60)
    
    return {
        'random_forest': rf,
        'xgboost': xgb,
        'scaler': scaler,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics,
        'feature_cols': feature_cols,
    }

def save_fault_visualizations(y_test, rf_preds, xgb_preds, rf_model, feature_cols):
    """Generate and save visualization plots"""
    
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    # ── 1. Confusion Matrix (XGBoost) ────────────────────────────────────────
    cm = confusion_matrix(y_test, xgb_preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues', ax=None)
    plt.title("Confusion Matrix — XGBoost Fault Classification", fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("✓ Saved: outputs/confusion_matrix.png")
    
    # ── 2. Feature Importance ────────────────────────────────────────────────
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(15)
    
    plt.figure(figsize=(8, 6))
    top_features.sort_values().plot(kind='barh', color='#3498db', edgecolor='black')
    plt.xlabel('Importance Score', fontsize=10)
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    logger.info("✓ Saved: outputs/feature_importance.png")
    
    # ── 3. Model Comparison (Accuracy) ──────────────────────────────────────
    from sklearn.metrics import accuracy_score
    
    rf_acc = float(accuracy_score(y_test, rf_preds))
    xgb_acc = float(accuracy_score(y_test, xgb_preds))
    
    models = ['Random Forest', 'XGBoost']
    accuracies = [rf_acc, xgb_acc]
    colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
    
    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.8)
    plt.ylabel('Test Accuracy', fontsize=10)
    plt.title('Model Performance Comparison', fontsize=12, fontweight='bold')
    plt.ylim([0, 1])
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        plt.text(i, float(acc) + 0.02, f'{float(acc):.2%}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150)
    plt.close()
    logger.info("✓ Saved: outputs/model_comparison.png")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading anomaly detection results...")
    df = pd.read_csv(ANOMALY_RESULTS_PATH, parse_dates=['timestamp'])
    
    results = run_fault_prediction(df)
    
    print("\n" + "="*60)
    print("[OK] FAULT PREDICTION COMPLETE")
    print("="*60)
