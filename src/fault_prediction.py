import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier
import joblib

def run_fault_prediction(df):
    if 'fault_type' not in df.columns:
        print("No 'fault_type' column found — skipping fault prediction.")
        return

    # ── Features ─────────────────────────────────────────────────────────────
    drop_cols = ['timestamp', 'fault_type', 'efficiency_loss_pct',
                 'time_to_fault_min', 'anomaly_score', 'anomaly_raw',
                 'pca1', 'pca2', 'reactor_id', 'operating_regime']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Encode any remaining categoricals
    df_model = df[feature_cols + ['fault_type']].copy()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

    X = df_model[feature_cols]
    y = df_model['fault_type']

    print(f"Target distribution:\n{y.value_counts()}\n")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_preds = rf.predict(X_test_s)
    print("=== Random Forest ===")
    print(classification_report(y_test, rf_preds))

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                        eval_metric='mlogloss', use_label_encoder=False)
    xgb.fit(X_train_s, y_train)
    xgb_preds = xgb.predict(X_test_s)
    print("=== XGBoost ===")
    print(classification_report(y_test, xgb_preds))

    # ── Confusion matrix (XGBoost) ───────────────────────────────────────────
    cm = confusion_matrix(y_test, xgb_preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix — XGBoost")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: outputs/confusion_matrix.png")

    # ── Feature importance ───────────────────────────────────────────────────
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top20 = importances.nlargest(20)
    plt.figure(figsize=(8, 7))
    top20.sort_values().plot(kind='barh', color='steelblue')
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("Saved: outputs/feature_importance.png")

    # ── Save models ──────────────────────────────────────────────────────────
    joblib.dump(rf,  "models/random_forest_model.pkl")
    joblib.dump(xgb, "models/xgboost_model.pkl")
    print("Saved: models/random_forest_model.pkl, models/xgboost_model.pkl")


if __name__ == "__main__":
    df = pd.read_csv("data/anomaly_results.csv", parse_dates=['timestamp'])
    run_fault_prediction(df)