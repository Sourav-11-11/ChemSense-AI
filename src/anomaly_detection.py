import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib

def run_anomaly_detection(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude target-like columns
    exclude = ['fault_type', 'efficiency_loss_pct', 'time_to_fault_min']
    feature_cols = [c for c in num_cols if c not in exclude]

    X = df[feature_cols].copy()

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Isolation Forest ─────────────────────────────────────────────────────
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    df['anomaly_score']  = iso.fit_predict(X_scaled)          # -1 = anomaly
    df['anomaly_raw']    = iso.decision_function(X_scaled)    # lower = more anomalous

    n_anomalies = (df['anomaly_score'] == -1).sum()
    print(f"Isolation Forest detected {n_anomalies} anomalies "
          f"({n_anomalies/len(df)*100:.2f}%)")

    # ── PCA visualisation ────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'], df['pca2'] = X_pca[:, 0], X_pca[:, 1]

    plt.figure(figsize=(10, 6))
    colors = df['anomaly_score'].map({1: 'steelblue', -1: 'red'})
    plt.scatter(df['pca1'], df['pca2'], c=colors, alpha=0.4, s=10)
    plt.title("Anomaly Detection — PCA Projection\n(Red = Anomaly)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("outputs/anomaly_pca.png", dpi=150)
    plt.close()
    print("Saved: outputs/anomaly_pca.png")

    # ── Anomaly score over time ───────────────────────────────────────────────
    if 'timestamp' in df.columns:
        plt.figure(figsize=(16, 4))
        plt.plot(df['timestamp'], df['anomaly_raw'], linewidth=0.5, color='teal')
        anomalies = df[df['anomaly_score'] == -1]
        plt.scatter(anomalies['timestamp'], anomalies['anomaly_raw'],
                    color='red', s=10, zorder=5, label='Anomaly')
        plt.title("Anomaly Score Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/anomaly_time.png", dpi=150)
        plt.close()
        print("Saved: outputs/anomaly_time.png")

    # ── Save model ───────────────────────────────────────────────────────────
    joblib.dump(iso,    "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Saved: models/isolation_forest.pkl, models/scaler.pkl")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=['timestamp'])
    df = run_anomaly_detection(df)
    df.to_csv("data/anomaly_results.csv", index=False)