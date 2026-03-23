import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    print("=== Basic Stats ===")
    print(df.describe())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── Distribution plots ───────────────────────────────────────────────────
    n = len(num_cols)
    cols_per_row = 4
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/distributions.png", dpi=150)
    plt.close()
    print("Saved: outputs/distributions.png")

    # ── Correlation heatmap ──────────────────────────────────────────────────
    plt.figure(figsize=(14, 10))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.3)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: outputs/correlation_heatmap.png")

    # ── Time series plots for key sensors ───────────────────────────────────
    key_sensors = ['reactor_temp', 'reactor_pressure', 'feed_flow',
                   'coolant_flow_rate', 'reaction_rate']
    available = [c for c in key_sensors if c in df.columns]

    if 'timestamp' in df.columns and available:
        fig, axes = plt.subplots(len(available), 1, figsize=(16, len(available) * 3))
        if len(available) == 1:
            axes = [axes]
        for ax, col in zip(axes, available):
            ax.plot(df['timestamp'], df[col], linewidth=0.6, color='teal')
            ax.set_ylabel(col, fontsize=9)
            ax.grid(alpha=0.3)
        plt.suptitle("Key Sensor Time Series", fontsize=13)
        plt.tight_layout()
        plt.savefig("outputs/time_series.png", dpi=150)
        plt.close()
        print("Saved: outputs/time_series.png")

    # ── Fault distribution ───────────────────────────────────────────────────
    if 'fault_type' in df.columns:
        plt.figure(figsize=(7, 4))
        df['fault_type'].value_counts().plot(kind='bar', color='salmon', edgecolor='white')
        plt.title("Fault Type Distribution")
        plt.xlabel("Fault Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("outputs/fault_distribution.png", dpi=150)
        plt.close()
        print("Saved: outputs/fault_distribution.png")

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=['timestamp'])
    run_eda(df)