import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    print("=== Initial Data Info ===")
    print(df.shape)
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # ── 1. Parse timestamps ──────────────────────────────────────────────────
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # ── 2. Drop columns that are >80% null ──────────────────────────────────
    threshold = 0.8
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    if cols_to_drop:
        print(f"\nDropping columns with >{int(threshold*100)}% nulls: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # ── 3. Separate categorical vs numeric columns ───────────────────────────
    cat_cols  = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude timestamp from imputation
    if 'timestamp' in num_cols:
        num_cols.remove('timestamp')

    # ── 4. Fill categorical nulls with mode ─────────────────────────────────
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ── 5. Fast median imputation for numeric columns ──────────────────────────
    if num_cols:
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)

    # ── 6. Remove duplicate rows ────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"\nRemoved {before - len(df)} duplicate rows")

    # ── 7. Clip extreme outliers using IQR (per numeric column) ─────────────
    # Exclude categorical/target columns from being squashed by outlier clipping
    exclude_from_clipping = ['fault_type', 'anomaly_score', 'reactor_id']
    clip_cols = [c for c in num_cols if c not in exclude_from_clipping]

    for col in clip_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
        df[col] = df[col].clip(lower, upper)

    print("\n=== After Cleaning ===")
    print(df.shape)
    print(df.isnull().sum().sum(), "total nulls remaining")

    return df


if __name__ == "__main__":
    df = load_data("data/chemical_process_timeseries.csv")
    df_clean = clean_data(df)
    df_clean.to_csv("data/cleaned_data.csv", index=False)
    print("Saved: data/cleaned_data.csv")