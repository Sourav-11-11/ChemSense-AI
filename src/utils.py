"""
Utility functions for data processing, model management, and validation
Shared across all pipeline modules to reduce code duplication
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import os

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logger(name: str, log_level=logging.INFO) -> logging.Logger:
    """Configure logger for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# ─────────────────────────────────────────────
#  MODEL PERSISTENCE
# ─────────────────────────────────────────────
def save_model(model: Any, filepath: str, overwrite: bool = True) -> bool:
    """Save model with error handling"""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        return True
    except Exception as e:
        logging.error(f"Failed to save model to {filepath}: {e}")
        return False

def load_model(filepath: str) -> Any:
    """Load model with validation and fallback paths"""
    from . import config
    
    try:
        # Check if file exists
        if os.path.exists(filepath):
            return joblib.load(filepath)
        
        # Check for fallback paths
        if hasattr(config, 'MODEL_PATHS_FALLBACK'):
            for key, paths in config.MODEL_PATHS_FALLBACK.items():
                if filepath in paths:
                    # Try each fallback path
                    for fallback_path in paths:
                        if os.path.exists(fallback_path):
                            logging.warning(f"Primary file {filepath} not found, using fallback: {fallback_path}")
                            return joblib.load(fallback_path)
        
        raise FileNotFoundError(f"Model file not found: {filepath}")
    except Exception as e:
        logging.error(f"Failed to load model from {filepath}: {e}")
        return None

# ─────────────────────────────────────────────
#  DATA VALIDATION
# ─────────────────────────────────────────────
def validate_input_features(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Check if DataFrame has required columns"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing columns: {missing}")
        return False
    return True

def validate_feature_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple]) -> pd.DataFrame:
    """
    Flag out-of-bounds sensor values
    Returns: boolean DataFrame marking violations
    """
    violations = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col, (low, high, _) in bounds.items():
        if col in df.columns:
            violations[col] = (df[col] < low) | (df[col] > high)
    return violations

def get_out_of_bounds_sensors(df: pd.DataFrame, bounds: Dict[str, Tuple]) -> List[str]:
    """Get list of sensors with out-of-bounds values"""
    out_of_bounds = []
    for col, (low, high, unit) in bounds.items():
        if col in df.columns:
            if ((df[col] < low) | (df[col] > high)).any():
                out_of_bounds.append(col)
    return out_of_bounds

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def create_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for time series (e.g., t-1, t-2, t-3)
    
    Args:
        df: DataFrame with timestamp and feature columns
        cols: Columns to lag
        lags: List of lag values (e.g., [1, 2, 3])
    
    Returns:
        DataFrame with new lag columns added
    """
    df = df.copy()
    
    # Ensure sorted by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    for col in cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def create_rolling_features(df: pd.DataFrame, cols: List[str], 
                           windows: List[int], stats: List[str] = None) -> pd.DataFrame:
    """
    Create rolling statistics (mean, std, min, max)
    
    Args:
        df: DataFrame with features
        cols: Columns to create rolling features for
        windows: Window sizes (e.g., [3, 5, 12])
        stats: Statistics to compute (e.g., ['mean', 'std', 'min', 'max'])
    
    Returns:
        DataFrame with rolling features added
    """
    if stats is None:
        stats = ['mean', 'std', 'min', 'max']
    
    df = df.copy()
    
    # Ensure sorted by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    for col in cols:
        if col in df.columns:
            for window in windows:
                for stat in stats:
                    new_col = f'{col}_rolling_{stat}_{window}'
                    if stat == 'mean':
                        df[new_col] = df[col].rolling(window, min_periods=1).mean()
                    elif stat == 'std':
                        df[new_col] = df[col].rolling(window, min_periods=1).std()
                    elif stat == 'min':
                        df[new_col] = df[col].rolling(window, min_periods=1).min()
                    elif stat == 'max':
                        df[new_col] = df[col].rolling(window, min_periods=1).max()
    
    return df

def create_difference_features(df: pd.DataFrame, cols: List[str], periods: List[int] = None) -> pd.DataFrame:
    """
    Create rate-of-change features (differences between consecutive periods)
    
    Args:
        df: DataFrame with features
        cols: Columns to compute differences for
        periods: Period differences (e.g., [1, 2] = t-t-1, t-t-2)
    
    Returns:
        DataFrame with difference features added
    """
    if periods is None:
        periods = [1]
    
    df = df.copy()
    
    # Ensure sorted by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    for col in cols:
        if col in df.columns:
            for period in periods:
                df[f'{col}_diff_{period}'] = df[col].diff(period)
    
    return df

# ─────────────────────────────────────────────
#  MODEL EVALUATION
# ─────────────────────────────────────────────
def get_model_info(model: Any) -> Dict:
    """Extract model metadata"""
    info = {
        'type': type(model).__name__,
        'n_features': getattr(model, 'n_features_in_', None),
        'classes': getattr(model, 'classes_', None),
    }
    if hasattr(model, 'feature_importances_'):
        info['has_feature_importance'] = True
    return info

def log_classification_results(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, logger: logging.Logger) -> Dict:
    """Compute and log key classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
    }
    
    # Handle multiclass
    try:
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1'] = f1_score(y_true, y_pred, average='weighted')
    except:
        pass
    
    logger.info(f"{model_name}: {results}")
    return results

# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_features_for_model(df: pd.DataFrame, feature_cols: List[str], 
                               scaler=None, fit_scaler: bool = False) -> Tuple[np.ndarray, Any]:
    """
    Prepare features for model - handle missing values and scaling
    
    Returns:
        (scaled_features, scaler_object)
    """
    from sklearn.preprocessing import StandardScaler
    
    X = df[feature_cols].copy()
    
    # Fill NaN with median
    X = X.fillna(X.median())
    
    # Scale
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values
    
    return X, scaler

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     stratify: bool = True, random_state: int = 42) -> Tuple:
    """Stratified train-test split with logging"""
    from sklearn.model_selection import train_test_split
    
    y_for_stratify = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y_for_stratify
    )
    
    logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                         cv: int = 5, scoring: str = 'accuracy') -> Dict:
    """
    Run k-fold cross-validation
    
    Returns:
        Dict with cv scores and mean/std
    """
    from sklearn.model_selection import cross_validate
    
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True
    )
    
    return {
        'test_scores': results['test_score'],
        'train_scores': results['train_score'],
        'test_mean': results['test_score'].mean(),
        'test_std': results['test_score'].std(),
    }
