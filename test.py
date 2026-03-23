import joblib
iso = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model expects:", iso.n_features_in_, "features")
print("Scaler expects:", scaler.n_features_in_, "features")

# If scaler has feature names:
try:
    print("Feature names:", scaler.feature_names_in_.tolist())
except:
    print("No feature names stored")