
import os
import joblib
import numpy as np

# =====================================
# FIXED PATHS
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "drug_likeness_rf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "drug_likeness_scaler.pkl")

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_from_descriptors(mw, logp, tpsa, hbd, hba, rtb):
    X = np.array([[mw, logp, tpsa, hbd, hba, rtb]])
    X_scaled = scaler.transform(X)

    prob = clf.predict_proba(X_scaled)[0][1]
    threshold = 0.45
    pred = 1 if prob >= threshold else 0

    return int(pred), float(prob)

