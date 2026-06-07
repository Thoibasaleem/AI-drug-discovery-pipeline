import numpy as np
import joblib

clf = joblib.load("models/drug_likeness_rf.pkl")
scaler = joblib.load("models/drug_likeness_scaler.pkl")


def predict_druglikeness(desc):

    x = np.array([[

        desc["MolWeight"],
        desc["LogP"],
        desc["TPSA"],
        desc["HBD"],
        desc["HBA"],
        desc["RotatableBonds"]

    ]])

    x_scaled = scaler.transform(x)

    prob = clf.predict_proba(x_scaled)[0][1]

    label = int(prob >= 0.4)

    return label, prob