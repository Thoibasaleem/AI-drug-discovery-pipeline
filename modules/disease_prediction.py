import joblib
import numpy as np

model = joblib.load("models/model.pkl")
le_gene = joblib.load("models/gene_encoder.pkl")
le_disease = joblib.load("models/disease_encoder.pkl")


def predict_disease(gene, binding, logp, mw, drugscore):

    g = le_gene.transform([gene])[0]

    X = np.array([[g, binding, logp, mw, drugscore]])

    probs = model.predict_proba(X)[0]

    top5 = probs.argsort()[-5:][::-1]

    results = []

    for idx in top5:
        disease = le_disease.inverse_transform([idx])[0]
        results.append((disease, probs[idx]))

    return results