import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data.csv")

feature_cols = [
    "MolecularWeight",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds"
]

X = df[feature_cols].values
y = df["Suitability"].astype(int).values

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("drug_likeness_rf.pkl")
scaler = joblib.load("drug_likeness_scaler.pkl")

X_scaled = scaler.transform(X)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

print("\n--- Drug-Likeness Model Evaluation ---")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_

plt.figure()
plt.bar(feature_cols, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()
