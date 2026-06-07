import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib


def main():
    df = pd.read_csv("data.csv")

    print("Data shape:", df.shape)
    print("Suitability value counts:")
    print(df["Suitability"].value_counts())

    feature_cols = ["MolecularWeight", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds"]
    X = df[feature_cols].values
    y = df["Suitability"].astype(int).values

    # Slightly larger train set for learning
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,   # you can also try 0.15 or 0.10
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=14,          # try 8, 12, 16 if you see overfitting
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",  # important for your label distribution
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    # Check overfitting
    y_train_pred = clf.predict(X_train_s)
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Classification report:\n", classification_report(y_test, y_pred))

    joblib.dump(clf, "drug_likeness_rf.pkl")
    joblib.dump(scaler, "drug_likeness_scaler.pkl")
    print("Saved model to drug_likeness_rf.pkl")
    print("Saved scaler to drug_likeness_scaler.pkl")


if __name__ == "__main__":
    main()
