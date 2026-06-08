import pandas as pd        # data loading and manipulation (reading CSVs, DataFrames)
import numpy as np         # numerical operations, used under the hood by scikit-learn
import os                  # file path handling so the script finds CSVs in the right folder

from sklearn.model_selection import train_test_split   # splits data into training and validation sets
from sklearn.preprocessing import StandardScaler       # normalizes features to the same scale before modeling
from sklearn.decomposition import PCA                  # dimensionality reduction, optional preprocessing step
from sklearn.pipeline import Pipeline                  # chains scaler + PCA + model into one clean object
from sklearn.linear_model import LogisticRegression    # our baseline linear classifier
from sklearn.svm import SVC                            # Support Vector Machine with RBF kernel, our non-linear classifier
from sklearn.metrics import (                          # evaluation metrics to measure model performance
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# LOAD DATA
# Get the folder where this script lives so file paths always work correctly
base = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(base, "train.csv"))
test  = pd.read_csv(os.path.join(base, "test.csv"))

# PREPARE DATA
drop_cols = ["team_name", "country_code", "confederation"]

# X is the features, y is the target label, X_test is the test set without a target
X = train.drop(columns=drop_cols + ["winner"])
y = train["winner"]
X_test = test.drop(columns=drop_cols)

# Train/test split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")
print(f"Features: {X_train.shape[1]}")


# LEARNING ALGORITHMS
pipelines = {
    # Logistic Regression — linear classifier, used as our baseline
    "LR (no PCA)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, random_state=42))
    ]),
    # Logistic Regression with PCA preprocessing — tests if dimensionality reduction helps LR
    "LR + PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=0.95)),
        ("model",  LogisticRegression(max_iter=1000, random_state=42))
    ]),
    # SVM with RBF kernel — non-linear classifier, captures complex feature relationships
    "SVM (no PCA)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(kernel="rbf", random_state=42))
    ]),
    # SVM with PCA preprocessing — tests if dimensionality reduction helps SVM
    "SVM + PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=0.95)),
        ("model",  SVC(kernel="rbf", random_state=42))
    ]),
}

# TRAIN AND EVALUATE
print("\n" + "="*60)
print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("="*60)

results = {}

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec  = recall_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"{name:<20} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

print("="*60)

# REPORT FOR EACH MODEL
for name, pipeline in pipelines.items():
    y_pred = pipeline.predict(X_val)
    print(f"\n--- {name} ---")
    print(classification_report(y_val, y_pred, target_names=["Not Winner", "Winner"]))

# BEST MODEL
best = max(results, key=lambda x: results[x]["F1"])
print(f"\nBest model by F1: {best} ({results[best]['F1']:.4f})")