import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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
