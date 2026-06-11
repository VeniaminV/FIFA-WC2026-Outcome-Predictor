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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# LOAD DATA
def load_data():
    # Get the folder where this script lives so file paths always work correctly
    base = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test  = pd.read_csv(os.path.join(base, "test.csv"))
    return train, test

# PREPARE DATA
def prepare_data(train, test):
    drop_cols = ["team_name", "country_code", "confederation"]

    # X is the features, y is the target label, X_test is the test set without a target
    X = train.drop(columns=drop_cols + ["winner"])
    y = train["winner"]
    X_test = test.drop(columns=drop_cols)

    # Train/test split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape} | Val size: {X_val.shape}")
    print(f"Features:   {X_train.shape[1]}")
    return X_train, X_val, y_train, y_val, X_test



# LEARNING ALGORITHMS
def build_pipelines():
    return {
        # Logistic Regression — linear classifier, used as our baseline
        "LR (no PCA)": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(max_iter=1000, random_state=42))
        ]),
        # Logistic Regression with PCA — tests if dimensionality reduction helps LR
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
        # SVM with PCA — tests if dimensionality reduction helps SVM
        "SVM + PCA": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=0.95)),
            ("model",  SVC(kernel="rbf", random_state=42))
        ]),
    }



# TRAIN MODEL 
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


# EVALUATE MODEL + CONFUSION MATRIX 
def evaluate_model(name, pipeline, X_train, y_train, X_val, y_val):
    train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_val)

    acc  = accuracy_score(y_val, y_pred)
    train_acc  = accuracy_score(y_train, train_pred)
    prec = precision_score(y_val, y_pred)
    rec  = recall_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred)

    print(f"\n--- {name} ---")
    print(classification_report(y_val, y_pred, target_names=["Not Winner", "Winner"]))

    # Plot and save confusion matrix
    cm   = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Winner", "Winner"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()

    # Save as PNG in the same folder as the script
    base     = os.path.dirname(os.path.abspath(__file__))
    filename = f"confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(base, filename))
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return {"Accuracy": acc, "Train Accuracy": train_acc, "Precision": prec, "Recall": rec, "F1": f1}


# PLOT TRAIN VS VALIDATION ACCURACY
def plot_train_val_accuracy(results):
    model_names = list(results.keys())
    train_scores = [results[name]["Train Accuracy"] for name in model_names]
    val_scores   = [results[name]["Accuracy"] for name in model_names]

    plt.figure()
    plt.plot(model_names, train_scores, marker='o', label="Train Accuracy")
    plt.plot(model_names, val_scores,   marker='o', linestyle='--', label="Val Accuracy")

    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("FIFA WC2026 — Train vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    base = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(base, "train_val_accuracy.png"))
    plt.show(block=False)
    plt.pause(5)
    plt.close()


# PRINT SUMMARY TABLE
def print_summary(results):
    print("\n" + "="*60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['Accuracy']:>10.4f} {metrics['Precision']:>10.4f} "
              f"{metrics['Recall']:>10.4f} {metrics['F1']:>10.4f}")
    print("="*60)

    best = max(results, key=lambda x: results[x]["F1"])
    print(f"\nBest model by F1: {best} ({results[best]['F1']:.4f})")


# INTERACTIVE TEAM PREDICTION
def predict_team(pipelines, test):
    print("\n" + "="*60)
    print("TEAM WIN PREDICTOR")
    print("="*60)

    drop_cols = ["team_name", "country_code", "confederation"]

    while True:
        team_name = input("\nEnter team name (or 'quit' to exit): ").strip()

        if team_name.lower() == "quit":
            break

        # search for team in test.csv 
        match = test[test["team_name"].str.lower() == team_name.lower()]

        if match.empty:
            print(f"  Team '{team_name}' not found. Available teams:")
            print("  " + ", ".join(test["team_name"].tolist()))
            continue

        team_row   = match.drop(columns=drop_cols)
        best_model = "SVM (no PCA)"  # use best performing model automatically
        pipeline   = pipelines[best_model]
        prediction = pipeline.predict(team_row)[0]

        print(f"\n  Team:       {match['team_name'].values[0]}")
        print(f"  Model:      {best_model}")
        print(f"  Prediction: {'Winner ' if prediction == 1 else 'Not Winner '}")


#MAIN
def main():
    train, test                              = load_data()
    X_train, X_val, y_train, y_val, X_test  = prepare_data(train, test)
    pipelines                                = build_pipelines()
    results                                  = {}

    for name, pipeline in pipelines.items():
        trained        = train_model(pipeline, X_train, y_train)
        results[name] = evaluate_model(name, trained, X_train, y_train, X_val, y_val)

    print_summary(results)
    plot_train_val_accuracy(results)
    predict_team(pipelines, test)

if __name__ == "__main__":
    main()