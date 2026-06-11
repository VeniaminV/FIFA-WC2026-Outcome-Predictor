# FIFA World Cup 2026 Outcome Predictor

CS 445: Machine Learning — Final Project
Portland State University, Spring 2026
**Veniamin Velikoretskikh & Pete Treemeth**

---

## Overview

This project uses machine learning to predict FIFA World Cup 2026 match outcomes from team-level soccer statistics. Two classifiers are compared — Logistic Regression (linear baseline) and Support Vector Machine with an RBF kernel (nonlinear) — each tested with and without Principal Component Analysis as a preprocessing step.

---

## Dataset

Download from Kaggle:
https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-prediction-system

Place the following files in the project root folder (same folder as the `.py` file):
- `train.csv` — 1000 labeled examples used for training and validation
- `test.csv` — 250 examples used for the interactive team predictor

> The CSV files are not included in this repository. Download them separately from the link above.

---

## Requirements

Install dependencies with:

```bash
pip install scikit-learn pandas matplotlib numpy
```

---

## How to Run

```bash
python FIFA_WC2026_outcome_prediction.py
```

The script will:
1. Load and preprocess the dataset
2. Train all four model configurations
3. Print classification reports and a summary table
4. Display and save confusion matrices as PNG files
5. Display and save a train vs validation accuracy plot
6. Launch an interactive team predictor

---

## Models

| Model | Description |
|---|---|
| LR (no PCA) | Logistic Regression — linear baseline |
| LR + PCA | Logistic Regression with dimensionality reduction |
| SVM (no PCA) | SVM with RBF kernel — nonlinear classifier |
| SVM + PCA | SVM with RBF kernel + dimensionality reduction |

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| LR (no PCA) | 0.6500 | 0.6344 | 0.6211 | 0.6277 |
| LR + PCA | 0.6200 | 0.6022 | 0.5895 | 0.5957 |
| SVM (no PCA) | **0.6600** | **0.6484** | 0.6211 | **0.6344** |
| SVM + PCA | 0.6300 | 0.6154 | 0.5895 | 0.6022 |

Best model: **SVM with RBF kernel (no PCA)**

---

## Interactive Predictor

After training completes, the script prompts you to enter a team name and returns a win prediction using the best model. Type `quit` to exit.

```
Enter team name (or 'quit' to exit): Argentina

  Team:       Argentina
  Model:      SVM (no PCA)
  Prediction: Winner ✓
```

---

## Output Files

The following files are saved to the project folder after running:

- `confusion_matrix_LR_no_PCA_.png`
- `confusion_matrix_LR_+_PCA.png`
- `confusion_matrix_SVM_no_PCA_.png`
- `confusion_matrix_SVM_+_PCA.png`
- `train_val_accuracy.png`

---

## Project Structure

```
FIFA-WC2026-Outcome-Predictor/
├── FIFA_WC2026_outcome_prediction.py
├── train.csv           # download from Kaggle (not in repo)
├── test.csv            # download from Kaggle (not in repo)
├── README.md
└── .gitignore
```

---

## References

- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*. https://link.springer.com/article/10.1007/BF00994018
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*. https://doi.org/10.48550/arXiv.1201.0490
- Rambe, R. F. (2026). FIFA World Cup 2026 Prediction System. Kaggle. https://www.kaggle.com/dsv/16165118
