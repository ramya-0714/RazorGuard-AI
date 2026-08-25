import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "models/razorguard_fraud_model.joblib"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("RAZORGUARD AI - RISK THRESHOLD ANALYSIS")
print("=" * 60)

print(f"\nDataset: {df.shape}")
print(f"Fraud rate: {df['fraud'].mean() * 100:.2f}%")


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

X = df.drop(columns=["fraud", "transaction_id"])
y = df["fraud"]


# ============================================================
# 3. RECREATE THE SAME TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. LOAD OUR TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("\nSaved RazorGuard model loaded successfully.")


# ============================================================
# 5. GET FRAUD PROBABILITIES
# ============================================================

fraud_probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. TEST DIFFERENT RISK THRESHOLDS
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


results = []


print("\n" + "=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

print(
    f"\n{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FPR':<12}"
)


for threshold in thresholds:

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    false_positive_rate = fp / (fp + tn)

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{false_positive_rate:<12.4f}"
    )


# ============================================================
# 7. FIND BEST F1 THRESHOLD
# ============================================================

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n" + "=" * 60)
print("BEST F1 THRESHOLD")
print("=" * 60)

print(
    f"\nThreshold:          "
    f"{best_row['threshold']:.2f}"
)

print(
    f"Precision:          "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall:             "
    f"{best_row['recall']:.4f}"
)

print(
    f"F1 Score:           "
    f"{best_row['f1']:.4f}"
)

print(
    f"False Positive Rate:"
    f"{best_row['false_positive_rate']:.4f}"
)


# ============================================================
# 8. SAVE THRESHOLD RESULTS
# ============================================================

results_df.to_csv(
    "results_threshold_analysis.csv",
    index=False
)

print(
    "\nThreshold analysis saved to:"
    " results_threshold_analysis.csv"
)

print("\nAnalysis complete!")