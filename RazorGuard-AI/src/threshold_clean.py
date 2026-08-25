import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================================================
# RAZORGUARD AI - CLEAN MODEL THRESHOLD ANALYSIS
# =========================================================

print("=" * 65)
print("RAZORGUARD AI - CLEAN XGBOOST THRESHOLD ANALYSIS")
print("=" * 65)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/transactions_v2.csv"
)

X = df.drop(
    columns=[
        "fraud",
        "transaction_id",
        "customer_id"
    ]
)

y = df["fraud"]


# Convert payment method
X = pd.get_dummies(
    X,
    columns=["payment_method"],
    dtype=int
)


# =========================================================
# 2. SAME TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 3. LOAD CLEAN MODEL
# =========================================================

model = XGBClassifier()

model.load_model(
    "models/razorguard_clean_model.json"
)

print("\nClean XGBoost model loaded successfully.")


# =========================================================
# 4. GET FRAUD PROBABILITIES
# =========================================================

fraud_probability = model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# 5. TEST DIFFERENT THRESHOLDS
# =========================================================

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


print("\n" + "=" * 65)
print("THRESHOLD COMPARISON")
print("=" * 65)

print(
    f"\n{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FPR':<12}"
)


results = []


for threshold in thresholds:

    predictions = (
        fraud_probability >= threshold
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

    fpr = fp / (
        fp + tn
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fpr:<12.4f}"
    )


# =========================================================
# 6. BEST F1 THRESHOLD
# =========================================================

results_df = pd.DataFrame(results)

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n" + "=" * 65)
print("BEST F1 THRESHOLD")
print("=" * 65)

print(
    f"\nThreshold:  "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"Precision:  "
    f"{best_f1['precision']:.4f}"
)

print(
    f"Recall:     "
    f"{best_f1['recall']:.4f}"
)

print(
    f"F1 Score:   "
    f"{best_f1['f1']:.4f}"
)

print(
    f"FPR:        "
    f"{best_f1['fpr']:.4f}"
)


# =========================================================
# 7. SAVE RESULTS
# =========================================================

results_df.to_csv(
    "results_clean_thresholds.csv",
    index=False
)

print(
    "\nResults saved to:"
    " results_clean_thresholds.csv"
)

print("\nThreshold analysis complete!")