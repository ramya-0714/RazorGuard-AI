import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# RAZORGUARD AI - FINAL SYSTEM EVALUATION
# =========================================================

print("=" * 65)
print("RAZORGUARD AI - SYSTEM EVALUATION")
print("=" * 65)


# =========================================================
# 1. LOAD BATCH RESULTS
# =========================================================

RESULTS_PATH = (
    "results/razorguard_batch_results.csv"
)

df = pd.read_csv(
    RESULTS_PATH
)


print(
    f"\nTransactions evaluated: "
    f"{len(df)}"
)


# =========================================================
# 2. CREATE HIGH-RISK PREDICTION
# =========================================================

# RazorGuard considers HIGH risk as a fraud alert.

df["predicted_fraud"] = (
    df["risk_level"] == "HIGH"
).astype(int)


actual = df["fraud"]

predicted = df["predicted_fraud"]


# =========================================================
# 3. METRICS
# =========================================================

precision = precision_score(
    actual,
    predicted,
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)


# =========================================================
# 4. CONFUSION MATRIX
# =========================================================

tn, fp, fn, tp = confusion_matrix(
    actual,
    predicted
).ravel()


false_positive_rate = fp / (
    fp + tn
)


# =========================================================
# 5. HIGH-RISK DETECTION RATE
# =========================================================

total_fraud = (
    actual == 1
).sum()

detected_fraud = (
    (actual == 1)
    &
    (predicted == 1)
).sum()


detection_rate = (
    detected_fraud
    /
    total_fraud
)


# =========================================================
# 6. DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 65)
print("RAZORGUARD SYSTEM PERFORMANCE")
print("=" * 65)


print(
    f"\nPrecision: "
    f"{precision:.4f}"
)


print(
    f"Recall:    "
    f"{recall:.4f}"
)


print(
    f"F1 Score:  "
    f"{f1:.4f}"
)


print(
    f"FPR:       "
    f"{false_positive_rate:.4f}"
)


print(
    f"\nFraud detection rate: "
    f"{detection_rate:.4f}"
)


# =========================================================
# 7. CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)


print(
    f"\nTrue Negatives : {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives : {tp}"
)


# =========================================================
# 8. RISK DISTRIBUTION
# =========================================================

print("\n" + "=" * 65)
print("RISK DISTRIBUTION")
print("=" * 65)


print(
    f"\nLOW: "
    f"{sum(df['risk_level'] == 'LOW')}"
)

print(
    f"MEDIUM: "
    f"{sum(df['risk_level'] == 'MEDIUM')}"
)

print(
    f"HIGH: "
    f"{sum(df['risk_level'] == 'HIGH')}"
)


# =========================================================
# 9. FRAUD BY RISK LEVEL
# =========================================================

print("\n" + "=" * 65)
print("FRAUD BY RISK LEVEL")
print("=" * 65)


for level in [
    "LOW",
    "MEDIUM",
    "HIGH"
]:

    subset = df[
        df["risk_level"] == level
    ]

    fraud_count = (
        subset["fraud"] == 1
    ).sum()

    total_count = len(subset)

    if total_count > 0:

        fraud_rate = (
            fraud_count
            /
            total_count
        )

    else:

        fraud_rate = 0


    print(
        f"\n{level}:"
    )

    print(
        f"  Transactions: "
        f"{total_count}"
    )

    print(
        f"  Fraud: "
        f"{fraud_count}"
    )

    print(
        f"  Fraud rate: "
        f"{fraud_rate * 100:.2f}%"
    )


# =========================================================
# 10. SAVE EVALUATION
# =========================================================

evaluation = pd.DataFrame({

    "metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "False Positive Rate",
        "Fraud Detection Rate"
    ],

    "value": [
        precision,
        recall,
        f1,
        false_positive_rate,
        detection_rate
    ]
})


evaluation.to_csv(
    "results/razorguard_evaluation.csv",
    index=False
)


print(
    "\nEvaluation saved to:"
    " results/razorguard_evaluation.csv"
)


print(
    "\nSystem evaluation complete!"
)