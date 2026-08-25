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
# RAZORGUARD AI - HELD-OUT EVALUATION
# =========================================================

print("=" * 65)
print("RAZORGUARD AI - HELD-OUT EVALUATION")
print("=" * 65)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/transactions_v2.csv"
)

print(
    f"\nTotal transactions: {len(df)}"
)

print(
    f"Total fraud: "
    f"{df['fraud'].sum()}"
)


# =========================================================
# 2. CREATE HELD-OUT TEST SET
# =========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["fraud"]
)

print(
    f"\nTraining portion: {len(train_df)}"
)

print(
    f"Held-out test portion: {len(test_df)}"
)

print(
    f"Held-out fraud: "
    f"{test_df['fraud'].sum()}"
)


# =========================================================
# 3. PREPARE TEST FEATURES
# =========================================================

X_test = test_df.drop(
    columns=[
        "fraud",
        "transaction_id",
        "customer_id"
    ],
    errors="ignore"
)

y_test = test_df["fraud"]


X_test = pd.get_dummies(
    X_test,
    columns=["payment_method"],
    dtype=int
)


# =========================================================
# 4. LOAD TRAINED CLEAN MODEL
# =========================================================

model = XGBClassifier()

model.load_model(
    "models/razorguard_clean_model.json"
)

print(
    "\nClean XGBoost model loaded successfully!"
)


# =========================================================
# 5. ALIGN FEATURES
# =========================================================

expected_features = (
    model.get_booster().feature_names
)

for feature in expected_features:

    if feature not in X_test.columns:

        X_test[feature] = 0


X_test = X_test[
    expected_features
]


# =========================================================
# 6. ML SCORE
# =========================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

ml_scores = probabilities * 100


# =========================================================
# 7. BEHAVIOURAL SCORING
# =========================================================

def calculate_signal_score(row):

    score = 0
    reasons = []


    # New device
    if row["is_new_device"] == 1:

        score += 15
        reasons.append("New device")


    # Location change
    if row["location_change"] == 1:

        score += 15
        reasons.append("Location change")


    # Velocity
    velocity = row[
        "transactions_last_10min"
    ]

    if velocity >= 5:

        score += 20
        reasons.append("High velocity")

    elif velocity >= 3:

        score += 8
        reasons.append("Elevated velocity")


    # Failed attempts
    failures = row[
        "failed_attempts"
    ]

    if failures >= 3:

        score += 15
        reasons.append("Multiple failed attempts")

    elif failures >= 2:

        score += 8
        reasons.append("Repeated failed attempts")


    # Amount deviation
    deviation = row[
        "amount_deviation"
    ]

    if deviation >= 7:

        score += 20
        reasons.append("Extreme amount deviation")

    elif deviation >= 4:

        score += 15
        reasons.append("High amount deviation")

    elif deviation >= 2:

        score += 7
        reasons.append("Above-normal amount")


    # Distance
    distance = row[
        "distance_from_last_transaction"
    ]

    if distance >= 1500:

        score += 15
        reasons.append("Very large distance")

    elif distance >= 700:

        score += 10
        reasons.append("Large distance")


    # New device + location
    if (
        row["is_new_device"] == 1
        and row["location_change"] == 1
    ):

        score += 10
        reasons.append(
            "New device + location change"
        )


    # New device + amount
    if (
        row["is_new_device"] == 1
        and row["amount_deviation"] >= 4
    ):

        score += 10
        reasons.append(
            "New device + unusual amount"
        )


    # Location + distance
    if (
        row["location_change"] == 1
        and row[
            "distance_from_last_transaction"
        ] >= 700
    ):

        score += 10
        reasons.append(
            "Location change + large distance"
        )


    return min(score, 100), reasons


# =========================================================
# 8. APPLY BEHAVIOURAL SCORE
# =========================================================

signal_scores = []
reason_lists = []


for _, row in test_df.iterrows():

    score, reasons = calculate_signal_score(
        row
    )

    signal_scores.append(score)
    reason_lists.append(reasons)


# =========================================================
# 9. HYBRID RISK SCORE
# =========================================================

risk_scores = (
    (ml_scores * 0.40)
    +
    (
        pd.Series(
            signal_scores,
            index=test_df.index
        )
        * 0.60
    )
)

risk_scores = risk_scores.clip(
    upper=100
).round(2)


# =========================================================
# 10. RISK LEVEL
# =========================================================

def get_risk_level(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


risk_levels = [
    get_risk_level(score)
    for score in risk_scores
]


# =========================================================
# 11. CREATE RESULTS
# =========================================================

results = test_df.copy()

results["ml_score"] = (
    ml_scores.round(2)
)

results["behavioural_score"] = (
    signal_scores
)

results["risk_score"] = (
    risk_scores
)

results["risk_level"] = (
    risk_levels
)


# =========================================================
# 12. BINARY HIGH-RISK EVALUATION
# =========================================================

predicted_fraud = (
    results["risk_level"] == "HIGH"
).astype(int)

actual_fraud = results["fraud"]


precision = precision_score(
    actual_fraud,
    predicted_fraud,
    zero_division=0
)

recall = recall_score(
    actual_fraud,
    predicted_fraud,
    zero_division=0
)

f1 = f1_score(
    actual_fraud,
    predicted_fraud,
    zero_division=0
)


tn, fp, fn, tp = confusion_matrix(
    actual_fraud,
    predicted_fraud
).ravel()


fpr = fp / (
    fp + tn
)


# =========================================================
# 13. DISPLAY PERFORMANCE
# =========================================================

print("\n" + "=" * 65)
print("HELD-OUT RAZORGUARD PERFORMANCE")
print("=" * 65)


print(
    f"\nPrecision: "
    f"{precision:.4f}"
)

print(
    f"Recall: "
    f"{recall:.4f}"
)

print(
    f"F1 Score: "
    f"{f1:.4f}"
)

print(
    f"False Positive Rate: "
    f"{fpr:.4f}"
)


# =========================================================
# 14. CONFUSION MATRIX
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
# 15. RISK DISTRIBUTION
# =========================================================

print("\n" + "=" * 65)
print("HELD-OUT RISK DISTRIBUTION")
print("=" * 65)


for level in [
    "LOW",
    "MEDIUM",
    "HIGH"
]:

    subset = results[
        results["risk_level"] == level
    ]

    fraud_count = (
        subset["fraud"] == 1
    ).sum()

    total_count = len(subset)

    fraud_rate = (
        fraud_count / total_count * 100
        if total_count > 0
        else 0
    )

    print(
        f"\n{level}:"
    )

    print(
        f"  Transactions: {total_count}"
    )

    print(
        f"  Fraud: {fraud_count}"
    )

    print(
        f"  Fraud rate: {fraud_rate:.2f}%"
    )


# =========================================================
# 16. SAVE HELD-OUT RESULTS
# =========================================================

results.to_csv(
    "results/razorguard_heldout_results.csv",
    index=False
)


print(
    "\nHeld-out results saved to:"
    " results/razorguard_heldout_results.csv"
)


print(
    "\nHeld-out evaluation complete!"
)
