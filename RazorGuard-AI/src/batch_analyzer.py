import pandas as pd
from xgboost import XGBClassifier


# =========================================================
# RAZORGUARD AI - BATCH TRANSACTION ANALYZER
# =========================================================

MODEL_PATH = "models/razorguard_clean_model.json"
DATA_PATH = "data/transactions_v2.csv"


# =========================================================
# 1. LOAD MODEL
# =========================================================

model = XGBClassifier()

model.load_model(MODEL_PATH)

print("RazorGuard model loaded successfully!")


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print(
    f"Transactions loaded: {len(df)}"
)


# =========================================================
# 3. PREPARE DATA FOR MODEL
# =========================================================

X = df.drop(
    columns=[
        "fraud",
        "transaction_id",
        "customer_id"
    ],
    errors="ignore"
)

X = pd.get_dummies(
    X,
    columns=["payment_method"],
    dtype=int
)


# Get the exact features expected by the model
expected_features = (
    model.get_booster().feature_names
)


# Add missing features
for feature in expected_features:

    if feature not in X.columns:
        X[feature] = 0


# Keep exact feature order
X = X[expected_features]


# =========================================================
# 4. GET ML RISK PROBABILITY
# =========================================================

probabilities = model.predict_proba(X)[:, 1]

ml_scores = probabilities * 100


# =========================================================
# 5. CALCULATE BEHAVIOURAL SCORE
# =========================================================

def calculate_signal_score(row):

    score = 0

    reasons = []


    # New device
    if row["is_new_device"] == 1:

        score += 15

        reasons.append(
            "New device"
        )


    # Location change
    if row["location_change"] == 1:

        score += 15

        reasons.append(
            "Location change"
        )


    # Velocity
    velocity = row[
        "transactions_last_10min"
    ]

    if velocity >= 5:

        score += 20

        reasons.append(
            "High velocity"
        )

    elif velocity >= 3:

        score += 8

        reasons.append(
            "Elevated velocity"
        )


    # Failed attempts
    failures = row[
        "failed_attempts"
    ]

    if failures >= 3:

        score += 15

        reasons.append(
            "Multiple failed attempts"
        )

    elif failures >= 2:

        score += 8

        reasons.append(
            "Repeated failed attempts"
        )


    # Amount deviation
    deviation = row[
        "amount_deviation"
    ]

    if deviation >= 7:

        score += 20

        reasons.append(
            "Extreme amount deviation"
        )

    elif deviation >= 4:

        score += 15

        reasons.append(
            "High amount deviation"
        )

    elif deviation >= 2:

        score += 7

        reasons.append(
            "Above-normal amount"
        )


    # Distance
    distance = row[
        "distance_from_last_transaction"
    ]

    if distance >= 1500:

        score += 15

        reasons.append(
            "Very large distance"
        )

    elif distance >= 700:

        score += 10

        reasons.append(
            "Large distance"
        )


    # Combination: new device + location
    if (
        row["is_new_device"] == 1
        and row["location_change"] == 1
    ):

        score += 10

        reasons.append(
            "New device + location change"
        )


    # Combination: new device + amount
    if (
        row["is_new_device"] == 1
        and row["amount_deviation"] >= 4
    ):

        score += 10

        reasons.append(
            "New device + unusual amount"
        )


    # Combination: location + distance
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
# 6. APPLY BEHAVIOURAL SCORING
# =========================================================

signal_scores = []
reason_lists = []


for _, row in df.iterrows():

    score, reasons = calculate_signal_score(
        row
    )

    signal_scores.append(score)

    reason_lists.append(
        reasons
    )


# =========================================================
# 7. HYBRID RISK SCORE
# =========================================================

risk_scores = (
    (ml_scores * 0.40)
    +
    (pd.Series(signal_scores) * 0.60)
)


risk_scores = risk_scores.clip(
    upper=100
).round(2)


# =========================================================
# 8. RISK LEVEL
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
# 9. RECOMMENDED ACTION
# =========================================================

def get_action(level):

    if level == "HIGH":
        return "MANUAL REVIEW"

    elif level == "MEDIUM":
        return "STEP-UP VERIFICATION"

    else:
        return "ALLOW"


actions = [
    get_action(level)
    for level in risk_levels
]


# =========================================================
# 10. CREATE RESULTS TABLE
# =========================================================

results = df.copy()

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

results["recommended_action"] = (
    actions
)


results["risk_reasons"] = [
    " | ".join(reasons)
    if reasons
    else "No major risk signals"
    for reasons in reason_lists
]


# =========================================================
# 11. SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("RAZORGUARD AI - BATCH ANALYSIS")
print("=" * 65)

print(
    f"\nTransactions analyzed: "
    f"{len(results)}"
)


print(
    f"\nHIGH RISK: "
    f"{sum(results['risk_level'] == 'HIGH')}"
)

print(
    f"MEDIUM RISK: "
    f"{sum(results['risk_level'] == 'MEDIUM')}"
)

print(
    f"LOW RISK: "
    f"{sum(results['risk_level'] == 'LOW')}"
)


print("\nRecommended actions:")

print(
    f"Manual Review: "
    f"{sum(results['recommended_action'] == 'MANUAL REVIEW')}"
)

print(
    f"Step-up Verification: "
    f"{sum(results['recommended_action'] == 'STEP-UP VERIFICATION')}"
)

print(
    f"Allow: "
    f"{sum(results['recommended_action'] == 'ALLOW')}"
)


# =========================================================
# 12. ACTUAL FRAUD INFORMATION
# =========================================================

if "fraud" in results.columns:

    actual_fraud = (
        results["fraud"] == 1
    )

    print(
        f"\nActual fraud transactions: "
        f"{actual_fraud.sum()}"
    )


    high_risk_actual_fraud = (
        actual_fraud
        &
        (results["risk_level"] == "HIGH")
    )

    print(
        f"Fraud detected as HIGH risk: "
        f"{high_risk_actual_fraud.sum()}"
    )


# =========================================================
# 13. SHOW TOP 10 RISKS
# =========================================================

print("\n" + "=" * 65)
print("TOP 10 HIGHEST-RISK TRANSACTIONS")
print("=" * 65)

display_columns = [
    "transaction_id",
    "risk_score",
    "risk_level",
    "recommended_action",
    "risk_reasons"
]

print(
    results[
        display_columns
    ]
    .sort_values(
        by="risk_score",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# =========================================================
# 14. SAVE RESULTS
# =========================================================

OUTPUT_PATH = (
    "results/razorguard_batch_results.csv"
)

import os

os.makedirs(
    "results",
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nResults saved to: "
    f"{OUTPUT_PATH}"
)

print("\nBatch analysis complete!")