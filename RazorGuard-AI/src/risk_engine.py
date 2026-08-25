import pandas as pd
from xgboost import XGBClassifier


# =========================================================
# RAZORGUARD AI - HYBRID RISK ENGINE
# =========================================================

MODEL_PATH = "models/razorguard_clean_model.json"


# =========================================================
# 1. LOAD MODEL
# =========================================================

model = XGBClassifier()

model.load_model(MODEL_PATH)

print("RazorGuard model loaded successfully!")


# =========================================================
# 2. RISK LEVEL
# =========================================================

def get_risk_level(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


# =========================================================
# 3. RECOMMENDED ACTION
# =========================================================

def get_action(risk_level):

    if risk_level == "HIGH":
        return "MANUAL REVIEW"

    elif risk_level == "MEDIUM":
        return "STEP-UP VERIFICATION"

    else:
        return "ALLOW"


# =========================================================
# 4. ANALYZE RISK SIGNALS
# =========================================================

def calculate_signal_score(transaction):

    score = 0
    reasons = []


    # -----------------------------------------------------
    # New device
    # -----------------------------------------------------

    if transaction["is_new_device"] == 1:

        score += 15

        reasons.append(
            "New device detected"
        )


    # -----------------------------------------------------
    # Location change
    # -----------------------------------------------------

    if transaction["location_change"] == 1:

        score += 15

        reasons.append(
            "Location changed"
        )


    # -----------------------------------------------------
    # High transaction velocity
    # -----------------------------------------------------

    velocity = transaction[
        "transactions_last_10min"
    ]

    if velocity >= 5:

        score += 20

        reasons.append(
            "High transaction velocity"
        )

    elif velocity >= 3:

        score += 8

        reasons.append(
            "Elevated transaction velocity"
        )


    # -----------------------------------------------------
    # Failed attempts
    # -----------------------------------------------------

    failures = transaction[
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


    # -----------------------------------------------------
    # Amount deviation
    # -----------------------------------------------------

    deviation = transaction[
        "amount_deviation"
    ]

    if deviation >= 7:

        score += 20

        reasons.append(
            "Transaction amount is extremely unusual"
        )

    elif deviation >= 4:

        score += 15

        reasons.append(
            "Transaction amount is unusually high"
        )

    elif deviation >= 2:

        score += 7

        reasons.append(
            "Transaction amount is above normal"
        )


    # -----------------------------------------------------
    # Distance anomaly
    # -----------------------------------------------------

    distance = transaction[
        "distance_from_last_transaction"
    ]

    if distance >= 1500:

        score += 15

        reasons.append(
            "Very large distance from previous transaction"
        )

    elif distance >= 700:

        score += 10

        reasons.append(
            "Large distance from previous transaction"
        )


    # -----------------------------------------------------
    # Combination bonus
    # -----------------------------------------------------

    # New device + location change
    if (
        transaction["is_new_device"] == 1
        and transaction["location_change"] == 1
    ):

        score += 10

        reasons.append(
            "New device combined with location change"
        )


    # New device + high amount
    if (
        transaction["is_new_device"] == 1
        and transaction["amount_deviation"] >= 4
    ):

        score += 10

        reasons.append(
            "New device combined with unusual amount"
        )


    # Location + large distance
    if (
        transaction["location_change"] == 1
        and transaction[
            "distance_from_last_transaction"
        ] >= 700
    ):

        score += 10

        reasons.append(
            "Location change with large travel distance"
        )


    return min(score, 100), reasons


# =========================================================
# 5. ANALYZE TRANSACTION
# =========================================================

def analyze_transaction(transaction):


    # -----------------------------------------------------
    # ML MODEL INPUT
    # -----------------------------------------------------

    transaction_df = pd.DataFrame(
        [transaction]
    )


    transaction_df = transaction_df.drop(
        columns=[
            "transaction_id",
            "customer_id"
        ],
        errors="ignore"
    )


    transaction_df = pd.get_dummies(
        transaction_df,
        columns=["payment_method"],
        dtype=int
    )


    # Get model features
    expected_features = (
        model.get_booster().feature_names
    )


    # Add missing features
    for feature in expected_features:

        if feature not in transaction_df.columns:

            transaction_df[feature] = 0


    # Correct feature order
    transaction_df = transaction_df[
        expected_features
    ]


    # -----------------------------------------------------
    # ML PROBABILITY
    # -----------------------------------------------------

    probability = float(
        model.predict_proba(
            transaction_df
        )[0][1]
    )


    ml_score = probability * 100


    # -----------------------------------------------------
    # BEHAVIOURAL SIGNAL SCORE
    # -----------------------------------------------------

    signal_score, reasons = (
        calculate_signal_score(
            transaction
        )
    )


    # -----------------------------------------------------
    # HYBRID SCORE
    # -----------------------------------------------------

    # 40% machine-learning signal
    # 60% transparent behavioural signals

    risk_score = (
        (ml_score * 0.40)
        +
        (signal_score * 0.60)
    )


    risk_score = float(
        round(
            min(risk_score, 100),
            2
        )
    )


    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    risk_level = get_risk_level(
        risk_score
    )


    # -----------------------------------------------------
    # ACTION
    # -----------------------------------------------------

    action = get_action(
        risk_level
    )


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "ml_score":
            round(
                ml_score,
                2
            ),

        "signal_score":
            signal_score,

        "reasons":
            reasons,

        "recommended_action":
            action
    }


# =========================================================
# 6. TEST TRANSACTION
# =========================================================

if __name__ == "__main__":


    # -----------------------------------------------------
    # CURRENT TEST
    #
    # This is the moderately suspicious transaction.
    # -----------------------------------------------------

    test_transaction = {

    "transaction_id": "TEST_HIGH_001",

    "customer_id": 10001,

    "amount": 5000,

    "transaction_hour": 2,

    "is_new_device": 1,

    "device_age_days": 3,

    "location_change": 1,

    "distance_from_last_transaction": 1200,

    "transactions_last_10min": 7,

    "failed_attempts": 3,

    "avg_customer_amount": 800,

    "amount_deviation": 6.25,

    "customer_account_age_days": 400,

    "payment_method": "CARD"
}

    # Analyze
    result = analyze_transaction(
        test_transaction
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    print("\n" + "=" * 60)
    print("RAZORGUARD AI - TRANSACTION ANALYSIS")
    print("=" * 60)


    print(
        f"\nRisk Score: "
        f"{result['risk_score']:.2f}/100"
    )


    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )


    print(
        f"\nML Model Score: "
        f"{result['ml_score']:.2f}"
    )


    print(
        f"Behavioural Signal Score: "
        f"{result['signal_score']:.2f}"
    )


    print("\nRisk Signals:")


    if result["reasons"]:

        for reason in result["reasons"]:

            print(
                f"  • {reason}"
            )

    else:

        print(
            "  • No major risk signals detected"
        )


    print(
        f"\nRecommended Action: "
        f"{result['recommended_action']}"
    )


    print("\n" + "=" * 60)