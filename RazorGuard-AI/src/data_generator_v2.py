import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# RAZORGUARD AI - SYNTHETIC FRAUD DATASET V2
# =========================================================

np.random.seed(42)

N_TRANSACTIONS = 12000
TARGET_FRAUD_RATE = 0.08


def generate_transactions(n=N_TRANSACTIONS):

    # -----------------------------------------------------
    # 1. BASIC TRANSACTION INFORMATION
    # -----------------------------------------------------

    transaction_id = [
        f"TXN{str(i).zfill(6)}"
        for i in range(1, n + 1)
    ]

    customer_id = np.random.randint(
        10000,
        13000,
        n
    )

    amount = np.round(
        np.random.lognormal(
            mean=6.8,
            sigma=0.85,
            size=n
        ),
        2
    )

    transaction_hour = np.random.randint(
        0,
        24,
        n
    )

    # -----------------------------------------------------
    # 2. DEVICE INFORMATION
    # -----------------------------------------------------

    is_new_device = np.random.binomial(
        1,
        0.12,
        n
    )

    device_age_days = np.where(
        is_new_device == 1,
        np.random.randint(
            0,
            20,
            n
        ),
        np.random.randint(
            30,
            1500,
            n
        )
    )

    # -----------------------------------------------------
    # 3. CUSTOMER ACCOUNT INFORMATION
    # -----------------------------------------------------

    customer_account_age_days = np.random.randint(
        10,
        2000,
        n
    )

    # -----------------------------------------------------
    # 4. LOCATION INFORMATION
    # -----------------------------------------------------

    location_change = np.random.binomial(
        1,
        0.08,
        n
    )

    distance_from_last_transaction = np.where(
        location_change == 1,
        np.random.randint(
            100,
            2500,
            n
        ),
        np.random.randint(
            0,
            100,
            n
        )
    )

    # -----------------------------------------------------
    # 5. TRANSACTION BEHAVIOUR
    # -----------------------------------------------------

    transactions_last_10min = np.random.poisson(
        1.3,
        n
    )

    failed_attempts = np.random.poisson(
        0.35,
        n
    )

    # -----------------------------------------------------
    # 6. CUSTOMER SPENDING HISTORY
    # -----------------------------------------------------

    avg_customer_amount = np.round(
        np.random.lognormal(
            mean=6.2,
            sigma=0.65,
            size=n
        ),
        2
    )

    amount_deviation = np.round(
        amount / (avg_customer_amount + 1),
        2
    )

    # -----------------------------------------------------
    # 7. PAYMENT METHOD
    # -----------------------------------------------------

    payment_method = np.random.choice(
        [
            "UPI",
            "CARD",
            "NETBANKING",
            "WALLET"
        ],
        size=n,
        p=[
            0.45,
            0.30,
            0.15,
            0.10
        ]
    )

    # -----------------------------------------------------
    # 8. START WITH ALL TRANSACTIONS AS LEGITIMATE
    # -----------------------------------------------------

    fraud = np.zeros(
        n,
        dtype=int
    )

    # -----------------------------------------------------
    # 9. FRAUD SCENARIO 1
    # ACCOUNT TAKEOVER
    # -----------------------------------------------------

    account_takeover = (
        (is_new_device == 1)
        & (location_change == 1)
        & (amount_deviation > 4)
    )

    fraud[account_takeover] = 1

    # -----------------------------------------------------
    # 10. FRAUD SCENARIO 2
    # TRANSACTION VELOCITY ATTACK
    # -----------------------------------------------------

    velocity_attack = (
        (transactions_last_10min >= 5)
        & (failed_attempts >= 2)
    )

    fraud[velocity_attack] = 1

    # -----------------------------------------------------
    # 11. FRAUD SCENARIO 3
    # IMPOSSIBLE TRAVEL / LOCATION ANOMALY
    # -----------------------------------------------------

    impossible_travel = (
        (distance_from_last_transaction > 700)
        & (location_change == 1)
        & (transactions_last_10min >= 3)
    )

    fraud[impossible_travel] = 1

    # -----------------------------------------------------
    # 12. FRAUD SCENARIO 4
    # EXTREME AMOUNT ANOMALY
    # -----------------------------------------------------

    amount_attack = (
        (amount_deviation > 7)
        & (is_new_device == 1)
    )

    fraud[amount_attack] = 1

    # -----------------------------------------------------
    # 13. ADDITIONAL HIDDEN FRAUD
    # -----------------------------------------------------

    hidden_fraud = (
        np.random.random(n) < 0.02
    )

    fraud[hidden_fraud] = 1

    # -----------------------------------------------------
    # 14. CONTROL FINAL FRAUD RATE
    # -----------------------------------------------------
    #
    # We want exactly 8% fraud.
    #
    # 12,000 × 8% = 960 fraud transactions.
    #
    # If our scenarios create more than 960,
    # we randomly keep 960.
    #
    # If they create fewer than 960,
    # we add additional fraud cases.
    # -----------------------------------------------------

    target_fraud_count = int(
        n * TARGET_FRAUD_RATE
    )

    fraud_indices = np.where(
        fraud == 1
    )[0]

    # Too many fraud cases
    if len(fraud_indices) > target_fraud_count:

        selected_indices = np.random.choice(
            fraud_indices,
            size=target_fraud_count,
            replace=False
        )

        fraud[:] = 0

        fraud[selected_indices] = 1

    # Too few fraud cases
    elif len(fraud_indices) < target_fraud_count:

        legitimate_indices = np.where(
            fraud == 0
        )[0]

        additional_count = (
            target_fraud_count
            - len(fraud_indices)
        )

        additional_indices = np.random.choice(
            legitimate_indices,
            size=additional_count,
            replace=False
        )

        fraud[additional_indices] = 1

    # -----------------------------------------------------
    # 15. CREATE DATAFRAME
    # -----------------------------------------------------

    df = pd.DataFrame({

        "transaction_id":
            transaction_id,

        "customer_id":
            customer_id,

        "amount":
            amount,

        "transaction_hour":
            transaction_hour,

        "is_new_device":
            is_new_device,

        "device_age_days":
            device_age_days,

        "location_change":
            location_change,

        "distance_from_last_transaction":
            distance_from_last_transaction,

        "transactions_last_10min":
            transactions_last_10min,

        "failed_attempts":
            failed_attempts,

        "avg_customer_amount":
            avg_customer_amount,

        "amount_deviation":
            amount_deviation,

        "customer_account_age_days":
            customer_account_age_days,

        "payment_method":
            payment_method,

        "fraud":
            fraud
    })

    return df


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("RAZORGUARD AI - V2 DATASET GENERATOR")
    print("=" * 60)

    print("\nGenerating synthetic transaction data...")

    df = generate_transactions()

    # -----------------------------------------------------
    # Create data directory
    # -----------------------------------------------------

    output_directory = Path("data")

    output_directory.mkdir(
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save dataset
    # -----------------------------------------------------

    output_file = (
        output_directory
        / "transactions_v2.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nDataset generated successfully!")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Fraud rate: "
        f"{df['fraud'].mean() * 100:.2f}%"
    )

    print("\nFraud distribution:")

    print(
        df["fraud"].value_counts()
    )

    print(
        f"\nSaved to: {output_file}"
    )


if __name__ == "__main__":
    main()