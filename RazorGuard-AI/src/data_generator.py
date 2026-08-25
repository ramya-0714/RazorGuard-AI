import numpy as np
import pandas as pd
from pathlib import Path


# Reproducibility
np.random.seed(42)

# Number of transactions
N_TRANSACTIONS = 10000


def generate_transactions(n=N_TRANSACTIONS):
    # -----------------------------
    # Basic transaction information
    # -----------------------------

    transaction_id = [
        f"TXN{str(i).zfill(6)}"
        for i in range(1, n + 1)
    ]

    customer_id = np.random.randint(10000, 13000, n)

    amount = np.round(
        np.random.lognormal(mean=7.0, sigma=1.0, size=n),
        2
    )

    transaction_hour = np.random.randint(0, 24, n)

    # -----------------------------
    # Device information
    # -----------------------------

    is_new_device = np.random.binomial(1, 0.12, n)

    device_age_days = np.where(
        is_new_device == 1,
        np.random.randint(0, 15, n),
        np.random.randint(30, 1500, n)
    )

    # -----------------------------
    # Location information
    # -----------------------------

    location_change = np.random.binomial(1, 0.08, n)

    distance_from_last_transaction = np.where(
        location_change == 1,
        np.random.randint(100, 2500, n),
        np.random.randint(0, 100, n)
    )

    # -----------------------------
    # Transaction behaviour
    # -----------------------------

    transactions_last_10min = np.random.poisson(1.5, n)

    failed_attempts = np.random.poisson(0.4, n)

    # -----------------------------
    # Customer history
    # -----------------------------

    avg_customer_amount = np.round(
        np.random.lognormal(mean=6.5, sigma=0.7, size=n),
        2
    )

    customer_account_age_days = np.random.randint(
        10,
        2000,
        n
    )

    # How unusual is this transaction?
    amount_deviation = np.round(
        amount / (avg_customer_amount + 1),
        2
    )

    # -----------------------------
    # Payment method
    # -----------------------------

    payment_method = np.random.choice(
        ["UPI", "CARD", "NETBANKING", "WALLET"],
        size=n,
        p=[0.45, 0.30, 0.15, 0.10]
    )

    # -----------------------------
    # Hidden fraud-risk calculation
    # -----------------------------
    #
    # We intentionally create realistic
    # suspicious behaviour patterns.

    risk_score = (
        0.8 * is_new_device
        + 1.0 * location_change
        + 0.35 * transactions_last_10min
        + 0.55 * failed_attempts
        + 0.75 * (amount_deviation > 4)
        + 0.45 * (amount_deviation > 8)
        + 0.35 * (distance_from_last_transaction > 500)
        + 0.30 * (transaction_hour < 5)
        + 0.25 * (customer_account_age_days < 30)
    )

    # Add a small amount of randomness
    risk_score += np.random.normal(0, 0.35, n)

    # Convert risk into fraud probability
    fraud_probability = 1 / (1 + np.exp(-(risk_score - 4.0)))

    # Generate fraud labels
    fraud = np.random.binomial(
        1,
        fraud_probability,
        n
    )

    # -----------------------------
    # Create dataframe
    # -----------------------------

    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "amount": amount,
        "transaction_hour": transaction_hour,
        "is_new_device": is_new_device,
        "device_age_days": device_age_days,
        "location_change": location_change,
        "distance_from_last_transaction": distance_from_last_transaction,
        "transactions_last_10min": transactions_last_10min,
        "failed_attempts": failed_attempts,
        "avg_customer_amount": avg_customer_amount,
        "amount_deviation": amount_deviation,
        "customer_account_age_days": customer_account_age_days,
        "payment_method": payment_method,
        "fraud": fraud
    })

    return df


def main():
    print("Generating RazorGuard AI transaction dataset...")

    df = generate_transactions()

    # Create data directory if needed
    output_directory = Path("data")
    output_directory.mkdir(exist_ok=True)

    output_file = output_directory / "transactions.csv"

    df.to_csv(output_file, index=False)

    print("\nDataset generated successfully!")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_file}")

    print("\nFraud distribution:")
    print(df["fraud"].value_counts())

    print("\nFirst 5 transactions:")
    print(df.head())


if __name__ == "__main__":
    main()