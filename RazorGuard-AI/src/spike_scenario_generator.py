import pandas as pd
import numpy as np
import os


# =========================================================
# RAZORGUARD AI - FRAUD SPIKE SCENARIO GENERATOR
# =========================================================

INPUT_PATH = "data/transactions_v2.csv"
OUTPUT_PATH = "data/transactions_spike_test.csv"

print("=" * 70)
print("RAZORGUARD AI - FRAUD SPIKE SCENARIO GENERATOR")
print("=" * 70)


# =========================================================
# 1. LOAD ORIGINAL DATA
# =========================================================

df = pd.read_csv(INPUT_PATH)

print(
    f"\nOriginal transactions: {len(df)}"
)


# =========================================================
# 2. COPY DATA
# =========================================================

scenario_df = df.copy()


# =========================================================
# 3. SELECT ATTACK WINDOWS
# =========================================================
#
# We will simulate two controlled fraud attacks:
#
# Hour 10 -> Attack Window 1
# Hour 18 -> Attack Window 2
#
# The original dataset remains unchanged.
# =========================================================

attack_hours = [10, 18]

print(
    f"\nAttack windows: {attack_hours}"
)


# =========================================================
# 4. CREATE RANDOM GENERATOR
# =========================================================

rng = np.random.default_rng(42)


# =========================================================
# 5. INJECT FRAUD-BURST BEHAVIOUR
# =========================================================

for hour in attack_hours:

    # Find transactions belonging to this hour
    indices = scenario_df[
        scenario_df["transaction_hour"] == hour
    ].index

    if len(indices) == 0:
        continue

    # Select approximately 30% of transactions
    # in the attack window.
    attack_count = int(
        len(indices) * 0.30
    )

    attack_indices = rng.choice(
        indices,
        size=attack_count,
        replace=False
    )

    # -----------------------------------------------------
    # Inject suspicious behaviour
    # -----------------------------------------------------

    # New devices
    scenario_df.loc[
        attack_indices,
        "is_new_device"
    ] = 1


    # Location changes
    scenario_df.loc[
        attack_indices,
        "location_change"
    ] = 1


    # High transaction velocity
    scenario_df.loc[
        attack_indices,
        "transactions_last_10min"
    ] = rng.integers(
        5,
        10,
        size=attack_count
    )


    # Failed attempts
    scenario_df.loc[
        attack_indices,
        "failed_attempts"
    ] = rng.integers(
        2,
        5,
        size=attack_count
    )


    # Large distance
    scenario_df.loc[
        attack_indices,
        "distance_from_last_transaction"
    ] = rng.integers(
        700,
        2000,
        size=attack_count
    )


    # Large amount deviation
    scenario_df.loc[
        attack_indices,
        "amount_deviation"
    ] = rng.uniform(
        4.5,
        8.0,
        size=attack_count
    )


    # Make the transaction amount consistent
    # with the deviation.
    scenario_df.loc[
        attack_indices,
        "amount"
    ] = (
        scenario_df.loc[
            attack_indices,
            "avg_customer_amount"
        ]
        *
        scenario_df.loc[
            attack_indices,
            "amount_deviation"
        ]
    )


    # Mark these injected transactions as fraud
    scenario_df.loc[
        attack_indices,
        "fraud"
    ] = 1


    print(
        f"Hour {hour}: "
        f"{attack_count} attack transactions injected"
    )


# =========================================================
# 6. SAVE SCENARIO DATASET
# =========================================================

os.makedirs(
    "data",
    exist_ok=True
)

scenario_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# 7. SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("SCENARIO DATASET CREATED")
print("=" * 70)

print(
    f"\nRows: {len(scenario_df)}"
)

print(
    f"Fraud transactions: "
    f"{scenario_df['fraud'].sum()}"
)

print(
    f"Overall fraud rate: "
    f"{scenario_df['fraud'].mean() * 100:.2f}%"
)

print(
    f"\nSaved to:"
    f" {OUTPUT_PATH}"
)

print(
    "\nFraud spike scenario generation complete!"
)
