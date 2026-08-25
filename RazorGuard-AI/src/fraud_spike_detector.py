import pandas as pd
import numpy as np
import os


# =========================================================
# RAZORGUARD AI - FRAUD SPIKE DETECTOR
# =========================================================

DATA_PATH = "data/transactions_spike_test.csv"
OUTPUT_PATH = "results/fraud_spike_analysis.csv"


print("=" * 70)
print("RAZORGUARD AI - FRAUD SPIKE DETECTOR")
print("=" * 70)


# =========================================================
# 1. LOAD TRANSACTION DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print(
    f"\nTransactions loaded: {len(df)}"
)


# =========================================================
# 2. CREATE TIME WINDOW
# =========================================================

df["time_window"] = (
    df["transaction_hour"]
    .astype(int)
)


# =========================================================
# 3. CREATE STRONG SUSPICIOUS-ACTIVITY SIGNAL
# =========================================================

new_device = (
    df["is_new_device"] == 1
)

location_change = (
    df["location_change"] == 1
)

high_velocity = (
    df["transactions_last_10min"] >= 5
)

failed_attempts = (
    df["failed_attempts"] >= 2
)

high_amount_deviation = (
    df["amount_deviation"] >= 4
)

large_distance = (
    df["distance_from_last_transaction"] >= 700
)


df["suspicious_signal"] = (

    (
        new_device
        &
        high_amount_deviation
    )

    |

    (
        location_change
        &
        large_distance
    )

    |

    (
        high_velocity
        &
        failed_attempts
    )

    |

    (
        new_device
        &
        location_change
    )

).astype(int)


# =========================================================
# 4. AGGREGATE BY TIME WINDOW
# =========================================================

hourly = (
    df.groupby("time_window")
    .agg(
        transactions=(
            "transaction_id",
            "count"
        ),

        suspicious_transactions=(
            "suspicious_signal",
            "sum"
        ),

        actual_fraud=(
            "fraud",
            "sum"
        )
    )
    .reset_index()
)


# =========================================================
# 5. SUSPICIOUS RATE
# =========================================================

hourly["suspicious_rate"] = (
    hourly["suspicious_transactions"]
    /
    hourly["transactions"]
)


# =========================================================
# 6. BASELINE
# =========================================================

baseline_rate = (
    hourly["suspicious_rate"]
    .mean()
)


print(
    f"\nBaseline suspicious rate: "
    f"{baseline_rate * 100:.2f}%"
)


# =========================================================
# 7. SPIKE RATIO
# =========================================================

if baseline_rate > 0:

    hourly["spike_ratio"] = (
        hourly["suspicious_rate"]
        /
        baseline_rate
    )

else:

    hourly["spike_ratio"] = 0


# =========================================================
# 8. SPIKE DETECTION
# =========================================================

hourly["spike_detected"] = (
    hourly["spike_ratio"] >= 2
)


# =========================================================
# 9. RISK LEVEL
# =========================================================

def get_spike_level(spike_ratio):

    if spike_ratio >= 3:

        return "HIGH"

    elif spike_ratio >= 2:

        return "MEDIUM"

    else:

        return "NORMAL"


hourly["risk_level"] = (
    hourly["spike_ratio"]
    .apply(get_spike_level)
)


# =========================================================
# 10. SPIKE SCORE
# =========================================================

hourly["spike_score"] = (

    hourly["spike_ratio"]
    .clip(upper=5)
    /
    5
    *
    100
)

hourly["spike_score"] = (
    hourly["spike_score"]
    .round(2)
)


# =========================================================
# 11. RECOMMENDED ACTION
# =========================================================

def get_action(risk_level):

    if risk_level == "HIGH":

        return "INVESTIGATE PAYMENT ACTIVITY"

    elif risk_level == "MEDIUM":

        return "MONITOR / STEP-UP REVIEW"

    else:

        return "NO ACTION"


hourly["recommended_action"] = (
    hourly["risk_level"]
    .apply(get_action)
)


# =========================================================
# 12. FRAUD RATE
# =========================================================

hourly["fraud_rate"] = (

    hourly["actual_fraud"]
    /
    hourly["transactions"]
    *
    100
).round(2)


# =========================================================
# 13. DISPLAY ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("HOURLY FRAUD-SPIKE ANALYSIS")
print("=" * 70)


display_columns = [

    "time_window",
    "transactions",
    "suspicious_transactions",
    "suspicious_rate",
    "actual_fraud",
    "fraud_rate",
    "spike_ratio",
    "spike_score",
    "risk_level",
    "recommended_action"
]


display_df = hourly[
    display_columns
].copy()


display_df["suspicious_rate"] = (
    display_df["suspicious_rate"]
    * 100
).round(2)


display_df["spike_ratio"] = (
    display_df["spike_ratio"]
    .round(2)
)


print(
    display_df.to_string(
        index=False
    )
)


# =========================================================
# 14. DETECTED SPIKES
# =========================================================

spikes = hourly[
    hourly["spike_detected"]
].copy()


print("\n" + "=" * 70)
print("DETECTED FRAUD SPIKES")
print("=" * 70)


print(
    f"\nNumber of spike windows: "
    f"{len(spikes)}"
)


if len(spikes) > 0:

    spike_display = spikes[
        display_columns
    ].copy()


    spike_display["suspicious_rate"] = (
        spike_display["suspicious_rate"]
        * 100
    ).round(2)


    spike_display["spike_ratio"] = (
        spike_display["spike_ratio"]
        .round(2)
    )


    print(
        spike_display
        .sort_values(
            by="spike_score",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nNo significant fraud spikes detected."
    )


# =========================================================
# 15. SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("SPIKE DETECTOR SUMMARY")
print("=" * 70)


print(
    f"\nBaseline suspicious rate: "
    f"{baseline_rate * 100:.2f}%"
)


print(
    f"Total time windows: "
    f"{len(hourly)}"
)


print(
    f"Spike windows detected: "
    f"{len(spikes)}"
)


print(
    f"High-risk windows: "
    f"{sum(hourly['risk_level'] == 'HIGH')}"
)


print(
    f"Medium-risk windows: "
    f"{sum(hourly['risk_level'] == 'MEDIUM')}"
)


print(
    f"Normal windows: "
    f"{sum(hourly['risk_level'] == 'NORMAL')}"
)


# =========================================================
# 16. SAVE RESULTS
# =========================================================

os.makedirs(
    "results",
    exist_ok=True
)


hourly.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nResults saved to: "
    f"{OUTPUT_PATH}"
)


print(
    "\nFraud spike analysis complete!"
)