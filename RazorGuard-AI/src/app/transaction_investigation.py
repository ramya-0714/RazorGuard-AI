import streamlit as st
import pandas as pd
import os


# =========================================================
# RAZORGUARD AI - TRANSACTION INVESTIGATION
# =========================================================

DATA_PATH = "data/transactions_spike_test.csv"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RazorGuard - Investigation",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

if not os.path.exists(DATA_PATH):
    st.error("Transaction dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)


# =========================================================
# CALCULATE DEMO RISK SCORE
# =========================================================

def calculate_risk(row):

    score = 0

    if row["is_new_device"] == 1:
        score += 15

    if row["location_change"] == 1:
        score += 15

    if row["transactions_last_10min"] >= 5:
        score += 20
    elif row["transactions_last_10min"] >= 3:
        score += 8

    if row["failed_attempts"] >= 3:
        score += 15
    elif row["failed_attempts"] >= 2:
        score += 8

    if row["amount_deviation"] >= 7:
        score += 20
    elif row["amount_deviation"] >= 4:
        score += 15
    elif row["amount_deviation"] >= 2:
        score += 7

    if row["distance_from_last_transaction"] >= 1500:
        score += 15
    elif row["distance_from_last_transaction"] >= 700:
        score += 10

    if (
        row["is_new_device"] == 1
        and row["location_change"] == 1
    ):
        score += 10

    if (
        row["is_new_device"] == 1
        and row["amount_deviation"] >= 4
    ):
        score += 10

    if (
        row["location_change"] == 1
        and row["distance_from_last_transaction"] >= 700
    ):
        score += 10

    return min(score, 100)


# =========================================================
# FIND HIGH-RISK DEMO TRANSACTION
# =========================================================

df["demo_risk_score"] = df.apply(
    calculate_risk,
    axis=1
)

high_risk_transactions = df[
    df["demo_risk_score"] >= 70
].sort_values(
    "demo_risk_score",
    ascending=False
)


# =========================================================
# HEADER
# =========================================================

st.title("🔎 Transaction Investigation")

st.caption(
    "Analyze individual transactions and identify "
    "potential payment-risk signals."
)


# =========================================================
# TRANSACTION SELECTION
# =========================================================

transaction_ids = (
    df["transaction_id"]
    .astype(str)
    .tolist()
)


# Start with the strongest transaction for demo purposes

if len(high_risk_transactions) > 0:

    default_transaction = str(
        high_risk_transactions.iloc[0]["transaction_id"]
    )

else:

    default_transaction = transaction_ids[0]


default_index = transaction_ids.index(
    default_transaction
)


selected_id = st.selectbox(
    "Select Transaction",
    transaction_ids,
    index=default_index
)


transaction = df[
    df["transaction_id"].astype(str)
    == selected_id
]


if transaction.empty:

    st.error(
        "Transaction not found."
    )

    st.stop()


row = transaction.iloc[0]


# =========================================================
# TRANSACTION DETAILS
# =========================================================

st.divider()

st.subheader(
    "Transaction Details"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Transaction ID",
        str(row["transaction_id"])
    )


with c2:

    st.metric(
        "Amount",
        f"₹{float(row['amount']):,.2f}"
    )


with c3:

    st.metric(
        "Transaction Hour",
        int(row["transaction_hour"])
    )


with c4:

    fraud_status = (
        "Fraud"
        if int(row["fraud"]) == 1
        else "Legitimate"
    )

    st.metric(
        "Dataset Label",
        fraud_status
    )


# =========================================================
# RISK SIGNALS
# =========================================================

st.subheader(
    "⚠️ Risk Signals"
)


signals = []


if row["is_new_device"] == 1:

    signals.append(
        "📱 New device detected"
    )


if row["location_change"] == 1:

    signals.append(
        "📍 Location changed"
    )


if row["transactions_last_10min"] >= 5:

    signals.append(
        "⚡ High transaction velocity"
    )


if row["failed_attempts"] >= 2:

    signals.append(
        "🔐 Multiple failed attempts"
    )


if row["amount_deviation"] >= 4:

    signals.append(
        "💰 Transaction amount is unusually high"
    )


if row["distance_from_last_transaction"] >= 700:

    signals.append(
        "🌍 Large distance from previous transaction"
    )


if (
    row["is_new_device"] == 1
    and row["location_change"] == 1
):

    signals.append(
        "🚨 New device + location change"
    )


if (
    row["is_new_device"] == 1
    and row["amount_deviation"] >= 4
):

    signals.append(
        "🚨 New device + unusual amount"
    )


if (
    row["location_change"] == 1
    and row["distance_from_last_transaction"] >= 700
):

    signals.append(
        "🚨 Location change + large distance"
    )


if len(signals) == 0:

    st.success(
        "🟢 No major risk signals detected."
    )

else:

    for signal in signals:

        st.warning(signal)


# =========================================================
# RISK SCORE
# =========================================================

risk_score = calculate_risk(row)


# =========================================================
# RISK LEVEL
# =========================================================

if risk_score >= 70:

    risk_level = "HIGH"

    action = "MANUAL REVIEW"

elif risk_score >= 40:

    risk_level = "MEDIUM"

    action = "STEP-UP VERIFICATION"

else:

    risk_level = "LOW"

    action = "ALLOW"


# =========================================================
# RISK ASSESSMENT
# =========================================================

st.divider()

st.subheader(
    "🛡️ RazorGuard Risk Assessment"
)


r1, r2, r3 = st.columns(3)


with r1:

    st.metric(
        "Risk Score",
        f"{risk_score:.2f}/100"
    )


with r2:

    st.metric(
        "Risk Level",
        risk_level
    )


with r3:

    st.metric(
        "Recommended Action",
        action
    )


# =========================================================
# RISK EXPLANATION
# =========================================================

st.subheader(
    "Risk Explanation"
)


if risk_level == "HIGH":

    st.error(
        "🚨 HIGH RISK: Multiple behavioural "
        "signals indicate that this transaction "
        "requires manual review."
    )

elif risk_level == "MEDIUM":

    st.warning(
        "⚠️ MEDIUM RISK: Additional verification "
        "is recommended before approving this transaction."
    )

else:

    st.success(
        "🟢 LOW RISK: No significant behavioural "
        "risk detected."
    )


# =========================================================
# SIGNAL SUMMARY
# =========================================================

st.subheader(
    "📊 Behavioural Indicators"
)


indicator_data = pd.DataFrame(
    {
        "Indicator": [
            "New Device",
            "Location Change",
            "Transactions / 10 min",
            "Failed Attempts",
            "Amount Deviation",
            "Distance From Previous"
        ],

        "Value": [
            "Yes"
            if row["is_new_device"] == 1
            else "No",

            "Yes"
            if row["location_change"] == 1
            else "No",

            int(
                row["transactions_last_10min"]
            ),

            int(
                row["failed_attempts"]
            ),

            f"{float(row['amount_deviation']):.2f}×",

            f"{float(row['distance_from_last_transaction']):.2f}"
        ]
    }
)


st.dataframe(
    indicator_data,
    width="stretch",
    hide_index=True
)


# =========================================================
# RAW FEATURES
# =========================================================

with st.expander(
    "View Complete Transaction Data"
):

    st.dataframe(
        pd.DataFrame(
            [row.drop(
                labels=["demo_risk_score"]
            )]
        ),
        width="stretch",
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "RazorGuard AI • Transaction Investigation Module"
)