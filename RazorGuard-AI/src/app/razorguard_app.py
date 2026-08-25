import streamlit as st
import pandas as pd
import os


# =========================================================
# RAZORGUARD AI - FINAL DASHBOARD
# =========================================================

DATA_PATH = "data/transactions_spike_test.csv"
RESULT_PATH = "results/fraud_spike_analysis.csv"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RazorGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
    }

    .threat-box {
        padding: 22px;
        border-radius: 14px;
        border: 2px solid #ff4b4b;
        background-color: #fff5f5;
        margin: 20px 0;
    }

    .threat-title {
        font-size: 25px;
        font-weight: 800;
    }

    .risk-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
    }

    .pipeline-card {
        text-align: center;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

if not os.path.exists(DATA_PATH):

    st.error(
        "Transaction dataset not found."
    )

    st.stop()


if not os.path.exists(RESULT_PATH):

    st.error(
        "Fraud spike analysis results not found."
    )

    st.stop()


df = pd.read_csv(DATA_PATH)

analysis = pd.read_csv(
    RESULT_PATH
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "# 🛡️ RazorGuard AI"
)

st.sidebar.caption(
    "Payment Risk Intelligence"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🚨 Fraud Spike Detection",
        "🔎 Transaction Investigation"
    ]
)


st.sidebar.divider()

st.sidebar.caption(
    "AI-Powered Payment Risk Intelligence"
)

st.sidebar.caption(
    "Fraud Detection Prototype"
)


# =========================================================
# COMMON STATISTICS
# =========================================================

total_transactions = len(df)

total_fraud = int(
    df["fraud"].sum()
)

fraud_rate = (
    total_fraud
    /
    total_transactions
    *
    100
)

baseline_rate = (
    analysis["suspicious_rate"]
    .mean()
    *
    100
)

high_spikes = analysis[
    analysis["risk_level"] == "HIGH"
].copy()

medium_spikes = analysis[
    analysis["risk_level"] == "MEDIUM"
].copy()


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">🛡️ RazorGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Payment Risk Intelligence & '
        'Fraud Spike Detection'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # SYSTEM METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Transactions",
            f"{total_transactions:,}"
        )


    with c2:

        st.metric(
            "Actual Fraud",
            f"{total_fraud:,}"
        )


    with c3:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )


    with c4:

        st.metric(
            "High-Risk Spikes",
            len(high_spikes)
        )


    # -----------------------------------------------------
    # THREAT STATUS
    # -----------------------------------------------------

    if len(high_spikes) > 0:

        st.markdown(
            """
            <div class="threat-box">

            <div class="threat-title">
            🚨 ACTIVE FRAUD THREAT DETECTED
            </div>

            <p>
            RazorGuard has detected significant increases
            in suspicious payment activity.
            Immediate investigation is recommended.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.success(
            "🟢 No active high-risk fraud spikes detected."
        )


    # -----------------------------------------------------
    # ACTIVE SPIKES
    # -----------------------------------------------------

    st.markdown(
        "## 🚨 Active Fraud Spikes"
    )


    if len(high_spikes) == 0:

        st.info(
            "No high-risk fraud spikes detected."
        )

    else:

        for _, row in high_spikes.iterrows():

            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    f"Hour {int(row['time_window'])}",
                    f"{row['spike_ratio']:.2f}×",
                    "Spike Ratio"
                )


            with col2:

                st.metric(
                    "Risk Score",
                    f"{row['spike_score']:.2f}/100"
                )


            with col3:

                st.metric(
                    "Fraud Rate",
                    f"{row['fraud_rate']:.2f}%"
                )


            with col4:

                st.metric(
                    "Actual Fraud",
                    int(row["actual_fraud"])
                )


            st.error(
                "Recommended Action: "
                f"**{row['recommended_action']}**"
            )


    # -----------------------------------------------------
    # ACTIVITY CHART
    # -----------------------------------------------------

    st.markdown(
        "## 📈 Suspicious Activity Over Time"
    )


    chart = analysis[
        [
            "time_window",
            "suspicious_rate"
        ]
    ].copy()


    chart["suspicious_rate"] = (
        chart["suspicious_rate"]
        * 100
    )


    chart = chart.set_index(
        "time_window"
    )


    chart = chart.rename(
        columns={
            "suspicious_rate":
            "Suspicious Activity (%)"
        }
    )


    st.line_chart(
        chart
    )


    st.caption(
        f"Normal suspicious activity baseline: "
        f"{baseline_rate:.2f}%"
    )


    # -----------------------------------------------------
    # HOW IT WORKS
    # -----------------------------------------------------

    st.markdown(
        "## 🧠 How RazorGuard Works"
    )


    st.write(
        """
        RazorGuard combines behavioural risk analysis
        with fraud-spike detection to identify unusual
        payment activity and support faster investigation.
        """
    )


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.markdown(
            "### 1️⃣ Monitor"
        )

        st.caption(
            "Track payment behaviour"
        )


    with p2:

        st.markdown(
            "### 2️⃣ Detect"
        )

        st.caption(
            "Identify abnormal activity"
        )


    with p3:

        st.markdown(
            "### 3️⃣ Investigate"
        )

        st.caption(
            "Explain transaction risk"
        )


    with p4:

        st.markdown(
            "### 4️⃣ Respond"
        )

        st.caption(
            "Recommend risk action"
        )


# =========================================================
# FRAUD SPIKE DETECTION PAGE
# =========================================================

elif page == "🚨 Fraud Spike Detection":

    st.markdown(
        '<div class="main-title">'
        '🚨 Fraud Spike Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Identify unusual increases in suspicious '
        'payment activity.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Baseline",
            f"{baseline_rate:.2f}%"
        )


    with c2:

        st.metric(
            "Spike Windows",
            int(
                analysis["spike_detected"].sum()
            )
        )


    with c3:

        st.metric(
            "High Risk",
            len(high_spikes)
        )


    with c4:

        st.metric(
            "Medium Risk",
            len(medium_spikes)
        )


    st.divider()


    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    st.subheader(
        "Suspicious Activity Over Time"
    )


    chart = analysis[
        [
            "time_window",
            "suspicious_rate"
        ]
    ].copy()


    chart["suspicious_rate"] *= 100


    chart = chart.set_index(
        "time_window"
    )


    chart = chart.rename(
        columns={
            "suspicious_rate":
            "Suspicious Activity (%)"
        }
    )


    st.line_chart(
        chart
    )


    # -----------------------------------------------------
    # SPIKE TABLE
    # -----------------------------------------------------

    st.subheader(
        "Detected Fraud Spikes"
    )


    if len(high_spikes) == 0:

        st.success(
            "No high-risk fraud spikes detected."
        )

    else:

        display_data = high_spikes[
            [
                "time_window",
                "transactions",
                "suspicious_rate",
                "actual_fraud",
                "fraud_rate",
                "spike_ratio",
                "spike_score",
                "risk_level",
                "recommended_action"
            ]
        ].copy()


        display_data[
            "suspicious_rate"
        ] *= 100


        display_data = display_data.rename(
            columns={
                "time_window": "Hour",
                "transactions": "Transactions",
                "suspicious_rate":
                    "Suspicious Rate (%)",
                "actual_fraud":
                    "Actual Fraud",
                "fraud_rate":
                    "Fraud Rate (%)",
                "spike_ratio":
                    "Spike Ratio",
                "spike_score":
                    "Risk Score",
                "risk_level":
                    "Risk Level",
                "recommended_action":
                    "Recommended Action"
            }
        )


        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True
        )


    # -----------------------------------------------------
    # RISK INTELLIGENCE
    # -----------------------------------------------------

    st.subheader(
        "🔍 Risk Intelligence"
    )


    for _, row in high_spikes.iterrows():

        hour = int(
            row["time_window"]
        )


        st.markdown(
            f"### 🚨 Hour {hour}"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Transactions",
                int(row["transactions"])
            )


        with c2:

            st.metric(
                "Suspicious Activity",
                f"{row['suspicious_rate'] * 100:.2f}%"
            )


        with c3:

            st.metric(
                "Actual Fraud",
                int(row["actual_fraud"])
            )


        with c4:

            st.metric(
                "Fraud Rate",
                f"{row['fraud_rate']:.2f}%"
            )


        st.write(
            f"**Spike Ratio:** "
            f"{row['spike_ratio']:.2f}×"
        )


        st.write(
            f"**Risk Score:** "
            f"{row['spike_score']:.2f}/100"
        )


        st.error(
            f"**Recommended Action:** "
            f"{row['recommended_action']}"
        )


        st.divider()


# =========================================================
# TRANSACTION INVESTIGATION PAGE
# =========================================================

else:

    st.markdown(
        '<div class="main-title">'
        '🔎 Transaction Investigation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Investigate individual transactions and '
        'identify behavioural risk signals.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # RISK FUNCTION
    # -----------------------------------------------------

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


        return min(
            score,
            100
        )


    # -----------------------------------------------------
    # CALCULATE SCORES
    # -----------------------------------------------------

    df["risk_score"] = df.apply(
        calculate_risk,
        axis=1
    )


    high_risk_transactions = df[
        df["risk_score"] >= 70
    ].sort_values(
        "risk_score",
        ascending=False
    )


    transaction_ids = (
        df["transaction_id"]
        .astype(str)
        .tolist()
    )


    if len(high_risk_transactions) > 0:

        default_id = str(
            high_risk_transactions.iloc[0][
                "transaction_id"
            ]
        )

    else:

        default_id = transaction_ids[0]


    default_index = transaction_ids.index(
        default_id
    )


    selected_id = st.selectbox(
        "Select Transaction",
        transaction_ids,
        index=default_index
    )


    row = df[
        df["transaction_id"].astype(str)
        == selected_id
    ].iloc[0]


    # -----------------------------------------------------
    # TRANSACTION DETAILS
    # -----------------------------------------------------

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

        label = (
            "Fraud"
            if int(row["fraud"]) == 1
            else "Legitimate"
        )

        st.metric(
            "Dataset Label",
            label
        )


    # -----------------------------------------------------
    # RISK SIGNALS
    # -----------------------------------------------------

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
            "💰 Unusually high transaction amount"
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

            st.warning(
                signal
            )


    # -----------------------------------------------------
    # RISK ASSESSMENT
    # -----------------------------------------------------

    risk_score = calculate_risk(
        row
    )


    if risk_score >= 70:

        risk_level = "HIGH"

        action = "MANUAL REVIEW"

    elif risk_score >= 40:

        risk_level = "MEDIUM"

        action = "STEP-UP VERIFICATION"

    else:

        risk_level = "LOW"

        action = "ALLOW"


    st.divider()

    st.subheader(
        "🛡️ RazorGuard Risk Assessment"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}/100"
        )


    with c2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with c3:

        st.metric(
            "Recommended Action",
            action
        )


    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    if risk_level == "HIGH":

        st.error(
            "🚨 HIGH RISK: Multiple behavioural "
            "signals indicate that this transaction "
            "requires manual review."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠️ MEDIUM RISK: Additional verification "
            "is recommended before approval."
        )

    else:

        st.success(
            "🟢 LOW RISK: No significant behavioural "
            "risk detected."
        )


    # -----------------------------------------------------
    # BEHAVIOURAL INDICATORS
    # -----------------------------------------------------

    st.subheader(
        "📊 Behavioural Indicators"
    )


    indicators = pd.DataFrame(
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
        indicators,
        width="stretch",
        hide_index=True
    )


    # -----------------------------------------------------
    # COMPLETE DATA
    # -----------------------------------------------------

    with st.expander(
        "View Complete Transaction Data"
    ):

        clean_row = row.drop(
            labels=["risk_score"]
        )

        st.dataframe(
            pd.DataFrame(
                [clean_row]
            ),
            width="stretch",
            hide_index=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "RazorGuard AI • AI-Powered Payment Risk Intelligence "
    "• Fraud Detection Prototype"
)