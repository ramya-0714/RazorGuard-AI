import streamlit as st
import pandas as pd
import os


# =========================================================
# RAZORGUARD AI - PROFESSIONAL DASHBOARD
# =========================================================

DATA_PATH = "data/transactions_spike_test.csv"
RESULT_PATH = "results/fraud_spike_analysis.csv"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RazorGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .alert-card {
        padding: 22px;
        border-radius: 14px;
        border: 2px solid #ff4b4b;
        background-color: #fff5f5;
        margin-bottom: 20px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f7f7f7;
    }

    .risk-high {
        font-size: 24px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ RazorGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Payment Risk Intelligence & Fraud Spike Detection'
    '</div>',
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
# BASIC STATISTICS
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
    analysis["suspicious_rate"].mean()
    *
    100
)


spike_count = int(
    analysis["spike_detected"].sum()
)


high_risk_spikes = analysis[
    analysis["risk_level"] == "HIGH"
].copy()


medium_risk_spikes = analysis[
    analysis["risk_level"] == "MEDIUM"
].copy()


# =========================================================
# SYSTEM OVERVIEW
# =========================================================

st.subheader(
    "System Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Actual Fraud",
        f"{total_fraud:,}"
    )


with col3:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


with col4:

    st.metric(
        "Spike Alerts",
        spike_count
    )


st.divider()


# =========================================================
# ACTIVE ALERT
# =========================================================

if len(high_risk_spikes) > 0:

    st.markdown(
        '<div class="alert-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="risk-high">'
        '🚨 ACTIVE FRAUD SPIKE DETECTED'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"RazorGuard detected "
        f"**{len(high_risk_spikes)} high-risk "
        f"time windows**."
    )

    st.write(
        "Suspicious payment activity is "
        "significantly above the normal baseline."
    )

    st.write(
        f"Normal suspicious activity baseline: "
        f"**{baseline_rate:.2f}%**"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.success(
        "🟢 No active fraud spike detected."
    )


# =========================================================
# SUSPICIOUS ACTIVITY CHART
# =========================================================

st.subheader(
    "📈 Suspicious Activity Over Time"
)


chart_data = analysis[
    [
        "time_window",
        "suspicious_rate"
    ]
].copy()


chart_data["suspicious_rate"] = (
    chart_data["suspicious_rate"]
    * 100
)


chart_data = chart_data.set_index(
    "time_window"
)


chart_data = chart_data.rename(
    columns={
        "suspicious_rate":
        "Suspicious Activity (%)"
    }
)


st.line_chart(
    chart_data
)


st.caption(
    f"Baseline suspicious activity: "
    f"{baseline_rate:.2f}%"
)


# =========================================================
# FRAUD SPIKE ALERTS
# =========================================================

st.subheader(
    "🚨 Detected Fraud Spikes"
)


if len(high_risk_spikes) == 0:

    st.info(
        "No high-risk spikes detected."
    )

else:

    for _, row in high_risk_spikes.iterrows():

        suspicious_percentage = (
            row["suspicious_rate"]
            *
            100
        )

        st.markdown(
            "---"
        )

        st.markdown(
            f"## 🚨 Hour {int(row['time_window'])}"
        )


        # -------------------------------------------------
        # ALERT METRICS
        # -------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Spike Ratio",
                f"{row['spike_ratio']:.2f}×"
            )


        with c2:

            st.metric(
                "Risk Score",
                f"{row['spike_score']:.2f}/100"
            )


        with c3:

            st.metric(
                "Suspicious Activity",
                f"{suspicious_percentage:.2f}%"
            )


        with c4:

            st.metric(
                "Fraud Rate",
                f"{row['fraud_rate']:.2f}%"
            )


        # -------------------------------------------------
        # WHY WAS IT DETECTED?
        # -------------------------------------------------

        st.markdown(
            "### 🔍 Why was this spike detected?"
        )


        reasons = []


        hour = int(
            row["time_window"]
        )


        hour_data = df[
            df["transaction_hour"]
            == hour
        ]


        if (
            hour_data["is_new_device"]
            .sum()
            > 0
        ):

            new_device_count = int(
                hour_data[
                    "is_new_device"
                ].sum()
            )

            reasons.append(
                f"📱 {new_device_count} "
                f"transactions involved new devices"
            )


        if (
            hour_data["location_change"]
            .sum()
            > 0
        ):

            location_count = int(
                hour_data[
                    "location_change"
                ].sum()
            )

            reasons.append(
                f"📍 {location_count} "
                f"transactions involved location changes"
            )


        velocity_count = int(
            (
                hour_data[
                    "transactions_last_10min"
                ]
                >= 5
            ).sum()
        )


        if velocity_count > 0:

            reasons.append(
                f"⚡ {velocity_count} "
                f"transactions showed high velocity"
            )


        failed_count = int(
            (
                hour_data[
                    "failed_attempts"
                ]
                >= 2
            ).sum()
        )


        if failed_count > 0:

            reasons.append(
                f"🔐 {failed_count} "
                f"transactions had multiple failed attempts"
            )


        deviation_count = int(
            (
                hour_data[
                    "amount_deviation"
                ]
                >= 4
            ).sum()
        )


        if deviation_count > 0:

            reasons.append(
                f"💰 {deviation_count} "
                f"transactions had high amount deviation"
            )


        distance_count = int(
            (
                hour_data[
                    "distance_from_last_transaction"
                ]
                >= 700
            ).sum()
        )


        if distance_count > 0:

            reasons.append(
                f"🌍 {distance_count} "
                f"transactions had large location distance"
            )


        if len(reasons) > 0:

            for reason in reasons:

                st.write(
                    f"• {reason}"
                )

        else:

            st.write(
                "• Multiple suspicious signals "
                "combined to create the spike."
            )


        # -------------------------------------------------
        # COMPARISON
        # -------------------------------------------------

        st.markdown(
            "### 📊 Baseline vs Current Activity"
        )


        comparison = pd.DataFrame(
            {
                "Metric": [
                    "Baseline",
                    f"Hour {hour}"
                ],

                "Suspicious Activity": [
                    f"{baseline_rate:.2f}%",
                    f"{suspicious_percentage:.2f}%"
                ]
            }
        )


        st.table(
            comparison
        )


        # -------------------------------------------------
        # ACTION
        # -------------------------------------------------

        st.warning(
            "Recommended Action: "
            f"**{row['recommended_action']}**"
        )


# =========================================================
# RISK DISTRIBUTION
# =========================================================

st.divider()

st.subheader(
    "Risk Distribution"
)


risk_counts = (
    analysis["risk_level"]
    .value_counts()
)


r1, r2, r3 = st.columns(3)


with r1:

    st.metric(
        "🔴 High Risk",
        int(
            risk_counts.get(
                "HIGH",
                0
            )
        )
    )


with r2:

    st.metric(
        "🟡 Medium Risk",
        int(
            risk_counts.get(
                "MEDIUM",
                0
            )
        )
    )


with r3:

    st.metric(
        "🟢 Normal",
        int(
            risk_counts.get(
                "NORMAL",
                0
            )
        )
    )


# =========================================================
# PROJECT EXPLANATION
# =========================================================

st.divider()

st.subheader(
    "🧠 How RazorGuard Works"
)


st.write(
    """
    RazorGuard continuously compares suspicious payment
    activity against a normal behavioural baseline.

    When suspicious activity rises significantly above
    that baseline, the system generates a fraud-spike
    alert, assigns a risk score, explains the contributing
    signals, and recommends an appropriate action.
    """
)


step1, step2, step3, step4 = st.columns(4)


with step1:

    st.markdown(
        "**1️⃣ Monitor**"
    )

    st.caption(
        "Track payment behaviour"
    )


with step2:

    st.markdown(
        "**2️⃣ Detect**"
    )

    st.caption(
        "Identify abnormal activity"
    )


with step3:

    st.markdown(
        "**3️⃣ Score**"
    )

    st.caption(
        "Calculate spike severity"
    )


with step4:

    st.markdown(
        "**4️⃣ Respond**"
    )

    st.caption(
        "Recommend merchant action"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "RazorGuard AI • AI-Powered Payment Risk Intelligence"
)