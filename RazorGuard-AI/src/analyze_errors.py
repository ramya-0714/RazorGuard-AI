import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


print("=" * 60)
print("RAZORGUARD AI - ERROR ANALYSIS")
print("=" * 60)


# Load our V2 dataset
df = pd.read_csv("data/transactions_v2.csv")


# Prepare the data
X = df.drop(
    columns=[
        "fraud",
        "transaction_id"
    ]
)

y = df["fraud"]


# Convert payment method to numbers
X = pd.get_dummies(
    X,
    columns=["payment_method"],
    dtype=int
)


# Split the data exactly like before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Load our saved XGBoost model
model = XGBClassifier()

model.load_model(
    "models/razorguard_xgboost_v2.json"
)


print("\nModel loaded successfully!")


# Make predictions
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nCorrect legitimate transactions : {tn}")
print(f"Legitimate wrongly flagged     : {fp}")
print(f"Fraud that we missed           : {fn}")
print(f"Fraud correctly detected       : {tp}")


# Create results table
results = X_test.copy()

results["actual_fraud"] = y_test.values
results["predicted_fraud"] = predictions
results["fraud_probability"] = probabilities


# Find fraud that the model missed
missed_fraud = results[
    (results["actual_fraud"] == 1) &
    (results["predicted_fraud"] == 0)
]


print("\n" + "=" * 60)
print("MISSED FRAUD")
print("=" * 60)

print(
    f"\nNumber of fraud transactions missed: "
    f"{len(missed_fraud)}"
)


# Show the characteristics of missed fraud
if len(missed_fraud) > 0:

    print("\nAverage characteristics of missed fraud:")

    print(
        missed_fraud[
            [
                "amount",
                "is_new_device",
                "location_change",
                "transactions_last_10min",
                "failed_attempts",
                "amount_deviation",
                "distance_from_last_transaction"
            ]
        ].mean()
    )


print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)