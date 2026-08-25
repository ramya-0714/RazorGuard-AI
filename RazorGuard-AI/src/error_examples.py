import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


print("=" * 65)
print("RAZORGUARD AI - ERROR EXAMPLES")
print("=" * 65)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/transactions_v2.csv")

print(f"\nTotal transactions: {len(df):,}")


# =========================================================
# PREPARE DATA
# =========================================================

y = df["fraud"]

# Keep transaction IDs separately
transaction_ids = df["transaction_id"]

# Remove ID and target
X = df.drop(
    columns=["transaction_id", "fraud"]
)


# =========================================================
# SAME ENCODING USED BY RAZORGUARD MODEL
# =========================================================

X = pd.get_dummies(
    X,
    columns=["payment_method"]
)


# Make sure all values are numeric
X = X.astype(float)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    transaction_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# LOAD MODEL
# =========================================================

model = xgb.XGBClassifier()

model.load_model(
    "models/razorguard_clean_model.json"
)

print("Clean XGBoost model loaded successfully!")


# =========================================================
# PREDICTIONS
# =========================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# =========================================================
# CREATE RESULTS
# =========================================================

test_results = X_test.copy()

test_results["transaction_id"] = id_test.values
test_results["actual"] = y_test.values
test_results["prediction"] = predictions
test_results["probability"] = probabilities


# =========================================================
# FIND TP / FP / FN
# =========================================================

true_positive = test_results[
    (test_results["actual"] == 1) &
    (test_results["prediction"] == 1)
]

false_positive = test_results[
    (test_results["actual"] == 0) &
    (test_results["prediction"] == 1)
]

false_negative = test_results[
    (test_results["actual"] == 1) &
    (test_results["prediction"] == 0)
]


# =========================================================
# DISPLAY EXAMPLE
# =========================================================

def show_example(title, data):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    if len(data) == 0:
        print("\nNo example found.")
        return

    example = data.iloc[0]

    print(
        f"\nTransaction ID: {example['transaction_id']}"
    )

    print(
        f"Model probability: "
        f"{float(example['probability']) * 100:.2f}%"
    )

    print(
        f"Actual label:     "
        f"{int(example['actual'])}"
    )

    print(
        f"Predicted label:  "
        f"{int(example['prediction'])}"
    )

    print("\nFeature values:")

    for feature in X.columns:
        print(
            f"  {feature}: {example[feature]}"
        )


# =========================================================
# SHOW EXAMPLES
# =========================================================

show_example(
    "TRUE POSITIVE - FRAUD CORRECTLY DETECTED",
    true_positive
)

show_example(
    "FALSE POSITIVE - LEGITIMATE FLAGGED AS FRAUD",
    false_positive
)

show_example(
    "FALSE NEGATIVE - FRAUD MISSED",
    false_negative
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()


print("\n" + "=" * 65)
print("CONFUSION MATRIX SUMMARY")
print("=" * 65)

print(f"\nTrue Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

print("\nError analysis complete!")