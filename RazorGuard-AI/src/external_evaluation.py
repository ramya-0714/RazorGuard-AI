import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier


print("=" * 65)
print("RAZORGUARD AI - EXTERNAL DATASET EVALUATION")
print("=" * 65)


# =========================================================
# LOAD KAGGLE DATASET
# =========================================================

print("\nLoading Kaggle credit-card dataset...")

df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")


# =========================================================
# DATASET INFORMATION
# =========================================================

print("\n" + "=" * 65)
print("DATASET INFORMATION")
print("=" * 65)

total = len(df)
fraud = int(df["Class"].sum())
legitimate = total - fraud

print(f"\nTotal transactions : {total:,}")
print(f"Legitimate         : {legitimate:,}")
print(f"Fraud              : {fraud:,}")
print(f"Fraud rate         : {(fraud / total) * 100:.4f}%")


# =========================================================
# FEATURES / TARGET
# =========================================================

X = df.drop(columns=["Class"])
y = df["Class"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 65)
print("TRAIN / TEST SPLIT")
print("=" * 65)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing samples  : {len(X_test):,}")


# =========================================================
# SCALE FEATURES
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================================
# CLASS IMBALANCE
# =========================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

class_weight = negative / positive

print(
    f"\nClass weight ratio: {class_weight:.2f}"
)


# =========================================================
# TRAIN XGBOOST
# =========================================================

print("\nTraining XGBoost on external dataset...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training complete!")


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
# METRICS
# =========================================================

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()

fpr = fp / (fp + tn)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 65)
print("EXTERNAL DATASET RESULTS")
print("=" * 65)

print(f"\nPrecision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"ROC-AUC             : {roc_auc:.4f}")
print(f"PR-AUC              : {pr_auc:.4f}")
print(f"False Positive Rate : {fpr:.4f}")


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(f"\nTrue Negatives : {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives : {tp:,}")


# =========================================================
# SAVE RESULTS
# =========================================================

results = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "False Positive Rate"
    ],
    "Value": [
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        fpr
    ]
})

results.to_csv(
    "results/external_dataset_results.csv",
    index=False
)

print(
    "\nResults saved to:"
)

print(
    "results/external_dataset_results.csv"
)


print("\n" + "=" * 65)
print("EXTERNAL DATASET EVALUATION COMPLETE!")
print("=" * 65)