import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/transactions.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("RAZORGUARD AI - XGBOOST FRAUD DETECTION")
print("=" * 65)

print(f"\nDataset shape: {df.shape}")
print(f"Fraud rate: {df['fraud'].mean() * 100:.2f}%")


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

# Remove identifiers that should not influence fraud prediction
X = df.drop(columns=["fraud", "transaction_id"])

y = df["fraud"]


# Convert payment method to numeric columns
X = pd.get_dummies(
    X,
    columns=["payment_method"],
    dtype=int
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# 4. CALCULATE CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"\nClass weight ratio: {scale_pos_weight:.2f}")


# ============================================================
# 5. BUILD XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. TRAIN
# ============================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# ============================================================
# 7. PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 8. METRICS
# ============================================================

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

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()

false_positive_rate = fp / (fp + tn)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("XGBOOST RESULTS")
print("=" * 65)

print(f"\nPrecision:           {precision:.4f}")
print(f"Recall:              {recall:.4f}")
print(f"F1 Score:            {f1:.4f}")
print(f"ROC-AUC:             {roc_auc:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 65)
print("TOP RISK SIGNALS")
print("=" * 65)

print(
    feature_importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "razorguard_xgboost_model.json"

model.save_model(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

print("\nXGBoost experiment complete!")