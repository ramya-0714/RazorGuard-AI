import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

DATA_PATH = Path("data/transactions.csv")

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("RAZORGUARD AI - FRAUD DETECTION MODEL")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Fraud rate: {df['fraud'].mean() * 100:.2f}%")


# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

X = df.drop(columns=["fraud", "transaction_id"])
y = df["fraud"]


# -----------------------------------------
# 3. Identify feature types
# -----------------------------------------

categorical_features = [
    "payment_method"
]

numeric_features = [
    column for column in X.columns
    if column not in categorical_features
]


# -----------------------------------------
# 4. Preprocessing
# -----------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# -----------------------------------------
# 5. Train/Test split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# -----------------------------------------
# 6. Models
# -----------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


results = {}


# -----------------------------------------
# 7. Train and evaluate
# -----------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

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

    results[model_name] = {
        "pipeline": pipeline,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "false_positive_rate": false_positive_rate
    }

    print(f"\nPrecision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"False Positive Rate:{false_positive_rate:.4f}")

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


# -----------------------------------------
# 8. Select best model
# -----------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_pipeline = results[best_model_name]["pipeline"]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"\nSelected model: {best_model_name}")
print(
    f"F1 Score: "
    f"{results[best_model_name]['f1']:.4f}"
)


# -----------------------------------------
# 9. Save model
# -----------------------------------------

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "razorguard_fraud_model.joblib"

joblib.dump(
    best_pipeline,
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")

print("\nTraining complete!")