from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

DATA_DIR = BASE_DIR / "data" / "processed" / "split_cleaned"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

# --------------------------------------------------
# 2. Load training and testing data
# --------------------------------------------------

print("Loading training and testing datasets...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

excluded_columns = {"Target", "Original_Label"}

if not excluded_columns.issubset(train_df.columns):
    raise ValueError("Required label columns are missing.")

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

missing_columns = [
    col for col in feature_columns
    if col not in test_df.columns
]

if missing_columns:
    raise ValueError(
        f"Test data is missing features: {missing_columns}"
    )

X_train = train_df[feature_columns]
y_train = train_df["Target"].astype(str)

X_test = test_df[feature_columns]
y_test = test_df["Target"].astype(str)

print("Training records:", f"{len(X_train):,}")
print("Testing records:", f"{len(X_test):,}")
print("Input features:", len(feature_columns))

# --------------------------------------------------
# 3. Create machine learning pipelines
# --------------------------------------------------

def build_model(name):
    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if name == "Decision Tree":
        classifier = DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        )

    elif name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

    elif name == "Logistic Regression":
        steps.append(("scaler", StandardScaler()))

        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    steps.append(("classifier", classifier))

    return Pipeline(steps)


models = [
    "Decision Tree",
    "Random Forest",
    "Logistic Regression"
]

# --------------------------------------------------
# 4. Create a validation split from training data
# --------------------------------------------------

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    stratify=y_train,
    random_state=42
)

# --------------------------------------------------
# 5. Train and compare candidate models
# --------------------------------------------------

validation_results = {}

print("\nStarting model comparison...")

for name in models:
    print("\n" + "=" * 55)
    print("Training:", name)

    model = build_model(name)

    start = time.time()
    model.fit(X_fit, y_fit)
    elapsed = time.time() - start

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val, predictions,
        pos_label="Attack",
        zero_division=0
    )

    recall = recall_score(
        y_val, predictions,
        pos_label="Attack",
        zero_division=0
    )

    f1 = f1_score(
        y_val, predictions,
        pos_label="Attack",
        zero_division=0
    )

    validation_results[name] = {
        "Accuracy": accuracy,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": f1,
        "Training_Seconds": round(elapsed, 2)
    }

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Attack Precision:{precision:.4f}")
    print(f"Attack Recall:   {recall:.4f}")
    print(f"Attack F1-score: {f1:.4f}")
    print(f"Training seconds:{elapsed:.2f}")

# --------------------------------------------------
# 6. Save validation comparison
# --------------------------------------------------

comparison = pd.DataFrame.from_dict(
    validation_results,
    orient="index"
)

comparison.index.name = "Model"
comparison = comparison.sort_values(
    by=["Attack_F1", "Attack_Recall"],
    ascending=False
)

comparison.to_csv(
    REPORT_DIR / "validation_model_comparison.csv"
)

print("\n" + "=" * 65)
print("VALIDATION MODEL COMPARISON")
print("=" * 65)
print(comparison.round(4).to_string())

# Select using validation results, not test results
best_model_name = comparison.index[0]

print("\nSelected model:", best_model_name)
print("Selection criterion: Attack F1-score, then Attack Recall")

# --------------------------------------------------
# 7. Retrain the selected model on all training data
# --------------------------------------------------

print("\nRetraining selected model on full training data...")

best_model = build_model(best_model_name)
best_model.fit(X_train, y_train)

# --------------------------------------------------
# 8. Evaluate exactly once on the held-out test set
# --------------------------------------------------

print("\nEvaluating selected model on held-out test data...")

test_predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)

precision = precision_score(
    y_test, test_predictions,
    pos_label="Attack",
    zero_division=0
)

recall = recall_score(
    y_test, test_predictions,
    pos_label="Attack",
    zero_division=0
)

f1 = f1_score(
    y_test, test_predictions,
    pos_label="Attack",
    zero_division=0
)

report = classification_report(
    y_test,
    test_predictions,
    labels=["Normal", "Attack"],
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=["Normal", "Attack"]
)

print("\nFINAL TEST RESULTS")
print("=" * 55)
print("Selected model:", best_model_name)
print(f"Accuracy:         {accuracy:.4f}")
print(f"Attack Precision: {precision:.4f}")
print(f"Attack Recall:    {recall:.4f}")
print(f"Attack F1-score:  {f1:.4f}")

print("\nClassification report:")
print(report)

print("Confusion matrix:")
print(pd.DataFrame(
    matrix,
    index=["Actual Normal", "Actual Attack"],
    columns=["Predicted Normal", "Predicted Attack"]
))

# --------------------------------------------------
# 9. Save model, features, and evaluation reports
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_DIR / "best_traffic_model.joblib"
)

with open(
    MODEL_DIR / "feature_columns.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(feature_columns, file, indent=2)

with open(
    REPORT_DIR / "final_test_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Selected model: {best_model_name}\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Attack precision: {precision:.4f}\n")
    file.write(f"Attack recall: {recall:.4f}\n")
    file.write(f"Attack F1-score: {f1:.4f}\n\n")
    file.write("Classification report:\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(matrix))

print("\nSaved model:", MODEL_DIR / "best_traffic_model.joblib")
print("Saved features:", MODEL_DIR / "feature_columns.json")
print("Saved comparison:", REPORT_DIR / "validation_model_comparison.csv")
print("Saved test report:", REPORT_DIR / "final_test_report.txt")

print("\nMODEL TRAINING AND EVALUATION COMPLETED")