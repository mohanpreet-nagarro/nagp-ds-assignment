"""Train and save the Telco Customer Churn model.

Run from the project root:
    python train.py
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from model_utils import FeatureEngineer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "TelcoCustomerChurn.csv"
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"

TARGET = "Churn"
ID_COLUMNS = ["customerID"]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "AverageMonthlySpend", "ServiceCount"
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate(name, pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ConfusionMatrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET] + ID_COLUMNS)
    y = df[TARGET].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model_1 = build_pipeline(
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42,
        )
    )

    model_2 = build_pipeline(
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
    )

    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)

    results = [
        evaluate("Decision Tree - Baseline", model_1, X_test, y_test),
        evaluate("Decision Tree - Recall Focused", model_2, X_test, y_test),
    ]

    results_df = pd.DataFrame(results)
    print("\nModel comparison:")
    print(results_df.drop(columns=["ConfusionMatrix"]).round(4).to_string(index=False))

    final_pipeline = model_2
    final_pipeline.fit(X_train, y_train)
    joblib.dump(final_pipeline, MODEL_PATH)

    print(f"\nSaved final pipeline to: {MODEL_PATH}")
    print("\nFinal confusion matrix:")
    print(confusion_matrix(y_test, final_pipeline.predict(X_test)))

    return final_pipeline, results_df


if __name__ == "__main__":
    train()
