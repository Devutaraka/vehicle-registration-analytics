import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


PROJECT_DIR = Path("/mnt/d/BDE 2/BDE")
DATA_PATH = PROJECT_DIR / "vehicle_data.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "ml_prediction"


def create_category(value):
    if value <= 25:
        return "Low"
    elif value <= 176:
        return "Medium"
    else:
        return "High"


def main():

    print("=" * 60)
    print("ML PREDICTION ANALYSIS")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"\nTotal records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Basic cleaning
    df["state"] = df["state"].astype(str).str.strip()
    df["rto"] = df["rto"].astype(str).str.strip()
    df["maker"] = df["maker"].astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    df = df.dropna(
        subset=["state", "rto", "maker", "count"]
    )

    # Create prediction target
    df["registration_category"] = df["count"].apply(create_category)

    print("\n=== CLASS DISTRIBUTION ===")
    print(df["registration_category"].value_counts())

    # Features and target
    X = df[["state", "rto", "maker"]]
    y = df["registration_category"]

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records : {len(X_test):,}")

    # Categorical preprocessing
    categorical_features = ["state", "rto", "maker"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    # Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    print("\nTraining ML model...")
    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n=== ML MODEL RESULTS ===")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    # Classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Confusion matrix
    print("=== CONFUSION MATRIX ===")

    labels = ["Low", "Medium", "High"]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("             Predicted")
    print("             Low  Medium  High")

    for label, row in zip(labels, cm):
        print(
            f"{label:<10} "
            f"{row[0]:>5} "
            f"{row[1]:>7} "
            f"{row[2]:>6}"
        )

    # Save predictions
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = X_test.copy()
    results["actual_category"] = y_test.values
    results["predicted_category"] = y_pred

    results.to_csv(
        OUTPUT_DIR / "ml_predictions.csv",
        index=False
    )

    # Save metrics
    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    metrics.to_csv(
        OUTPUT_DIR / "ml_metrics.csv",
        index=False
    )

    print(
        f"\nPredictions saved to: "
        f"{OUTPUT_DIR / 'ml_predictions.csv'}"
    )

    print(
        f"Metrics saved to: "
        f"{OUTPUT_DIR / 'ml_metrics.csv'}"
    )


if __name__ == "__main__":
    main()
