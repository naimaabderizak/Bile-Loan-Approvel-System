from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_approval.csv"
MODEL_PATH = ROOT / "models" / "loan_approval_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.csv"

TARGET = "Loan_Status"
ID_COLUMN = "Loan_ID"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.drop_duplicates()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities) if probabilities is not None else None,
    }
    return result


def main() -> None:
    df = load_data()
    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].map({"N": 0, "Y": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            max_depth=6,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    trained_models = {}
    metrics = []
    for name, classifier in candidates.items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics.append(evaluate_model(name, model, X_test, y_test))

    metrics_df = pd.DataFrame(metrics).sort_values(
        by=["f1", "roc_auc", "accuracy"],
        ascending=False,
        na_position="last",
    )
    best_name = metrics_df.iloc[0]["model"]

    artifact = {
        "model": trained_models[best_name],
        "model_name": best_name,
        "features": X.columns.tolist(),
        "metrics": metrics_df,
        "target_mapping": {"N": 0, "Y": 1},
        "inverse_target_mapping": {0: "Not Approved", 1: "Approved"},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print(f"Saved best model: {best_name}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
