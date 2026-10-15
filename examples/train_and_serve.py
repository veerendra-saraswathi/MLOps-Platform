"""Example: train a tiny model, log to MLflow, and prepare for KServe serving."""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("iris-demo")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Model logged. You can now register it via the control plane API.")


if __name__ == "__main__":
    main()