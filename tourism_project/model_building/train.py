import os
import pandas as pd
import joblib
import xgboost as xgb
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

os.environ["MLFLOW_SKLEARN_SUPPORT_UNSAFE_DESERIALIZATION"] = "true"

def train_model():
    print("Loading split datasets...")
    # Load the train and test data from the workflow artifact
    train_path = "tourism_project/data/train.csv"
    test_path = "tourism_project/data/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "ProdTaken"
    X_train = train_df.drop(columns=[target_col], errors="ignore")
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col], errors="ignore")
    y_test = test_df[target_col]

    # One-hot encode string columns if any exist in the mock/loaded data
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols)
        X_test = pd.get_dummies(X_test, columns=cat_cols)
        # Align train and test features
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Define MLflow Experiment
    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run():
        print("Tuning XGBoost hyperparameters with GridSearchCV...")
        # Define model
        base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'n_estimators': [50, 100]
        }

        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Log best parameters
        print(f"Best Parameters: {best_params}")
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Evaluate
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Save the best model
        model_dir = "tourism_project/deployment"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

        # Log model artifact to MLflow safely with skops_trusted_types defined
        mlflow.sklearn.log_model(
            best_model,
            "model",
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
        )

if __name__ == '__main__':
    train_model()
