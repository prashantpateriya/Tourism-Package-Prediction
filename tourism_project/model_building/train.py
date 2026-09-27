
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import os

# Define paths for data and model output
X_TRAIN_PATH = "Xtrain.csv"
X_TEST_PATH = "Xtest.csv"
y_TRAIN_PATH = "ytrain.csv"
y_TEST_PATH = "ytest.csv"

MODEL_OUTPUT_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "xgboost_model.pkl")

def train_model():
    # Load data
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(y_TRAIN_PATH).squeeze() # .squeeze() to convert DataFrame to Series
    y_test = pd.read_csv(y_TEST_PATH).squeeze()

    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing pipeline for categorical and numerical features
    # One-hot encode categorical features, scale numerical features (optional for XGBoost but good practice)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create XGBoost Classifier model
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Create a pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1]
    }

    # Setup MLflow
    mlflow.set_experiment("Tourism Package Prediction")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("stratify_target", True)

        # Perform GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nBest Parameters:", grid_search.best_params_)
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save the best model
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        print(f"Best model saved to {MODEL_PATH}")

        # Log model with MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="xgboost_model",
            registered_model_name="XGBoostTourismClassifier"
        )

if __name__ == "__main__":
    train_model()
