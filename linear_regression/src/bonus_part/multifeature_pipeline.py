import os
import pickle

from linear_regression.src.bonus_part.multifeature_preprocessing import preprocess_multi
from linear_regression.src.bonus_part.multifeature_model import MultiFeatureLinearRegression
from linear_regression.src.model_evaluation import evaluate_model
from linear_regression.utils.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger("multi_pipeline", "logs/multi_pipeline.log")


def run_multi_pipeline():
    try:
        logger.info("===== Starting Multiple Linear Regression Pipeline =====")

        # Load data
        df = preprocess_multi()

        # Split
        X = df.drop(columns=['Sales'])
        y = df['Sales']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        # Model
        model = MultiFeatureLinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = evaluate_model(y_test, y_pred)

        logger.info(f"Metrics: {metrics}")

        print("\n===== MULTI FEATURE LINEAR REGRESSION =====")
        print(metrics)

        print("\nModel Parameters:")
        intercept, coeffs = model.get_params()
        print("Intercept:", intercept)
        print("Coefficients:", coeffs)

        # Save model
        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/multi_model.pkl", "wb") as f:
            pickle.dump(model, f)

        logger.info("Model saved successfully")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise