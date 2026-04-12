from sklearn.linear_model import LinearRegression
import numpy as np
from linear_regression.utils.logger import get_logger

logger = get_logger("multi_model", "logs/multi_model.log")


class MultiFeatureLinearRegression:
    def __init__(self):
        self.model = LinearRegression()
        logger.info("Initialized MultiFeature LinearRegression")

    def fit(self, X, y):
        try:
            X = np.array(X)
            y = np.array(y)

            self.model.fit(X, y)

            logger.info("Training completed")

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            raise

    def predict(self, X):
        X = np.array(X)
        return self.model.predict(X)

    def get_params(self):
        return self.model.intercept_, self.model.coef_