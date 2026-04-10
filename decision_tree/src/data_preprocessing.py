import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from ..utils.logger import get_logger

logger = get_logger("preprocessor", "logs/preprocessor.log")


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_columns = None
        self.categorical_columns = None

    # -----------------------------
    # 1. Missing Value Handling
    # -----------------------------
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values")

        df = df.copy()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        return df

    # -----------------------------
    # 2. Split features/target
    # -----------------------------
    def split_features_target(self, df: pd.DataFrame, target: str):
        logger.info(f"Splitting features and target: {target}")

        X = df.drop(columns=[target])
        y = df[target]

        return X, y

    # -----------------------------
    # 3. Fit Encoders (TRAIN ONLY)
    # -----------------------------
    def fit_transform_features(self, X: pd.DataFrame):
        logger.info("Fitting encoders and scaler")

        X = X.copy()

        self.numeric_columns = X.select_dtypes(include=np.number).columns
        self.categorical_columns = X.select_dtypes(exclude=np.number).columns

        # ---- Encode categorical ----
        for col in self.categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # ---- Scale numeric ----
        X[self.numeric_columns] = self.scaler.fit_transform(X[self.numeric_columns])

        return X

    # -----------------------------
    # 4. Transform ONLY (TEST DATA)
    # -----------------------------
    def transform_features(self, X: pd.DataFrame):
        logger.info("Transforming new data")

        X = X.copy()

        # ---- categorical ----
        for col in self.categorical_columns:
            le = self.label_encoders[col]
            X[col] = le.transform(X[col])

        # ---- numeric ----
        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        return X

    # -----------------------------
    # 5. Full pipeline
    # -----------------------------
    def preprocess(self, df: pd.DataFrame, target: str):
        df = self.handle_missing_values(df)

        X, y = self.split_features_target(df, target)

        X = self.fit_transform_features(X)

        return X, y