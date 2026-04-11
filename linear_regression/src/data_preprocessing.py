import pandas as pd
from linear_regression.src.data_ingestion import load_data_from_kaggle
from sklearn.preprocessing import LabelEncoder
from linear_regression.utils.logger import get_logger

logger = get_logger("data_loader", "logs/data_loader.log")


def preprocess() -> pd.DataFrame:
    try:
        logger.info("Starting preprocessing")

        # Load data
        df = load_data_from_kaggle("ashydv/advertising-dataset")
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Select features
        select_features = df[['TV', 'Sales']]
        logger.info(f"Selected features: {list(select_features.columns)}")

        new_df = select_features.copy()
        logger.info(f"New data shape: {new_df.shape}")

        return new_df

    except Exception as e:
        logger.exception(f"Error during preprocessing: {e}")
        raise
