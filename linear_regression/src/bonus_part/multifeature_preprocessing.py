import pandas as pd
from linear_regression.src.data_ingestion import load_data_from_kaggle
from linear_regression.utils.logger import get_logger

logger = get_logger("data_loader_multi", "logs/data_loader_multi.log")


def preprocess_multi() -> pd.DataFrame:
    try:
        logger.info("Starting preprocessing for Multiple Linear Regression")

        df = load_data_from_kaggle("ashydv/advertising-dataset")

        new_df = df[['TV', 'Radio', 'Newspaper', 'Sales']].copy()

        logger.info(f"Multi-feature dataset shape: {new_df.shape}")
        logger.info(f"Columns: {list(new_df.columns)}")

        return new_df

    except Exception as e:
        logger.exception(f"Error in multi preprocessing: {e}")
        raise