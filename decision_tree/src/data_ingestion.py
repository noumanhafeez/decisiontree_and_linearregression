import os
import requests
import pandas as pd
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger("data_loader", "logs/data_loader.log")


class DataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, filename: str) -> str:
        """
        Download file from URL and store locally.
        """
        file_path = self.data_dir / filename

        if file_path.exists():
            logger.info(f"File already exists: {file_path}. Skipping download.")
            return str(file_path)

        try:
            logger.info(f"Downloading data from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Saved file to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV into pandas DataFrame.
        """
        try:
            logger.info(f"Loading CSV from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def load_from_url(self, url: str, filename: str) -> pd.DataFrame:
        """
        Full pipeline: download + load
        """
        local_path = self.download_file(url, filename)
        return self.load_csv(local_path)