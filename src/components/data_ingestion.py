import os
import shutil
import logging
from pathlib import Path
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants import TRAIN_DIR, VALIDATION_DIR

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiate data ingestion process
        Returns DataIngestionArtifact with paths to training and validation data
        """
        try:
            logger.info("Starting data ingestion")
            
            # Create root directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Define paths
            train_data_path = os.path.join(self.config.root_dir, TRAIN_DIR)
            validation_data_path = os.path.join(self.config.root_dir, VALIDATION_DIR)
            
            # Create directories if they don't exist
            os.makedirs(train_data_path, exist_ok=True)
            os.makedirs(validation_data_path, exist_ok=True)
            
            logger.info(f"Train data path: {train_data_path}")
            logger.info(f"Validation data path: {validation_data_path}")
            
            # If unzip_data_dir exists, copy data from there (optional)
            if os.path.exists(self.config.unzip_dir):
                logger.info(f"Copying data from {self.config.unzip_dir}")
                
                # Copy train data
                source_train = os.path.join(self.config.unzip_dir, "train")
                if os.path.exists(source_train):
                    for emotion_folder in os.listdir(source_train):
                        src = os.path.join(source_train, emotion_folder)
                        dst = os.path.join(train_data_path, emotion_folder)
                        if os.path.isdir(src):
                            os.makedirs(dst, exist_ok=True)
                            for file in os.listdir(src):
                                shutil.copy2(os.path.join(src, file), dst)
                
                # Copy validation data
                source_val = os.path.join(self.config.unzip_dir, "validation")
                if os.path.exists(source_val):
                    for emotion_folder in os.listdir(source_val):
                        src = os.path.join(source_val, emotion_folder)
                        dst = os.path.join(validation_data_path, emotion_folder)
                        if os.path.isdir(src):
                            os.makedirs(dst, exist_ok=True)
                            for file in os.listdir(src):
                                shutil.copy2(os.path.join(src, file), dst)
            
            # Verify data exists
            logger.info("Data ingestion completed successfully")
            
            # Return artifact
            artifact = DataIngestionArtifact(
                trained_file_path=Path(train_data_path),
                test_file_path=Path(validation_data_path)
            )
            
            logger.info(f"Data Ingestion Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise e
