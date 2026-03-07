import os
import logging
import json
from pathlib import Path
from typing import Dict, List
import cv2
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.constants import EMOTION_LABELS

logger = logging.getLogger(__name__)


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_data_files(self) -> bool:
        """Validate if all required data files and directories exist"""
        try:
            logger.info("Validating data files and directories")
            
            # Check if data directories exist
            if not os.path.exists(self.config.unzip_data_dir):
                logger.error(f"Data directory not found: {self.config.unzip_data_dir}")
                return False
            
            # Check for train and validation directories
            train_dir = os.path.join(self.config.unzip_data_dir, "train")
            val_dir = os.path.join(self.config.unzip_data_dir, "validation")
            
            if not os.path.exists(train_dir):
                logger.error(f"Train directory not found: {train_dir}")
                return False
            
            if not os.path.exists(val_dir):
                logger.error(f"Validation directory not found: {val_dir}")
                return False
            
            logger.info("Data directories validated successfully")
            return True
        except Exception as e:
            logger.error(f"Error validating data files: {str(e)}")
            return False

    def validate_schema(self) -> bool:
        """Validate if all required emotion directories exist in train and validation"""
        try:
            logger.info("Validating data schema")
            
            train_dir = os.path.join(self.config.unzip_data_dir, "train")
            val_dir = os.path.join(self.config.unzip_data_dir, "validation")
            
            # Check if all emotion folders exist in train
            train_emotions = set(os.listdir(train_dir))
            required_emotions = set(EMOTION_LABELS)
            
            missing_train = required_emotions - train_emotions
            if missing_train:
                logger.error(f"Missing emotion folders in train: {missing_train}")
                return False
            
            # Check if all emotion folders exist in validation
            val_emotions = set(os.listdir(val_dir))
            missing_val = required_emotions - val_emotions
            if missing_val:
                logger.error(f"Missing emotion folders in validation: {missing_val}")
                return False
            
            logger.info("Data schema validated successfully")
            return True
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            return False

    def is_valid_image(self, file_path: str) -> bool:
        """Check if file is a valid image"""
        try:
            img = cv2.imread(file_path)
            if img is None:
                return False
            return True
        except Exception as e:
            logger.error(f"Error reading image {file_path}: {str(e)}")
            return False

    def validate_image_quality(self) -> Dict[str, List[str]]:
        """Validate image quality and return invalid image paths"""
        try:
            logger.info("Validating image quality")
            invalid_images = {"train": [], "validation": []}
            
            train_dir = os.path.join(self.config.unzip_data_dir, "train")
            val_dir = os.path.join(self.config.unzip_data_dir, "validation")
            
            # Validate train images
            for emotion in os.listdir(train_dir):
                emotion_path = os.path.join(train_dir, emotion)
                if os.path.isdir(emotion_path):
                    for image_file in os.listdir(emotion_path):
                        image_path = os.path.join(emotion_path, image_file)
                        if not self.is_valid_image(image_path):
                            invalid_images["train"].append(image_path)
            
            # Validate validation images
            for emotion in os.listdir(val_dir):
                emotion_path = os.path.join(val_dir, emotion)
                if os.path.isdir(emotion_path):
                    for image_file in os.listdir(emotion_path):
                        image_path = os.path.join(emotion_path, image_file)
                        if not self.is_valid_image(image_path):
                            invalid_images["validation"].append(image_path)
            
            logger.info(f"Invalid train images: {len(invalid_images['train'])}")
            logger.info(f"Invalid validation images: {len(invalid_images['validation'])}")
            
            return invalid_images
        except Exception as e:
            logger.error(f"Error validating image quality: {str(e)}")
            return {"train": [], "validation": []}

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Initiate complete data validation process"""
        try:
            logger.info("Starting data validation")
            
            # Create root directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Validate files exist
            files_valid = self.validate_data_files()
            
            # Validate schema
            schema_valid = self.validate_schema()
            
            # Validate image quality
            invalid_images = self.validate_image_quality()
            
            # Overall validation status
            validation_status = files_valid and schema_valid and \
                              len(invalid_images["train"]) == 0 and \
                              len(invalid_images["validation"]) == 0
            
            # Create drift report
            drift_report = {
                "files_valid": files_valid,
                "schema_valid": schema_valid,
                "invalid_train_images": invalid_images["train"],
                "invalid_validation_images": invalid_images["validation"],
                "validation_status": validation_status
            }
            
            # Save drift report
            drift_report_path = os.path.join(
                self.config.root_dir, 
                self.config.STATUS_FILE
            )
            
            with open(drift_report_path, 'w') as f:
                json.dump(drift_report, f, indent=4)
            
            logger.info(f"Validation report saved to {drift_report_path}")
            
            # Create valid and invalid data directories
            valid_train_path = os.path.join(self.config.root_dir, "valid_train")
            invalid_train_path = os.path.join(self.config.root_dir, "invalid_train")
            valid_val_path = os.path.join(self.config.root_dir, "valid_validation")
            invalid_val_path = os.path.join(self.config.root_dir, "invalid_validation")
            
            os.makedirs(valid_train_path, exist_ok=True)
            os.makedirs(invalid_train_path, exist_ok=True)
            os.makedirs(valid_val_path, exist_ok=True)
            os.makedirs(invalid_val_path, exist_ok=True)
            
            # Return artifact
            artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=Path(valid_train_path),
                valid_test_file_path=Path(valid_val_path),
                invalid_train_file_path=Path(invalid_train_path),
                invalid_test_file_path=Path(invalid_val_path),
                drift_report_file_path=Path(drift_report_path)
            )
            
            logger.info(f"Data Validation Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            raise e
