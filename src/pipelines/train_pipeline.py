import logging
from pathlib import Path
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    ModelConfig,
    ModelTrainerConfig
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.model import ModelBuilder
from src.components.model_trainer import ModelTrainer
from src.constants import (
    DATA_DIR, TRAIN_DIR, VALIDATION_DIR,    
    MODEL_PATH, LEARNING_RATE, EPOCHS
)

logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(self):
        self.config = None
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Create DataIngestionConfig"""
        try:
            config = DataIngestionConfig(
                root_dir=Path("artifacts/data_ingestion"),
                source_URL="",  # Optional: add data source URL if needed
                local_data_file=Path("artifacts/data_ingestion/data.zip"),
                unzip_dir=Path("artifacts/data_ingestion")
            )
            logger.info(f"DataIngestionConfig created: {config}")
            return config
        except Exception as e:
            logger.error(f"Error creating DataIngestionConfig: {str(e)}")
            raise e

    def get_data_validation_config(self) -> DataValidationConfig:
        """Create DataValidationConfig"""
        try:
            config = DataValidationConfig(
                root_dir=Path("artifacts/data_validation"),
                STATUS_FILE="validation_status.json",
                unzip_data_dir=Path(DATA_DIR),
                all_schema={
                    "Angry": {"count": 0},
                    "Disgust": {"count": 0},
                    "Fear": {"count": 0},
                    "Happy": {"count": 0},
                    "Neutral": {"count": 0},
                    "Sad": {"count": 0},
                    "Surprise": {"count": 0}
                }
            )
            logger.info(f"DataValidationConfig created: {config}")
            return config
        except Exception as e:
            logger.error(f"Error creating DataValidationConfig: {str(e)}")
            raise e

    def get_model_config(self) -> ModelConfig:
        """Create ModelConfig"""
        try:
            config = ModelConfig(
                root_dir=Path("artifacts/model"),
                base_model_path=Path(MODEL_PATH),
                updated_base_model_path=Path(f"artifacts/model/{MODEL_PATH.split('/')[-1]}"),
                params_learning_rate=LEARNING_RATE,
                params_include_top=False,
                params_weights="imagenet",
                params_image_size=[224, 224, 3],
                params_classes=7,
                params_pooling="avg",
                params_activation="relu"
            )
            logger.info(f"ModelConfig created: {config}")
            return config
        except Exception as e:
            logger.error(f"Error creating ModelConfig: {str(e)}")
            raise e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Create ModelTrainerConfig"""
        try:
            config = ModelTrainerConfig(
                root_dir=Path("artifacts/model_trainer"),
                data_path=Path(DATA_DIR),
                model_ckpt=Path("artifacts/model/emotion_mobilenetv2.h5"),
                base_model_path=Path("artifacts/model/emotion_mobilenetv2.h5"),
                updated_base_model_path=Path("artifacts/model_trainer/emotion_mobilenetv2_trained.h5"),
                params_epochs=EPOCHS,
                params_batch_size=32,
                params_is_augmentation=True,
                params_image_size=[224, 224, 3]
            )
            logger.info(f"ModelTrainerConfig created: {config}")
            return config
        except Exception as e:
            logger.error(f"Error creating ModelTrainerConfig: {str(e)}")
            raise e

    def run_data_ingestion(self) -> DataIngestionConfig:
        """Execute data ingestion stage"""
        try:
            logger.info("Starting DataIngestion stage")
            
            config = self.get_data_ingestion_config()
            data_ingestion = DataIngestion(config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logger.info(f"DataIngestion completed: {data_ingestion_artifact}")
            return config
            
        except Exception as e:
            logger.error(f"Error in DataIngestion: {str(e)}")
            raise e

    def run_data_validation(self) -> DataValidationConfig:
        """Execute data validation stage"""
        try:
            logger.info("Starting DataValidation stage")
            
            config = self.get_data_validation_config()
            data_validation = DataValidation(config)
            data_validation_artifact = data_validation.initiate_data_validation()
            
            if not data_validation_artifact.validation_status:
                logger.warning("Data validation failed. Check validation report.")
            else:
                logger.info("Data validation passed successfully")
            
            logger.info(f"DataValidation completed: {data_validation_artifact}")
            return config
            
        except Exception as e:
            logger.error(f"Error in DataValidation: {str(e)}")
            raise e

    def run_model_builder(self) -> ModelConfig:
        """Execute model building stage"""
        try:
            logger.info("Starting ModelBuilder stage")
            
            config = self.get_model_config()
            model_builder = ModelBuilder(config)
            model_builder_artifact = model_builder.initiate_model_builder()
            
            logger.info(f"ModelBuilder completed: {model_builder_artifact}")
            return config
            
        except Exception as e:
            logger.error(f"Error in ModelBuilder: {str(e)}")
            raise e

    def run_model_trainer(self) -> ModelTrainerConfig:
        """Execute model training stage"""
        try:
            logger.info("Starting ModelTrainer stage")
            
            config = self.get_model_trainer_config()
            model_trainer = ModelTrainer(config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            logger.info(f"ModelTrainer completed: {model_trainer_artifact}")
            return config
            
        except Exception as e:
            logger.error(f"Error in ModelTrainer: {str(e)}")
            raise e

    def run(self):
        """Run the complete training pipeline"""
        try:
            logger.info("=" * 50)
            logger.info("Starting Emotion Detection Training Pipeline")
            logger.info("=" * 50)
            
            # Stage 1: Data Ingestion
            logger.info("\n--- Stage 1: Data Ingestion ---")
            self.run_data_ingestion()
            
            # Stage 2: Data Validation
            logger.info("\n--- Stage 2: Data Validation ---")
            self.run_data_validation()
            
            # Stage 3: Model Building
            logger.info("\n--- Stage 3: Model Building ---")
            self.run_model_builder()
            
            # Stage 4: Model Training
            logger.info("\n--- Stage 4: Model Training ---")
            self.run_model_trainer()
            
            logger.info("\n" + "=" * 50)
            logger.info("Training Pipeline completed successfully!")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise e


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = TrainPipeline()
    pipeline.run()
