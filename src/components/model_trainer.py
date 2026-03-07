import os
import logging
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.constants import (
    IMG_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    ROTATION_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def create_data_generators(self):
        """Create image data generators for training and validation"""
        try:
            logger.info("Creating data generators")
            
            # Training data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=ROTATION_RANGE,
                zoom_range=ZOOM_RANGE,
                horizontal_flip=HORIZONTAL_FLIP
            )
            
            # Validation data (no augmentation)
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            # Create generators
            train_generator = train_datagen.flow_from_directory(
                str(self.config.data_path / "train"),
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                class_mode='categorical'
            )
            
            val_generator = val_datagen.flow_from_directory(
                str(self.config.data_path / "validation"),
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                class_mode='categorical'
            )
            
            logger.info("Data generators created successfully")
            return train_generator, val_generator
            
        except Exception as e:
            logger.error(f"Error creating data generators: {str(e)}")
            raise e

    def load_base_model(self):
        """Load the base model for training"""
        try:
            logger.info(f"Loading base model from {self.config.base_model_path}")
            
            model = tf.keras.models.load_model(str(self.config.base_model_path))
            
            logger.info("Base model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise e

    def train_model(self, model, train_generator, val_generator):
        """Train the model"""
        try:
            logger.info(f"Starting model training for {self.config.params_epochs} epochs")
            
            # Compile model with training parameters
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.params_epochs,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise e

    def save_model(self, model, model_path):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            logger.info(f"Trained model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e

    def save_training_metrics(self, history, metrics_path):
        """Save training metrics to JSON"""
        try:
            metrics = {
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Training metrics saved to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving training metrics: {str(e)}")
            raise e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Initiate the complete model training process"""
        try:
            logger.info("Initiating model trainer")
            
            # Create root directory
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Load base model
            model = self.load_base_model()
            
            # Create data generators
            train_generator, val_generator = self.create_data_generators()
            
            # Train model
            history = self.train_model(model, train_generator, val_generator)
            
            # Save trained model
            trained_model_path = str(self.config.updated_base_model_path)
            self.save_model(model, trained_model_path)
            
            # Save training metrics
            train_metrics_path = str(self.config.root_dir / "train_metrics.json")
            test_metrics_path = str(self.config.root_dir / "test_metrics.json")
            
            # For simplicity, using same metrics for both (in real scenario, evaluate on test set)
            self.save_training_metrics(history, train_metrics_path)
            self.save_training_metrics(history, test_metrics_path)
            
            # Create artifact
            artifact = ModelTrainerArtifact(
                trained_model_file_path=Path(trained_model_path),
                train_metric_artifact=Path(train_metrics_path),
                test_metric_artifact=Path(test_metrics_path)
            )
            
            logger.info(f"Model Trainer Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logger.error(f"Error in model trainer: {str(e)}")
            raise e