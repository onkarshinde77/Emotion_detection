import os
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from src.entity.config_entity import ModelConfig
from src.entity.artifact_entity import ModelArtifact
from src.constants import (
    IMG_SIZE, NUM_CLASSES, DENSE_UNITS,
    ACTIVATION, OUTPUT_ACTIVATION
)

logger = logging.getLogger(__name__)

class ModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config

    def build_model(self) -> Model:
        """Build the emotion detection model using MobileNetV2"""
        try:
            logger.info("Building emotion detection model")
            
            # Load Pretrained MobileNetV2
            base_model = MobileNetV2(
                weights=self.config.params_weights,
                include_top=self.config.params_include_top,
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            
            # Freeze base layers
            logger.info("Freezing base model layers")
            for layer in base_model.layers:
                layer.trainable = False
            
            # Custom classifier
            logger.info("Building custom classifier layers")
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(DENSE_UNITS, activation=ACTIVATION)(x)
            x = Dropout(0.5)(x)
            
            predictions = Dense(NUM_CLASSES, activation=OUTPUT_ACTIVATION)(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            logger.info("Model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise e

    def compile_model(self, model: Model) -> Model:
        """Compile the model"""
        try:
            logger.info("Compiling model")
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Model compiled successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error compiling model: {str(e)}")
            raise e

    def save_model(self, model: Model, model_path: str) -> None:
        """Save the model to disk"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e

    def initiate_model_builder(self) -> ModelArtifact:
        """Build and save the model"""
        try:
            logger.info("Initiating model builder")
            
            # Build model
            model = self.build_model()
            
            # Compile model
            model = self.compile_model(model)
            
            # Create output directory
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Save model
            model_path = self.config.updated_base_model_path
            self.save_model(model, str(model_path))
            
            # Create artifact
            artifact = ModelArtifact(
                trained_model_file_path=Path(model_path),
                model_history_file_path=Path(self.config.root_dir)
            )
            
            logger.info(f"Model Builder Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logger.error(f"Error in model builder: {str(e)}")
            raise e
