from dataclasses import dataclass
from pathlib import Path
from src.constants import *

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class ModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_image_size: list
    params_classes: int
    params_pooling: str
    params_activation: str
    