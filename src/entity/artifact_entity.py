from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    trained_file_path: Path
    test_file_path: Path


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: Path
    valid_test_file_path: Path
    invalid_train_file_path: Path
    invalid_test_file_path: Path
    drift_report_file_path: Path


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: Path
    train_metric_artifact: Path
    test_metric_artifact: Path


@dataclass
class ModelArtifact:
    trained_model_file_path: Path
    model_history_file_path: Path