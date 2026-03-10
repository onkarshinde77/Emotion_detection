# Image and Model Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

# Paths
DATA_DIR = "data"
TRAIN_DIR = f"{DATA_DIR}/train"
VALIDATION_DIR = f"{DATA_DIR}/validation"
MODEL_PATH = "artifacts/model_trainer/emotion_mobilenetv2_trained.h5"
FACE_CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

# Emotion Labels (These MUST match the alphabetical order of data directories)
# flow_from_directory sorts folders alphabetically: angry, disgust, fear, happy, neutral, sad, surprise
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Training Hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 10  # Adjust based on needs
DROPOUT_RATE = 0.5

# Data Augmentation Parameters
ROTATION_RANGE = 20
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

# Face Detection Parameters
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5

# Model Architecture
BASE_MODEL = "MobileNetV2"
DENSE_UNITS = 128
ACTIVATION = "relu"
OUTPUT_ACTIVATION = "softmax"

