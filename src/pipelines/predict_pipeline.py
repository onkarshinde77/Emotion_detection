import logging
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.constants import (
    IMG_SIZE, EMOTION_LABELS, MODEL_PATH,
    FACE_CASCADE_PATH, SCALE_FACTOR, MIN_NEIGHBORS
)

logger = logging.getLogger(__name__)


class PredictPipeline:
    def __init__(self, model_path: str = MODEL_PATH, 
                 face_cascade_path: str = FACE_CASCADE_PATH):
        """
        Initialize the prediction pipeline
        
        Args:
            model_path: Path to the trained emotion detection model
            face_cascade_path: Path to the Haar Cascade XML file for face detection
        """
        self.model_path = model_path
        self.face_cascade_path = face_cascade_path
        self.model = None
        self.face_cascade = None
        self.emotion_labels = EMOTION_LABELS
        
        # Color mapping for emotions
        self.emotion_colors = {
            "Angry": (0, 0, 255),        # Red
            "Disgust": (0, 255, 0),      # Green
            "Fear": (255, 0, 0),         # Blue
            "Happy": (0, 255, 255),      # Yellow
            "Neutral": (128, 128, 128),  # Gray
            "Sad": (128, 0, 128),        # Purple
            "Surprise": (255, 255, 0)    # Cyan
        }
        
    def load_model(self) -> bool:
        """Load the trained emotion detection model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def load_face_cascade(self) -> bool:
        """Load the Haar Cascade classifier for face detection"""
        try:
            logger.info(f"Loading face cascade from {self.face_cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(str(self.face_cascade_path))
            
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                return False
            
            logger.info("Face cascade loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading face cascade: {str(e)}")
            return False

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model prediction
        
        Args:
            face: Face image array
            
        Returns:
            Preprocessed face image
        """
        try:
            # Resize to model input size
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values
            face = face.astype('float32') / 255.0
            
            # Reshape to match model input
            face = np.expand_dims(face, axis=0)
            
            return face
        except Exception as e:
            logger.error(f"Error preprocessing face: {str(e)}")
            return None

    def predict_emotion(self, face: np.ndarray) -> tuple:
        """
        Predict emotion for a given face image
        
        Args:
            face: Face image array
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        try:
            # Preprocess face
            preprocessed_face = self.preprocess_face(face)
            
            if preprocessed_face is None:
                return "Unknown", 0.0
            
            # Get predictions
            predictions = self.model.predict(preprocessed_face, verbose=0)
            emotion_index = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_index])
            emotion = self.emotion_labels[emotion_index]
            
            return emotion, confidence
        except Exception as e:
            logger.error(f"Error predicting emotion: {str(e)}")
            return "Unknown", 0.0

    def run_live_camera(self, camera_id: int = 0, display_confidence: bool = True):
        """
        Run live emotion detection using webcam
        
        Args:
            camera_id: ID of the camera to use (default: 0 for built-in webcam)
            display_confidence: Whether to display confidence scores
        """
        try:
            # Load model and face cascade
            if not self.load_model():
                logger.error("Failed to load model. Exiting.")
                return
            
            if not self.load_face_cascade():
                logger.error("Failed to load face cascade. Exiting.")
                return
            
            logger.info("Starting live camera prediction")
            logger.info("Press 'q' to quit")
            
            # Start video capture
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=(30, 30)
                )
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(face_roi)
                    
                    # Get color for the emotion
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Prepare text to display
                    if display_confidence:
                        text = f"{emotion} ({confidence:.2f})"
                    else:
                        text = emotion
                    
                    # Get text size for background
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.9, 2)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, 
                                (x, y - text_size[1] - 10),
                                (x + text_size[0] + 10, y),
                                color, -1)
                    
                    # Put text on frame
                    cv2.putText(frame, text, (x + 5, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.9, (255, 255, 255), 2)
                
                # Display FPS
                cv2.putText(frame, "Press 'q' to quit", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow("Emotion Detection - Live Camera", frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quitting live camera prediction")
                    break
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Live camera prediction ended")
            
        except Exception as e:
            logger.error(f"Error in live camera prediction: {str(e)}")
            raise e

    def predict_image_from_array(self, image_array: np.ndarray) -> dict:
        """
        Predict emotions in an image array
        
        Args:
            image_array: Image as numpy array (BGR format)
            
        Returns:
            Dictionary with emotions and confidence scores for all faces
        """
        try:
            # Load model and face cascade if not loaded
            if not self.load_model():
                logger.error("Failed to load model")
                return {}
            
            if not self.load_face_cascade():
                logger.error("Failed to load face cascade")
                return {}
            
            logger.info("Processing image array")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(30, 30)
            )
            
            results = {}
            
            # Process each face
            for idx, (x, y, w, h) in enumerate(faces):
                face_roi = image_array[y:y+h, x:x+w]
                emotion, confidence = self.predict_emotion(face_roi)
                
                results[f"face_{idx}"] = {
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "coordinates": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                }
            
            logger.info(f"Found {len(faces)} face(s) in image")
            return results
            
        except Exception as e:
            logger.error(f"Error predicting image array: {str(e)}")
            return {}

    def process_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame for display with emotion detection overlay
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with emotion labels
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence = self.predict_emotion(face_roi)
                
                # Get color for the emotion
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Prepare text to display
                text = f"{emotion} ({confidence:.2f})"
                
                # Get text size for background
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.9, 2)[0]
                
                # Draw background rectangle for text
                cv2.rectangle(frame, 
                            (x, y - text_size[1] - 10),
                            (x + text_size[0] + 10, y),
                            color, -1)
                
                # Put text on frame
                cv2.putText(frame, text, (x + 5, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.9, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run live camera prediction
    pipeline = PredictPipeline()
    pipeline.run_live_camera(display_confidence=True)
