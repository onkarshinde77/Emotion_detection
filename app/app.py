from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import tensorflow as tf
import io
from PIL import Image
import time

from src.logging.logger import logging
from src.pipelines.predict_pipeline import PredictPipeline

app = Flask(__name__)
# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/live')
def live():
    """Live camera prediction page"""
    return render_template('live.html')

@app.route('/upload')
def upload():
    """Image upload page"""
    return render_template('upload.html')

@app.route('/api/predict_image', methods=['POST'])
def predict_image():
    """API endpoint for image prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image file
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)

        # Convert to BGR format (OpenCV format)
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif image_array.shape[2] == 3:  # RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Run prediction
        results = predict_pipeline.predict_image_from_array(image_array)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        logging.error(f"Error in predict_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/live_feed')
def live_feed():
    """Video feed for live camera prediction"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate frames for live video feed"""
    try:
        # Load model and face cascade
        if not predict_pipeline.load_model():
            logging.error("Failed to load model")
            return

        if not predict_pipeline.load_face_cascade():
            logging.error("Failed to load face cascade")
            return

        # Start video capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logging.error("Failed to open camera")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Process frame for emotion detection
            processed_frame = predict_pipeline.process_frame_for_display(frame)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)

        cap.release()

    except Exception as e:
        logging.error(f"Error in generate_frames: {str(e)}")

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predict_pipeline.model is not None,
        'face_cascade_loaded': predict_pipeline.face_cascade is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)