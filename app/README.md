# Emotion Detection Web App

A Flask-based web application for real-time emotion detection using computer vision and deep learning.

## Features

- **Live Camera Detection**: Real-time emotion recognition from webcam feed
- **Image Upload**: Upload images to detect emotions in photos
- **REST API**: API endpoints for integration with other applications
- **Responsive Design**: Works on desktop and mobile devices

## Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Installation

1. Navigate to the app directory:
```bash
cd app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have trained the model first by running the training pipeline:
```bash
python -m src.pipelines.train_pipeline
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:5000`

3. Choose from the available options:
   - **Live Camera**: Real-time emotion detection
   - **Upload Image**: Analyze emotions in uploaded photos
   - **API Access**: Use the REST API endpoints

## API Endpoints

- `GET /`: Home page
- `GET /live`: Live camera page
- `GET /upload`: Image upload page
- `POST /api/predict_image`: Predict emotions in uploaded image
- `GET /api/live_feed`: MJPEG stream for live camera
- `GET /api/health`: Health check endpoint

## Project Structure

```
app/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css       # CSS stylesheets
└── templates/
    ├── index.html      # Home page
    ├── live.html       # Live camera page
    └── upload.html     # Image upload page
```

## Requirements

- Python 3.8+
- Webcam (for live detection)
- Trained emotion detection model
- Haar Cascade classifier for face detection

## Troubleshooting

1. **Camera not working**: Make sure your webcam is not being used by other applications
2. **Model not found**: Ensure you've trained the model using the training pipeline
3. **Face cascade not found**: Check that `models/haarcascade_frontalface_default.xml` exists

## License

This project is part of the Emotion Detection System.