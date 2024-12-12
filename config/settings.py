# API Configuration
UBIDOTS_API_KEY = "YOUR_API_KEY"

# Face Recognition Settings
CONFIDENCE_THRESHOLD = 60
VERIFICATION_TIME = 8
VERIFICATION_CONFIDENCE = 0.5

# Drowsiness Detection Settings
EYE_EAR_THRESHOLD = 0.25
MOUTH_EAR_THRESHOLD = 1.0
DETECTION_INTERVAL = 2  # seconds
PUBLISH_INTERVAL = 60   # seconds

# Camera Settings
FRAME_WIDTH = 400
FRAME_HEIGHT = 300

# User Database
USERS = {
    '1': {'name': 'Leo', 'variable_id': '63c1890182d6f4000c6a555f'},
    '2': {'name': 'Chen', 'variable_id': '63c249eab019de000cf16ef3'}
}

# Resource Paths
FACE_CASCADE_PATH = "resources/haarcascade_frontalface_default.xml"
SHAPE_PREDICTOR_PATH = "resources/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNIZER_MODEL = "models/face_model.yml"