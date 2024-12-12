import cv2
import dlib
from config.settings import (
    FACE_CASCADE_PATH, 
    SHAPE_PREDICTOR_PATH, 
    FACE_RECOGNIZER_MODEL,
    CONFIDENCE_THRESHOLD,
    USERS
)

class FaceDetector:
    def __init__(self):
        """Initialize face detection and recognition components"""
        # Load face detection models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(FACE_RECOGNIZER_MODEL)
        
    def detect_faces(self, frame):
        """
        Detect faces in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (faces, gray_frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return faces, gray
        
    def get_facial_landmarks(self, gray_frame):
        """
        Detect facial landmarks using dlib
        
        Args:
            gray_frame: Grayscale input frame
            
        Returns:
            list: List of facial landmark points
        """
        faces = self.detector(gray_frame)
        if not faces:
            return None
            
        landmarks = self.predictor(gray_frame, faces[0])
        return [(point.x, point.y) for point in landmarks.parts()]
        
    def verify_face(self, gray_frame, face_region):
        """
        Verify face identity
        
        Args:
            gray_frame: Grayscale input frame
            face_region: Region containing the face (x, y, w, h)
            
        Returns:
            tuple: (user_id, confidence)
        """
        x, y, w, h = face_region
        face_roi = gray_frame[y:y+h, x:x+w]
        user_id, confidence = self.recognizer.predict(face_roi)
        
        if confidence < CONFIDENCE_THRESHOLD:
            user_info = USERS.get(str(user_id))
            if user_info:
                return user_info['name'], user_info['variable_id'], confidence
                
        return None, None, confidence