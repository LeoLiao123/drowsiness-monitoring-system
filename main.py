import cv2
import time
from src.face_detector import FaceDetector
from src.drowsiness_detector import DrowsinessDetector
from src.data_publisher import DataPublisher
from src.utils import draw_text
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, DETECTION_INTERVAL, PUBLISH_INTERVAL

class DrowsinessMonitoringSystem:
    def __init__(self):
        """Initialize the drowsiness monitoring system"""
        self.face_detector = FaceDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.data_publisher = DataPublisher()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.user_verified = False
        self.user_name = "Unknown"
        
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input video frame
        """
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        faces, gray = self.face_detector.detect_faces(frame)
        
        # Handle face detection and verification
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if not self.user_verified:
                name, variable_id, confidence = self.face_detector.verify_face(gray, (x, y, w, h))
                if name and variable_id:
                    self.user_verified = True
                    self.user_name = name
                    self.data_publisher.set_variable(variable_id)
            
            draw_text(frame, self.user_name, (x, y-5))
            
        # Process drowsiness detection
        landmarks = self.face_detector.get_facial_landmarks(gray)
        if landmarks:
            is_tired, is_yawning, metrics = self.drowsiness_detector.analyze_landmarks(landmarks)
            
            # Draw metrics
            draw_text(frame, f"Eye EAR: {metrics[0]:.2f}", (10, 30))
            draw_text(frame, f"Mouth EAR: {metrics[1]:.2f}", (10, 60))
            
        return frame
        
    def run(self):
        """Run the main monitoring loop"""
        last_detection_time = time.time()
        last_publish_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                processed_frame = self.process_frame(frame)
                
                # Periodic drowsiness check
                current_time = time.time()
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    tired_rate, yawning_rate = self.drowsiness_detector.get_drowsiness_metrics()
                    self.drowsiness_detector.reset_counters()
                    last_detection_time = current_time
                    
                    if tired_rate > 0.8 or yawning_rate > 0.8:
                        print("Warning: Drowsiness detected!")
                
                # Periodic data publishing
                if current_time - last_publish_time >= PUBLISH_INTERVAL:
                    tired_rate, _ = self.drowsiness_detector.get_drowsiness_metrics()
                    self.data_publisher.publish_drowsiness_rate(tired_rate)
                    last_publish_time = current_time
                
                cv2.imshow("Drowsiness Monitoring", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    monitoring_system = DrowsinessMonitoringSystem()
    monitoring_system.run()