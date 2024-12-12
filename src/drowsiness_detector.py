from src.utils import calculate_ear
from config.settings import EYE_EAR_THRESHOLD, MOUTH_EAR_THRESHOLD

class DrowsinessDetector:
    def __init__(self):
        """Initialize drowsiness detection parameters"""
        self.reset_counters()
        
    def reset_counters(self):
        """Reset all detection counters"""
        self.tired_count = 0
        self.mouth_open_count = 0
        self.frame_count = 0
        
    def analyze_landmarks(self, landmarks):
        """
        Analyze facial landmarks for signs of drowsiness
        
        Args:
            landmarks (list): Facial landmark points
            
        Returns:
            tuple: (is_tired, is_yawning, metrics)
        """
        if not landmarks or len(landmarks) < 69:
            return False, False, (0, 0)
            
        # Calculate eye EAR
        left_eye_points = [landmarks[36], landmarks[37], landmarks[38], 
                          landmarks[39], landmarks[40], landmarks[41]]
        right_eye_points = [landmarks[42], landmarks[43], landmarks[44], 
                           landmarks[45], landmarks[46], landmarks[47]]
        
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate mouth EAR
        mouth_points = [landmarks[61], landmarks[63], landmarks[65], 
                       landmarks[67], landmarks[69], landmarks[65]]
        mouth_ear = calculate_ear(mouth_points)
        
        # Update counters
        self.frame_count += 1
        if avg_ear < EYE_EAR_THRESHOLD:
            self.tired_count += 1
        if mouth_ear > MOUTH_EAR_THRESHOLD:
            self.mouth_open_count += 1
            
        return (avg_ear < EYE_EAR_THRESHOLD, 
                mouth_ear > MOUTH_EAR_THRESHOLD, 
                (avg_ear, mouth_ear))
                
    def get_drowsiness_metrics(self):
        """
        Calculate drowsiness metrics
        
        Returns:
            tuple: (tired_rate, yawning_rate)
        """
        if self.frame_count == 0:
            return 0, 0
            
        tired_rate = self.tired_count / self.frame_count
        yawning_rate = self.mouth_open_count / self.frame_count
        
        return tired_rate, yawning_rate