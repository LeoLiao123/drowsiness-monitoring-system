import math
import cv2

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1 (tuple): First point coordinates (x, y)
        point2 (tuple): Second point coordinates (x, y)
    
    Returns:
        float: Distance between the two points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR)
    
    Args:
        eye_points (list): List of eye landmark points
    
    Returns:
        float: Calculated EAR value
    """
    if calculate_distance(eye_points[0], eye_points[3]) == 0:
        return 0
    
    return (calculate_distance(eye_points[1], eye_points[5]) + 
            calculate_distance(eye_points[2], eye_points[4])) / (
            2 * calculate_distance(eye_points[0], eye_points[3]))

def draw_text(frame, text, position, scale=0.8, color=(255, 255, 255), thickness=2):
    """
    Draw text on frame with consistent styling
    
    Args:
        frame: Image frame
        text (str): Text to draw
        position (tuple): Position coordinates (x, y)
        scale (float): Font scale
        color (tuple): RGB color values
        thickness (int): Text thickness
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                scale, color, thickness, cv2.LINE_AA)