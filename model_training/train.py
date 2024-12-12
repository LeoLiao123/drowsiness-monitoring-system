import cv2
import numpy as np
import os
import re

def get_face_folders(base_path='.'):
    # Find all folders starting with 'face'
    folders = [f for f in os.listdir(base_path) if os.path.isdir(f) and f.startswith('face')]
    
    # Extract folders with proper format (face followed by numbers)
    face_folders = []
    for folder in folders:
        # Extract numbers after 'face'
        match = re.search(r'face(\d+)', folder)
        if match:
            label = int(match.group(1))  # Convert extracted number to integer
            face_folders.append((folder, label))
    
    # Sort by label number
    return sorted(face_folders, key=lambda x: x[1])

def train_face_recognizer(model_path='../resources/haarcascade_frontalface_default.xml'):
    try:
        # Initialize face detection and recognition
        detector = cv2.CascadeClassifier(model_path)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        faces = []  # Store face regions
        labels = []  # Store face labels
        
        # Get all face folders
        face_folders = get_face_folders()
        
        if not face_folders:
            print("No valid face folders found! Folders should be named 'face01', 'face02', etc.")
            return False
            
        print(f"Found {len(face_folders)} face folders to process")
        
        # Process each folder
        for folder, label in face_folders:
            print(f"Processing folder: {folder} (Label: {label})")
            
            # Get all jpg/jpeg images in the folder
            image_files = [f for f in os.listdir(folder) 
                         if f.lower().endswith(('.jpg', '.jpeg'))]
            
            for image_file in image_files:
                image_path = os.path.join(folder, image_file)
                img = cv2.imread(image_path)
                
                if img is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_np = np.array(gray, 'uint8')
                
                # Detect faces
                face_regions = detector.detectMultiScale(gray)
                
                if len(face_regions) == 0:
                    print(f"Warning: No face detected in {image_path}")
                    continue
                    
                # Process detected faces
                for (x, y, w, h) in face_regions:
                    faces.append(img_np[y:y+h, x:x+w])
                    labels.append(label)
        
        if not faces:
            print("No faces were detected in any of the images!")
            return False
            
        print(f"\nStarting training with {len(faces)} faces...")
        recognizer.train(faces, np.array(labels))
        recognizer.save('../model/face_model.yml')
        print("Training completed successfully!")
        
        # Print summary
        unique_labels = len(set(labels))
        print(f"\nSummary:")
        print(f"- Total faces processed: {len(faces)}")
        print(f"- Number of unique individuals: {unique_labels}")
        print(f"- Model saved as: /model/face_model.yml")
        
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    # Set path to haar cascade classifier
    CASCADE_PATH = '../resources/haarcascade_frontalface_default.xml'
    
    # Verify cascade file exists
    if not os.path.exists(CASCADE_PATH):
        print(f"Error: Cascade file not found at {CASCADE_PATH}")
        print("Please check the path to haarcascade_frontalface_default.xml")
        exit(1)
    
    # Run training
    print("Face Recognition Model Trainer")
    print("-----------------------------")
    train_face_recognizer(CASCADE_PATH)