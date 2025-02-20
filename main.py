import cv2
import numpy as np
import os
import logging
from datetime import datetime
import requests
import json
import websockets
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class FaceRecognitionSystem:
    def __init__(self, camera_sources, faces_dir, access_system_url, api_key=None):
        self.setup_logging()
        
        # Initialize cameras
        self.cameras = []
        for source in camera_sources:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                self.cameras.append(cap)
            else:
                logging.error(f"Failed to open camera source: {source}")

        # Load known faces
        self.known_faces = self.load_known_faces(faces_dir)
        
        # Access system configuration
        self.access_system_url = access_system_url
        self.api_key = api_key
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Processing flags and cooldown
        self.is_running = True
        self.recent_detections = {}
        self.detection_cooldown = 5  # seconds
        
    def setup_logging(self):
        logging.basicConfig(
            filename=f'face_recognition_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_known_faces(self, faces_dir):
        """
        Load known faces from directory.
        Directory structure should be:
        faces_dir/
            badge_id1/
                photo1.jpg
                photo2.jpg
            badge_id2/
                photo1.jpg
        """
        known_faces = {}
        
        for badge_id in os.listdir(faces_dir):
            badge_path = os.path.join(faces_dir, badge_id)
            if os.path.isdir(badge_path):
                face_encodings = []
                for image_file in os.listdir(badge_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(badge_path, image_file)
                        image = cv2.imread(image_path)
                        if image is not None:
                            # Convert to grayscale for face detection
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            faces = self.face_cascade.detectMultiScale(
                                gray, 1.1, 4, minSize=(30, 30)
                            )
                            
                            for (x, y, w, h) in faces:
                                face = image[y:y+h, x:x+w]
                                # Resize for consistent comparison
                                face = cv2.resize(face, (128, 128))
                                face_encodings.append(face)
                
                if face_encodings:
                    known_faces[badge_id] = face_encodings
                    
        return known_faces

    def compare_faces(self, face1, face2, threshold=0.8):
        """Compare two faces using normalized correlation coefficient"""
        # Convert to grayscale
        face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
        face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
        
        # Resize to same size
        face1_resized = cv2.resize(face1_gray, (128, 128))
        face2_resized = cv2.resize(face2_gray, (128, 128))
        
        # Calculate correlation
        correlation = cv2.matchTemplate(
            face1_resized, 
            face2_resized, 
            cv2.TM_CCORR_NORMED
        )[0][0]
        
        return correlation > threshold

    def process_frame(self, frame):
        """Process a single frame and return recognized badge IDs"""
        recognized_badges = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Compare each detected face with known faces
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Compare with each known face
            for badge_id, known_faces in self.known_faces.items():
                for known_face in known_faces:
                    if self.compare_faces(face, known_face):
                        recognized_badges.append(badge_id)
                        break
                if badge_id in recognized_badges:
                    break
                    
        return recognized_badges

    def send_badge_signal(self, badge_id):
        """Send badge ID to access control system"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}' if self.api_key else None
            }
            
            data = {
                "badge_id": badge_id,
                "timestamp": datetime.now().isoformat(),
                "source": "face_recognition_system"
            }
            
            response = requests.post(
                self.access_system_url,
                json=data,
                headers=headers
            )
            
            if response.status_code == 200:
                logging.info(f"Successfully sent badge signal for ID: {badge_id}")
                return True
            else:
                logging.error(f"Failed to send badge signal: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Error sending badge signal: {str(e)}")
            return False

    def process_camera(self, camera_index):
        """Process frames from a single camera"""
        camera = self.cameras[camera_index]
        
        while self.is_running:
            ret, frame = camera.read()
            if not ret:
                logging.error(f"Failed to read from camera {camera_index}")
                continue
                
            recognized_badges = self.process_frame(frame)
            
            current_time = time.time()
            for badge_id in recognized_badges:
                if badge_id not in self.recent_detections or \
                   current_time - self.recent_detections[badge_id] > self.detection_cooldown:
                    
                    self.recent_detections[badge_id] = current_time
                    logging.info(f"Recognized user with badge ID: {badge_id}")
                    self.send_badge_signal(badge_id)

    def run(self):
        """Run the face recognition system"""
        logging.info("Starting face recognition system")
        
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            futures = []
            for i in range(len(self.cameras)):
                futures.append(executor.submit(self.process_camera, i))

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        for camera in self.cameras:
            camera.release()
        cv2.destroyAllWindows()
        logging.info("Face recognition system shutdown complete")

# Example usage
if __name__ == "__main__":
    # Configuration
    camera_sources = [0, 2, 4, 6, 8]  # Adjust based on your camera setup
    faces_dir = "/path/to/faces/directory"
    access_system_url = "https://access-control-system.company.com/api/badge-scan"
    api_key = "your_api_key_here"  # If required
    
    system = FaceRecognitionSystem(
        camera_sources=camera_sources,
        faces_dir=faces_dir,
        access_system_url=access_system_url,
        api_key=api_key
    )
    
    try:
        system.run()
    except KeyboardInterrupt:
        system.cleanup()
