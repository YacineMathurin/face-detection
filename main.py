import cv2
import numpy as np
import tensorflow as tf
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import requests
import json
import websockets
import asyncio
import ssl

class BadgeSignalEmitter:
    def __init__(self, access_system_url, api_key=None):
        """
        Initialize badge signal emitter that communicates with the access control system
        via websockets or REST API depending on the system's requirements
        """
        self.access_system_url = access_system_url
        self.api_key = api_key
        self.websocket = None
        
    async def connect_websocket(self):
        """
        Establish websocket connection to access control system
        """
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.websocket = await websockets.connect(
                self.access_system_url,
                ssl=ssl_context
            )
            logging.info("Connected to access control system")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to access control system: {str(e)}")
            return False

    async def emit_badge_signal(self, badge_id):
        """
        Send badge ID to access control system
        Supports both WebSocket and REST API methods
        """
        try:
            if self.access_system_url.startswith('ws'):
                # WebSocket method
                if not self.websocket:
                    await self.connect_websocket()
                
                message = {
                    "type": "badge_scan",
                    "badge_id": badge_id,
                    "timestamp": datetime.now().isoformat(),
                    "source": "face_recognition_system"
                }
                
                await self.websocket.send(json.dumps(message))
                response = await self.websocket.recv()
                logging.info(f"Badge signal sent successfully: {response}")
                
            else:
                # REST API method
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}' if self.api_key else None
                }
                
                response = requests.post(
                    self.access_system_url,
                    json={
                        "badge_id": badge_id,
                        "timestamp": datetime.now().isoformat(),
                        "source": "face_recognition_system"
                    },
                    headers=headers
                )
                
                if response.status_code == 200:
                    logging.info(f"Badge signal sent successfully via REST API")
                else:
                    logging.error(f"Failed to send badge signal: {response.status_code}")
                    
            return True
            
        except Exception as e:
            logging.error(f"Error sending badge signal: {str(e)}")
            return False
            
    async def close(self):
        if self.websocket:
            await self.websocket.close()

class FaceRecognitionSystem:
    def __init__(self, camera_sources, face_model_path, access_system_url, api_key=None):
        self.setup_logging()
        
        # Initialize cameras
        self.cameras = []
        for source in camera_sources:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                self.cameras.append(cap)
            else:
                logging.error(f"Failed to open camera source: {source}")
        
        # Load face detection model
        prototxt_path = "deploy.prototxt"
        caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        
        # Load TensorFlow face recognition model
        self.face_recognizer = self.load_face_recognition_model(face_model_path)
        
        # Initialize badge emitter
        self.badge_emitter = BadgeSignalEmitter(access_system_url, api_key)
        
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Processing flags
        self.is_running = True
        
        # Track recent detections
        self.recent_detections = {}
        self.detection_cooldown = 5  # seconds
        
    def setup_logging(self):
        logging.basicConfig(
            filename=f'face_recognition_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    # [Previous methods remain the same: load_face_recognition_model, detect_faces, process_frame]
    
    def process_camera(self, camera_index):
        camera = self.cameras[camera_index]
        
        while self.is_running:
            ret, frame = camera.read()
            if not ret:
                logging.error(f"Failed to read from camera {camera_index}")
                continue
                
            recognized_badges = self.process_frame(frame)
            
            for badge_id in recognized_badges:
                current_time = time.time()
                if badge_id not in self.recent_detections or \
                   current_time - self.recent_detections[badge_id] > self.detection_cooldown:
                    
                    self.recent_detections[badge_id] = current_time
                    logging.info(f"Recognized user with badge ID: {badge_id}")
                    
                    # Send badge signal asynchronously
                    asyncio.run_coroutine_threadsafe(
                        self.badge_emitter.emit_badge_signal(badge_id),
                        self.loop
                    )
    
    def run(self):
        logging.info("Starting face recognition system")
        
        # Start the async event loop in a separate thread
        loop_thread = threading.Thread(target=lambda: self.loop.run_forever())
        loop_thread.start()
        
        # Start camera processing
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            futures = []
            for i in range(len(self.cameras)):
                futures.append(executor.submit(self.process_camera, i))
                
    async def cleanup(self):
        self.is_running = False
        for camera in self.cameras:
            camera.release()
        cv2.destroyAllWindows()
        await self.badge_emitter.close()
        self.loop.stop()
        logging.info("Face recognition system shutdown complete")

# Example usage
if __name__ == "__main__":
    # Configuration
    camera_sources = [0, 2, 4, 6, 8]  # Adjust based on your camera setup
    face_model_path = "/path/to/your/tensorflow/model"
    
    # Access control system configuration
    # Example for WebSocket
    access_system_url = "ws://access-control-system.company.com/badge-reader"
    # Example for REST API
    # access_system_url = "https://access-control-system.company.com/api/badge-scan"
    api_key = "your_api_key_here"  # If required by the access control system
    
    recognition_system = FaceRecognitionSystem(
        camera_sources=camera_sources,
        face_model_path=face_model_path,
        access_system_url=access_system_url,
        api_key=api_key
    )
    
    try:
        recognition_system.run()
    except KeyboardInterrupt:
        asyncio.run(recognition_system.cleanup())
