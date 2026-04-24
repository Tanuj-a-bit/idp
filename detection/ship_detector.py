import cv2
import numpy as np
import torch
from ultralytics import YOLO

class ShipDetector:
    def __init__(self, model_path='yolov8n.pt', device='cpu'):
        """
        Initialises the YOLOv8 ship detector. 
        In production, we wrap this for TensorRT.
        """
        # Load YOLOv8 model (using 'n'ano for speed, 's'/'m' for accuracy)
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            # Fall back directly to CPU. (Avoiding Apple 'mps' due to 
            # a known Ultralytics Non-Maximum Suppression (NMS) freezing bug)
            self.device = 'cpu'

        
        self.model.to(self.device)
        
        # Ship-related COCO class IDs (8: boat)
        self.ship_class_id = 8

    def detect(self, frame):
        """
        Detects ships in a raw image frame.
        Returns: [x1, y1, x2, y2, confidence, class_id] per detection.
        """
        results = self.model(frame, classes=[self.ship_class_id], verbose=False)
        
        # Get annotated frame with boxes/labels directly from YOLO
        annotated_frame = results[0].plot()
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = box.cls[0].cpu().item()
                detections.append(np.append(coords, [conf, cls]))
                
        return np.array(detections), annotated_frame

if __name__ == "__main__":
    # Smoke test on dummy image
    detector = ShipDetector()
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    results = detector.detect(dummy_img)
    print(f"Detected {len(results)} ships in dummy image.")
    if results.size > 0:
        print(f"First detection: {results[0]}")
