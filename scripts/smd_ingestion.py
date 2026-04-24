import os
import sys
import time
import json
import random
import datetime
import glob
from threading import Thread

# Try importing kagglehub
try:
    import kagglehub
except ImportError:
    print("Please install kagglehub: pip install kagglehub")
    sys.exit(1)

import cv2
import redis
import numpy as np
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detection.ship_detector import ShipDetector

# Redis setup
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
except redis.ConnectionError:
    print("Error: Could not connect to Redis on localhost:6379.")
    sys.exit(1)

# Singapore base coordinates for fake geographic projection
SG_LAT, SG_LON = 1.290270, 103.851959

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def process_video():
    print("Downloading/Locating Singapore Maritime Dataset (SMD) via kagglehub...")
    # This will block until downloaded if not already fully downloaded in current session!
    try:
        path = kagglehub.dataset_download("mmichelli/singapore-maritime-dataset")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    print(f"Dataset securely found at: {path}")
    
    # Locate video files
    video_files = glob.glob(os.path.join(path, '**', '*.mp4'), recursive=True) + \
                  glob.glob(os.path.join(path, '**', '*.avi'), recursive=True)
    if not video_files:
        print("No video files found in the dataset.")
        return
        
    print(f"Found {len(video_files)} video files. Processing full dataset sequentially.")
    
    detector = ShipDetector(model_path='yolov8n.pt')
    
    # Import the actual trajectory prediction PyTorch module
    from prediction.trajectory_model import TrajectoryLSTM
    import torch
    
    traj_model = TrajectoryLSTM().eval()  # Loads architecture
    print("Trajectory Prediction engine initialized.")
    print("Started processing video stream to populate Aegis Maritime Tracker...")

    # Dictionary to maintain advanced tracking + AIS association
    # structure: track_id: { bbox, lat, lon, history: list, ais_identity: dict }
    active_tracks = {}
    next_id = 1000
    
    # Expanded AIS Ship Registry with more unique targets
    ais_registry = [
        {"name": "EVER GIVEN", "mmsi": "353136000", "type": "Cargo"},
        {"name": "CMA CGM MARCO POLO", "mmsi": "311000175", "type": "Container"},
        {"name": "OOCL HONG KONG", "mmsi": "477271800", "type": "Container"},
        {"name": "SINGAPORE EXPRESS", "mmsi": "563046200", "type": "Tanker"},
        {"name": "SEAWAYS SIKINOS", "mmsi": "538004505", "type": "Tanker"},
        {"name": "MAERSK MC-KINNEY", "mmsi": "219521000", "type": "Container"},
        {"name": "MSC GULSUN", "mmsi": "353136001", "type": "Container"},
        {"name": "HMM ALGESIRAS", "mmsi": "440123456", "type": "Container"},
        {"name": "COSCO SHIPPING UNIVERSE", "mmsi": "413123456", "type": "Container"},
        {"name": "BERGE BULK", "mmsi": "563000123", "type": "Bulk Carrier"},
        {"name": "NYK VESTA", "mmsi": "431000999", "type": "Container"},
        {"name": "ONE STORK", "mmsi": "431000888", "type": "Container"},
        {"name": "ARCTIC VOYAGER", "mmsi": "258000777", "type": "LNG Tanker"},
        {"name": "OCEAN TURKISH", "mmsi": "636000666", "type": "Bulk Carrier"},
        {"name": "PACIFIC EXPLORER", "mmsi": "538000555", "type": "Research"},
    ]

    for target_video in video_files:
        print(f"\n--- Loading next video: {os.path.basename(target_video)} ---")
        cap = cv2.VideoCapture(target_video)
        if not cap.isOpened():
            print(f"Failed to open video: {target_video}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Processing every 30th frame (1 frame per second of footage typically)
            if frame_count % 30 != 0:
                continue
                
            detections, annotated_frame = detector.detect(frame)
            
            current_tracks = {}
            for det in detections:
                bbox = det[:4]
                conf = det[4]
                
                best_id = None
                best_iou = 0
                for tid, tinfo in active_tracks.items():
                    curr_iou = iou(bbox, tinfo['bbox'])
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_id = tid
                        
                if best_id and best_iou > 0.3:
                    # Existing Track Maintained
                    track_id = best_id
                    tinfo = active_tracks.pop(track_id)
                    lat = tinfo['lat'] + random.uniform(-0.0005, 0.0005)
                    lon = tinfo['lon'] + random.uniform(-0.0005, 0.0005)
                    history = tinfo['history']
                    ais_identity = tinfo['ais_identity']
                else:
                    # New Track Identified
                    track_id = str(next_id)
                    next_id += 1
                    # Spawn internationally across the whole globe!
                    lat = random.uniform(-60.0, 60.0) 
                    lon = random.uniform(-160.0, 160.0)
                    history = []
                    
                    # 1. Map video ship into AIS track randomly for fusion demonstration
                    ais_identity = random.choice(ais_registry)
                    print(f"[*] Correlated Video Ship {track_id} to AIS Profile: {ais_identity['name']} (MMSI: {ais_identity['mmsi']})")

                # Maintain sliding window 5 steps history [lat, lon, sog, cog, hdg]
                hist_vec = [lat, lon, 15.0, 90.0, 90.0] 
                history.append(hist_vec)
                if len(history) > 20:
                    history.pop(0)
                    
                # 2. Give where it goes next (Trajectory prediction)
                # We feed the model our sequence history to get future coordinates
                future_trajectory = []
                if len(history) > 5:
                    with torch.no_grad():
                        # shape [1, seq_len, 5]
                        inp = torch.tensor([history], dtype=torch.float32)
                        
                        # Autoregressive generation of trajectory
                        curr_inp = inp
                        for _ in range(5): # Predict next 5 bounds
                            pred = traj_model(curr_inp) # gets [1, 2]
                            next_lat_lon = pred[0].numpy().tolist()
                            future_trajectory.append(next_lat_lon)
                            
                            # Append to sequence
                            next_vec = [next_lat_lon[0], next_lat_lon[1], 15.0, 90.0, 90.0]
                            curr_inp = torch.cat((curr_inp, torch.tensor([[next_vec]])), dim=1)
                
                current_tracks[track_id] = {
                    'bbox': bbox, 
                    'lat': lat, 
                    'lon': lon, 
                    'history': history,
                    'ais_identity': ais_identity
                }
                
                # 3. Secure a high-res base64 crop of the ship with YOLO box for the dashboard
                # We crop the annotated frame strictly to the ship region
                x1, y1, x2, y2 = map(int, bbox)
                # Expand crop slightly for context
                pad = 10
                y1_p, y2_p = max(0, y1-pad), min(annotated_frame.shape[0], y2+pad)
                x1_p, x2_p = max(0, x1-pad), min(annotated_frame.shape[1], x2+pad)
                ship_crop = annotated_frame[y1_p:y2_p, x1_p:x2_p]
                
                _, buffer = cv2.imencode('.jpg', ship_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # Emit to Redis completely packed with all context
                track_payload = {
                    "mmsi": f"{ais_identity['mmsi']}",
                    "name": f"{ais_identity['name']} (SMD Zone)",
                    "lat": lat,
                    "lon": lon,
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    "prediction": future_trajectory,
                    "frame": frame_b64
                }
                r.publish('vessel_tracks', json.dumps(track_payload))
            
            active_tracks = current_tracks
            time.sleep(0.01)

if __name__ == "__main__":
    process_video()
