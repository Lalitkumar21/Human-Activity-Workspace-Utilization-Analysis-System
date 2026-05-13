import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import math
from datetime import datetime
import os

# 1. Configuration & Setup
# ==========================================
VIDEO_PATH = r'c:/ml/petrol-pump/with UI-design/petrol-pump.mp4'  

OUTPUT_PATH = 'output_video.mp4'
# Added a unique timestamp to filename to prevent PermissionError (e.g., if open in Excel)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f'activity_log_{timestamp_str}.csv'

# Load Model
model = YOLO('yolov8n.pt')

# ROIs (Adjust these coordinates based on your specific video perspective)
ROI_PUMP_LEFT = np.array([[380, 150], [480, 150], [480, 350], [380, 350]], np.int32)
ROI_PUMP_RIGHT = np.array([[550, 150], [680, 150], [680, 350], [550, 350]], np.int32)
ROIS = [ROI_PUMP_LEFT, ROI_PUMP_RIGHT]

# Tracking Parameters
track_history = {} 
MAX_HISTORY = 30  
STATIONARY_THRESHOLD = 20 
activity_logs = []

# 2. Helper Functions
# ==========================================
def calculate_movement(history):
    if len(history) < 2:
        return 0.0
    return math.dist(history[0], history[-1])

def check_in_roi(point, rois):
    for roi in rois:
        if cv2.pointPolygonTest(roi, point, False) >= 0:
            return True
    return False

# 3. Video Initialization
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH) # Fixed: Now uses VIDEO_PATH variable

import os

if not os.path.exists(VIDEO_PATH):
    print(f"FAILED: The file '{VIDEO_PATH}' does not exist in this directory.")
    print(f"Current Directory: {os.getcwd()}")
else:
    print(f"SUCCESS: Found '{VIDEO_PATH}'. Proceeding to open...")


if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0

# 4. Main Processing Loop
# ==========================================
print("Processing video... Press 'q' to stop early if running locally.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run tracking
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

    # Draw ROIs on frame
    for roi in ROIS:
        cv2.polylines(frame, [roi], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, "Pump Zone", (roi[0][0], roi[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            if int(cls) != 0: # Process only Person (Class 0)
                continue
                
            x1, y1, x2, y2 = map(int, box)
            # Use feet position (bottom-center) for accuracy
            bottom_center = (int((x1 + x2) / 2), y2)
            track_id = int(track_id)

            # Update track history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(bottom_center)
            
            if len(track_history[track_id]) > MAX_HISTORY:
                track_history[track_id].pop(0)

            # Logic Logic:
            # 1. Is the person in the pump zone?
            # 2. Are they stationary?
            is_in_workspace = check_in_roi(bottom_center, ROIS)
            movement_dist = calculate_movement(track_history[track_id])
            is_stationary = movement_dist < STATIONARY_THRESHOLD

            # Determine State
            if is_in_workspace and is_stationary:
                state, color = "Working", (0, 255, 0) # Green
            elif not is_stationary:
                state, color = "Moving", (0, 165, 255) # Orange
            else:
                state, color = "Idle", (0, 0, 255) # Red

            # Log data once per second
            if frame_count % fps == 0:
                activity_logs.append({
                    "Timestamp": current_time,
                    "Person_ID": track_id,
                    "Activity_State": state,
                    "In_ROI": is_in_workspace
                })

            # Draw Bounding Box and Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {state}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw movement trail
            if len(track_history[track_id]) > 1:
                pts = np.array(track_history[track_id], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(255, 255, 0), thickness=1)

    out.write(frame)

# 5. Clean Up & Save Data
# ==========================================
cap.release()
out.release()
cv2.destroyAllWindows()

if activity_logs:
    df = pd.DataFrame(activity_logs)
    try:
        df.to_csv(LOG_FILE, index=False)
        print(f"\nProcessing complete.")
        print(f"Video saved to: {OUTPUT_PATH}")
        print(f"Logs saved to: {LOG_FILE}")
    except PermissionError:
        # Fallback if the original file is locked
        alt_file = f"activity_log_fallback_{np.random.randint(1000)}.csv"
        df.to_csv(alt_file, index=False)
        print(f"\nPermission denied for {LOG_FILE}. Saved to {alt_file} instead.")
else:
    print("\nProcessing complete, but no activities were logged.")