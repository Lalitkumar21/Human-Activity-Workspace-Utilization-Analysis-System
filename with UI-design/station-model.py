import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import math
import tempfile
from datetime import datetime

# ==========================================
# 1. Page Configuration & Caching
# ==========================================
st.set_page_config(page_title="Workspace Activity Analyzer", layout="wide")

# Cache the model so it doesn't reload every time you click a button
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# ==========================================
# 2. Helper Functions
# ==========================================
def calculate_movement(history):
    if len(history) < 2: return 0.0
    return math.dist(history[0], history[-1])

def check_in_roi(point, rois):
    for roi in rois:
        if cv2.pointPolygonTest(roi, point, False) >= 0:
            return True
    return False

# ==========================================
# 3. Sidebar Configuration (User Inputs)
# ==========================================
st.sidebar.title("⚙️ System Settings")
st.sidebar.markdown("Upload a video to analyze activity states.")

uploaded_file = st.sidebar.file_uploader("Upload CCTV Footage (.mp4)", type=['mp4', 'avi', 'mov'])

st.sidebar.subheader("Logic Thresholds")
# Let the user adjust the "Stationary" threshold dynamically
STATIONARY_THRESHOLD = st.sidebar.slider("Stationary Threshold (pixels)", min_value=5, max_value=50, value=20)
MAX_HISTORY = st.sidebar.slider("Movement History (frames)", min_value=10, max_value=60, value=30)

# ==========================================
# 4. Main Application UI
# ==========================================
st.title("📊 Workspace Utilization & Activity Tracker")
st.markdown("This system tracks human presence, defines Regions of Interest (ROI), and classifies activity as **Working**, **Moving**, or **Idle**.")

if uploaded_file is not None:
    # Save uploaded file to a temporary location so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("🚀 Start Processing Video"):
        
        # UI Placeholders for real-time updates
        col1, col2 = st.columns([3, 1])
        with col1:
            stframe = st.empty()  # Placeholder for the video frame
        with col2:
            st_metrics = st.empty() # Placeholder for live stats
            st_logs = st.empty()    # Placeholder for recent logs
            
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Hardcoded ROIs for the gas station scenario
        ROI_PUMP_LEFT = np.array([[380, 150], [480, 150], [480, 350], [380, 350]], np.int32)
        ROI_PUMP_RIGHT = np.array([[550, 150], [680, 150], [680, 350], [550, 350]], np.int32)
        ROIS = [ROI_PUMP_LEFT, ROI_PUMP_RIGHT]

        track_history = {}
        activity_logs = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")

            # Run Tracker
            results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

            # Draw ROIs
            for roi in ROIS:
                cv2.polylines(frame, [roi], isClosed=True, color=(255, 0, 0), thickness=2)

            current_working = 0
            current_idle = 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    if int(cls) != 0: continue # Only process persons
                        
                    x1, y1, x2, y2 = map(int, box)
                    bottom_center = (int((x1 + x2) / 2), y2)
                    track_id = int(track_id)
                    
                    if track_id not in track_history: track_history[track_id] = []
                    track_history[track_id].append(bottom_center)
                    if len(track_history[track_id]) > MAX_HISTORY: track_history[track_id].pop(0)

                    # Logic Engine
                    is_in_workspace = check_in_roi(bottom_center, ROIS)
                    movement_distance = calculate_movement(track_history[track_id])
                    is_stationary = movement_distance < STATIONARY_THRESHOLD

                    if is_in_workspace and is_stationary:
                        state = "Working"
                        color = (0, 255, 0)
                        current_working += 1
                    elif not is_in_workspace and not is_stationary:
                        state = "Moving"
                        color = (0, 165, 255)
                    else:
                        state = "Idle"
                        color = (0, 0, 255)
                        current_idle += 1

                    # Logging (once per second)
                    if frame_count % fps == 0:
                        activity_logs.append({
                            "Timestamp": current_time,
                            "Person_ID": track_id,
                            "State": state
                        })

                    # Draw graphics
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {track_id} {state}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert BGR (OpenCV) to RGB (Streamlit) for correct color display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update Live Metrics in the sidebar column
            st_metrics.markdown(f"### Live Status\n* **People Working:** {current_working}\n* **People Idle:** {current_idle}")

        cap.release()
        
        st.success("✅ Video Processing Complete!")
        
        # Display Data and Provide Download
        if activity_logs:
            df = pd.DataFrame(activity_logs)
            st.subheader("Activity Log Data")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Activity Logs as CSV",
                data=csv,
                file_name='activity_logs.csv',
                mime='text/csv',
            )
else:
    st.info("Please upload a video file from the sidebar to begin.")
