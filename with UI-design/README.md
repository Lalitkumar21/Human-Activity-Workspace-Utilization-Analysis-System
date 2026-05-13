# MediaWire Petrol Pump Setup

This folder contains the MediaWire petrol pump tracking demo using YOLOv8 and MediaPipe.

## Files

- `station-model.py` — main Python script for video processing, YOLO detection, and logging.
- `requirements.txt` — Python dependencies required for MediaWire.
- `petrol-pump.mp4` — input video file used by the demo.
- `yolov8n.pt` — YOLOv8 model weights used for detection.

## Setup

1. Open a terminal in `c:\ml\petrol-pump\with UI-design`
2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the project

Run the main processing script:

```powershell
python station-model.py
```

The script will produce:

- `output_video.mp4` — annotated output video.
- `activity_log_<timestamp>.csv` — activity and posture log CSV.

## Troubleshooting

- If `station-model.py` cannot find `petrol-pump.mp4`, make sure the file is present in this folder.
- If `yolov8n.pt` is missing, download the YOLOv8 small weights and place the file here.
- If MediaPipe import fails, ensure the virtual environment is activated and packages installed.
