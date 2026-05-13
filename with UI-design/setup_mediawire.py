import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(BASE_DIR, "requirements.txt")
VIDEO_FILE = os.path.join(BASE_DIR, "petrol-pump.mp4")
MODEL_FILE = os.path.join(BASE_DIR, "yolov8n.pt")
SCRIPT_FILE = os.path.join(BASE_DIR, "station-model.py")


def check_files():
    missing = []
    for path, name in [(VIDEO_FILE, "video file"), (MODEL_FILE, "YOLO weights"), (SCRIPT_FILE, "script")]:
        if not os.path.isfile(path):
            missing.append((name, path))
    return missing


def print_status():
    print("MediaWire Petrol Pump Setup")
    print("===========================\n")
    print(f"Working directory: {BASE_DIR}")
    print(f"Python executable: {sys.executable}\n")

    missing = check_files()
    if missing:
        print("Missing required files:")
        for name, path in missing:
            print(f" - {name}: {path}")
        print("\nPlease add the missing files before running the demo.")
    else:
        print("All required files are present.")
        print(f" - Video: {VIDEO_FILE}")
        print(f" - Model: {MODEL_FILE}")
        print(f" - Script: {SCRIPT_FILE}\n")
        print("To install dependencies:")
        print("  python -m pip install -r requirements.txt")
        print("To run the demo:")
        print("  python station-model.py")


def install_dependencies():
    if not os.path.isfile(REQUIREMENTS):
        print(f"requirements.txt not found at {REQUIREMENTS}")
        sys.exit(1)

    print("Installing dependencies from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS])
    print("Dependencies installed successfully.")


if __name__ == "__main__":
    if "--install" in sys.argv or "install" in sys.argv:
        install_dependencies()
    else:
        print_status()
