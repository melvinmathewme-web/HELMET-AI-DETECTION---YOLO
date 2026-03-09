# AI-Based Helmet Detection System

This project is a real-time AI Helmet Detection system using YOLOv8, OpenCV, and PyTorch. It captures a video stream (from a webcam or file), detects whether people are wearing helmets, and draws colored bounding boxes around them. It features an integrated audio alarm system for violations.

## Features
*   **Dual-Class Tracking:** Accurately distinguishes between `head` (violator) and `helmet` (compliant) classes.
*   **Live Violation Alarms:** Sounds an audio buzzer when an unhelmeted rider is detected on Windows.
*   **Dynamic UI:** Draws bounding boxes (Green for Helmet, Red for No Helmet) and active violation counters.

## Prerequisites
- Python 3.9+ installed.
- A webcam connected to your computer for the live demo.

## Setup Instructions

**1. Open a terminal in this directory**
You can use Command Prompt or PowerShell.

**2. Install Required Libraries**
Install YOLOv8 (Ultralytics), OpenCV, and PyTorch by running:
```bash
pip install -r requirements.txt
```

**3. Run the Live Webcam System**
Once your webcam is connected, simply run:
```bash
python main.py
```

## How to Test with Recorded Video Files
You can easily swap the video source without editing code by using terminal arguments.

```bash
# Test a local video file
python main.py --source motorcycle_test.mp4

# Change the AI confidence threshold (default is 0.5)
python main.py --confidence 0.70
```

## Stopping the Program
Press the `q` key on your keyboard while the video window is selected to stop the program safely.
