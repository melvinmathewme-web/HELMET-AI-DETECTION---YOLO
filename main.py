import cv2
import argparse
import sys
from ultralytics import YOLO
import math
import os
import time
import platform
import threading

if platform.system() == "Windows":
    import winsound

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Helmet Detection System")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file")
    parser.add_argument("--weights", type=str, default="best_v2.pt", help="YOLO trained model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum detection confidence")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle webcam source (0) or video file ('video.mp4')
    if args.source.isdigit():
        source = int(args.source)
    else:
        # It's a file path! Keep it as a string but make it absolute just like the weights
        script_dir = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(script_dir, args.source)
        
        if not os.path.exists(source):
            print(f"❌ Error: Could not find video file at {source}")
            sys.exit(1)

    # Make the weights path absolute based on where the python script is located
    # This prevents 'file not found' errors if the user runs the script from a different folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, args.weights)

    print(f"Loading YOLO Model using weights: {weights_path}...")
    try:
        model = YOLO(weights_path)
        print("✅ YOLO Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        print("Please ensure your 'best.pt' file exists in this directory.")
        sys.exit(1)

    class_names = model.names
    
    print(f"Opening Video Source: {source}...")
    
    # On Windows, using the DirectShow backend (cv2.CAP_DSHOW) fixes the "MSMF" camera crash error!
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ Cannot open camera or video stream.")
        return

    print("✅ Video Stream Started! Press 'q' on your keyboard to exit.")

    last_beep_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("End of stream or disconnected.")
            break

        # Run inference. `stream=True` makes it more memory efficient for live video
        results = model(frame, stream=True, conf=args.confidence)
        
        # We will track active violations visible in the CURRENT frame
        current_violations = 0

        for r in results:
            for box in r.boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Confidence score
                confidence = math.ceil((box.conf[0] * 100)) / 100
                
                # Class mapping
                cls_id = int(box.cls[0])
                label = class_names[cls_id].lower()

                # The new motorcycle helmet model uses {0: 'helmet', 1: 'head'}
                if label == "with helmet" or label == "helmet":
                    color = (0, 255, 0) # Green for Helmet
                    text = f"Helmet {confidence}"
                elif label == "without helmet" or label == "head" or label == "no helmet":
                    color = (0, 0, 255) # Red for No Helmet
                    text = f"No Helmet {confidence}"
                    current_violations += 1
                else:
                    continue # Ignore generic "person" boxes to just focus on the head/helmet

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Render beautiful labels with backgrounds
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the number of violations actively in frame
        status_y = 50
        cv2.putText(frame, "HELMET DETECTION SYSTEM", (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(frame, "HELMET DETECTION SYSTEM", (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 144, 30), 2) # Orange styling
        
        # Display violation warning dynamicallyq
        if current_violations > 0:
            warning_text = f"WARNING! Active Violations: {current_violations}"
            cv2.putText(frame, warning_text, (20, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # Red
            
            # Sound the buzzer alert natively on Windows (with a 1-second cooldown so it doesn't freeze the video)
            if platform.system() == "Windows":
                current_time = time.time()
                if current_time - last_beep_time > 1.0:
                    threading.Thread(target=winsound.Beep, args=(1000, 300), daemon=True).start()
                    last_beep_time = current_time
                    
        else:
            cv2.putText(frame, "Status: Safe", (20, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Green
            
        # Display output
        cv2.imshow("Real-Time Helmet Detection", frame)

        # Listen for the 'q' key to quit. 
        # Using 30ms delay to make pre-recorded videos play smoothly at around 30 FPS instead of instantly flashing
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
