import cv2
import numpy as np
import threading
import winsound
import datetime
import os
import csv
import time
from ultralytics import YOLO

class SmartSurveillanceSystem:
    def __init__(self, camera_index=0):
        # Configuration
        self.CONFIDENCE_THRESHOLD = 0.6
        self.MIN_MOTION_AREA = 5000
        self.SCREENSHOT_COOLDOWN = 5.0  
        self.ALERT_GRACE_PERIOD = 5.0  
        
        # System State
        self.is_recording = False
        self.alarm_active = False
        self.last_screenshot_time = 0.0
        self.alert_active_until = 0.0  
        self.video_writer = None
        self.bg_frame = None  
        
        # Setup Environment
        self._setup_directories()
        self._init_logger()
        
        # Initialize Models & Hardware
        print("[INFO] Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")
        
        print("[INFO] Initializing camera...")
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be initialized.")

    def _setup_directories(self):
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)

    def _init_logger(self):
        self.log_file = "event_log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Event"])

    def log_event(self, event_text):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event_text])
        print(f"[{timestamp}] {event_text}")

    # --- NEW AUDIO LOGIC ---
    def trigger_long_alarm(self):
        """Plays a continuous, loud alarm while a person is detected."""
        def sound():
            while self.alarm_active:
                # 2500Hz is loud/piercing. 2000ms makes it a long, continuous tone.
                winsound.Beep(2500, 2000) 
                
        if not self.alarm_active:
            self.alarm_active = True
            threading.Thread(target=sound, daemon=True).start()

    def trigger_safe_beeps(self):
        """Plays 3 short, friendly beeps when the area is clear."""
        def sound():
            # 1500Hz is a slightly lower, less aggressive pitch
            for _ in range(3):
                winsound.Beep(1500, 250)
                time.sleep(0.1)
                
        threading.Thread(target=sound, daemon=True).start()
    # -----------------------

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.bg_frame is None:
            self.bg_frame = gray.astype("float")
            return False

        cv2.accumulateWeighted(gray, self.bg_frame, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg_frame))
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > self.MIN_MOTION_AREA:
                return True
        return False

    def process_frame(self, frame):
        person_detected = False
        motion = self.detect_motion(frame)

        if motion:
            results = self.model(frame, verbose=False)[0]
            for box in results.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > self.CONFIDENCE_THRESHOLD:
                    person_detected = True
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"PERSON {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return person_detected, frame

    def run(self):
        print("[INFO] System running. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                current_time = time.time()
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                person_detected, frame = self.process_frame(frame)

                if person_detected:
                    self.alert_active_until = current_time + self.ALERT_GRACE_PERIOD

                in_alert_state = current_time < self.alert_active_until

                # --- STATE MACHINE HANDLING ---
                if in_alert_state:
                    cv2.putText(frame, "ALERT: PERSON DETECTED!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Start the long alarm loop
                    self.trigger_long_alarm()

                    if not self.is_recording:
                        self.is_recording = True
                        filename = f"recordings/REC_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))
                        self.log_event("Recording Started")

                    if (current_time - self.last_screenshot_time) > self.SCREENSHOT_COOLDOWN:
                        screenshot_name = f"screenshots/SNAP_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot_name, frame)
                        self.last_screenshot_time = current_time
                        self.log_event("Screenshot Captured")

                else:
                    cv2.putText(frame, "SYSTEM ARMED", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Transition from Danger back to Safe
                    if self.alarm_active:
                        self.alarm_active = False          # Stops the long alarm loop
                        self.trigger_safe_beeps()          # Triggers the 3 short beeps
                    
                    if self.is_recording:
                        self.is_recording = False
                        if self.video_writer:
                            self.video_writer.release()
                        self.log_event("Recording Stopped")

                # ------------------------------

                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)

                cv2.putText(frame, timestamp_str, (10, 460), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("AI Smart Surveillance", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
        finally:
            self.cleanup()

    def cleanup(self):
        print("[INFO] Cleaning up resources...")
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SmartSurveillanceSystem()
    system.run()