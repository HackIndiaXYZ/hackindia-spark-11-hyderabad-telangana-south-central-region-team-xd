import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import time

SIGNS = [
    "Hello", "Thank You", "Yes", "No", "Please",
    "Help", "Water", "Food", "Home", "Name",
    "Good", "Bad", "Stop", "Come", "Go",
    "I", "You", "Love", "India", "Emergency"
]

SAMPLES_PER_SIGN = 40
OUTPUT_FILE = "../data/raw/isl_landmarks.csv"
MODEL_PATH = "../backend/hand_landmarker.task"

os.makedirs("../data/raw", exist_ok=True)

def extract_landmarks(hand_landmarks) -> list:
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def collect():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(21)] + \
                 [f"y{i}" for i in range(21)] + \
                 [f"z{i}" for i in range(21)] + ["label"]
        writer.writerow(header)

        for sign in SIGNS:
            print(f"\n{'='*50}")
            print(f"  Next sign: '{sign}'")
            print(f"  Press 'S' to start recording, 'Q' to quit")
            print(f"{'='*50}")

            samples = 0
            recording = False

            while samples < SAMPLES_PER_SIGN:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                status_color = (0, 255, 0) if recording else (0, 165, 255)
                status_text = f"RECORDING: {samples}/{SAMPLES_PER_SIGN}" if recording else "Press S to start"

                cv2.putText(frame, f"Sign: {sign}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, status_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, "Q = quit | S = start", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if result.hand_landmarks:
                    for hand in result.hand_landmarks:
                        for lm in hand:
                            h, w, _ = frame.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    if recording:
                        landmarks = extract_landmarks(result.hand_landmarks[0])
                        writer.writerow(landmarks + [sign])
                        f.flush()
                        samples += 1
                        time.sleep(0.05)

                cv2.imshow("BharatSign - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\nStopped early.")
                    return
                elif key == ord('s'):
                    recording = True

            print(f"  ✅ {samples} samples collected for '{sign}'")
            recording = False
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    collect()