"""
BharatSign — Step 1: Data Collection
Run this script to record ISL gestures from your webcam.
Each team member records ~40 samples per sign.
Controls:
  Press 'S' to start recording a sign
  Press 'Q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

SIGNS = [
    "Hello", "Thank You", "Yes", "No", "Please",
    "Help", "Water", "Food", "Home", "Name",
    "Good", "Bad", "Stop", "Come", "Go",
    "I", "You", "Love", "India", "Emergency"
]

SAMPLES_PER_SIGN = 40
OUTPUT_FILE = "../data/raw/isl_landmarks.csv"

os.makedirs("../data/raw", exist_ok=True)

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def collect():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7)

    existing_data = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            existing_data = list(csv.reader(f))

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)

        if not existing_data:
            header = [f"x{i}" for i in range(21)] + \
                     [f"y{i}" for i in range(21)] + \
                     [f"z{i}" for i in range(21)] + ["label"]
            writer.writerow(header)
        else:
            for row in existing_data:
                writer.writerow(row)

        for sign in SIGNS:
            print(f"\n{'='*50}")
            print(f"  Next sign: '{sign}'")
            print(f"  Get ready... press 'S' when ready to record")
            print(f"{'='*50}")

            samples = 0
            recording = False

            while samples < SAMPLES_PER_SIGN:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                status_color = (0, 255, 0) if recording else (0, 165, 255)
                status_text = f"RECORDING: {samples}/{SAMPLES_PER_SIGN}" if recording else "Press S to start"

                cv2.putText(frame, f"Sign: {sign}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, status_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, "Q = quit | S = start recording", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if result.multi_hand_landmarks:
                    for hl in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                        if recording:
                            landmarks = extract_landmarks(hl)
                            writer.writerow(landmarks + [sign])
                            f.flush()
                            samples += 1
                            time.sleep(0.05)

                cv2.imshow("BharatSign - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\nCollection stopped early.")
                    return
                elif key == ord('s'):
                    recording = True

            print(f"  ✅ Collected {samples} samples for '{sign}'")
            recording = False
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ All done! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    collect()