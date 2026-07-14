"""
BharatSign — Backend API
FastAPI server that:
  - Receives base64 webcam frames
  - Runs MediaPipe + classifier
  - Returns predicted sign + confidence
  - /tts endpoint converts text to speech (base64 audio)
  - /signs endpoint returns all supported signs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mediapipe as mp
import numpy as np
import pickle
import base64
import cv2
import io
import os
from gtts import gTTS

app = FastAPI(title="BharatSign API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model.pkl")
ENCODER_PATH = os.path.join(BASE, "label_encoder.pkl")

model = None
label_encoder = None

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"✅ Model loaded. Classes: {list(label_encoder.classes_)}")
else:
    print("⚠️  Model not found. Run train_model.py first.")

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

class FrameRequest(BaseModel):
    image: str

class PredictResponse(BaseModel):
    sign: str
    confidence: float
    hand_detected: bool
    all_probs: dict

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"

class TTSResponse(BaseModel):
    audio_base64: str

def decode_frame(b64_image: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_image)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

def extract_landmarks(hand_landmarks) -> list:
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

@app.get("/")
def root():
    return {
        "status": "BharatSign API running",
        "model_loaded": model is not None,
        "signs_supported": list(label_encoder.classes_) if label_encoder else []
    }

@app.get("/signs")
def get_signs():
    if label_encoder is None:
        return {"signs": []}
    return {"signs": list(label_encoder.classes_)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: FrameRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    frame = decode_frame(req.image)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    if not result.multi_hand_landmarks:
        return PredictResponse(
            sign="",
            confidence=0.0,
            hand_detected=False,
            all_probs={}
        )

    landmarks = extract_landmarks(result.multi_hand_landmarks[0])
    X = np.array(landmarks).reshape(1, -1)

    probs = model.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    pred_sign = label_encoder.classes_[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = {
        label_encoder.classes_[i]: round(float(p), 3)
        for i, p in enumerate(probs)
    }

    return PredictResponse(
        sign=pred_sign if confidence > 0.6 else "",
        confidence=confidence,
        hand_detected=True,
        all_probs=all_probs
    )

@app.post("/tts", response_model=TTSResponse)
def text_to_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    lang_map = {"en": "en", "hi": "hi", "te": "te"}
    lang = lang_map.get(req.lang, "en")

    tts = gTTS(text=req.text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return TTSResponse(audio_base64=audio_b64)

@app.get("/health")
def health():
    return {"status": "ok", "model": model is not None}