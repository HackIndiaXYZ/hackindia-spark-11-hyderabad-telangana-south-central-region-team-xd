"""
BharatSign — Step 2: Train the Classifier
Run after collect_data.py has gathered enough samples.
Trains a Random Forest on hand landmarks.
Saves model + label encoder for use in the backend.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "../data/raw/isl_landmarks.csv"
MODEL_OUT = "../backend/model.pkl"
ENCODER_OUT = "../backend/label_encoder.pkl"

def train():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"   Total samples: {len(df)}")
    print(f"   Signs: {sorted(df['label'].unique())}")
    print(f"   Samples per sign:\n{df['label'].value_counts()}\n")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("🤖 Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs("../backend", exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)

    print(f"\n✅ Model saved to {MODEL_OUT}")
    print(f"✅ Label encoder saved to {ENCODER_OUT}")
    print(f"\n🚀 Classes: {list(le.classes_)}")

if __name__ == "__main__":
    train()