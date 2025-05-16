import pickle
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("[predict.py] Path patched")

try:
    with open("model/randomforest.pkl", "rb") as f:
        print("[predict.py] Loading model...")
        forest = pickle.load(f)
        print("[predict.py] Model loaded ✅")
except Exception as e:
    print("[ERROR] Failed to load forest:", e)
    forest = None


def predict_from_landmarks(landmarks):
    if len(landmarks) != 63:
        return None
    landmarks_array = np.array(landmarks).reshape(1, -1)
    return forest.forest_predict(landmarks_array)[0]