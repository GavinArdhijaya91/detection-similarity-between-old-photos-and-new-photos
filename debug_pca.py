import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Load pretrained
NPZ_PATH = "pretrained_eigenspace.npz"
data = np.load(NPZ_PATH)
PRETRAINED = {
    "mean_face": data['mean_face'],
    "eigenfaces": data['eigenfaces'],
    "singular_values": data['singular_values'],
}
print(f"Mean face min/max: {PRETRAINED['mean_face'].min()}, {PRETRAINED['mean_face'].max()}")
print(f"Eigenfaces min/max: {PRETRAINED['eigenfaces'].min()}, {PRETRAINED['eigenfaces'].max()}")

def preprocess_fastapi(img):
    # Simulated FastAPI preprocessing
    TARGET_SIZE = (128, 128)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float64) / 255.0
    return normalized

def preprocess_colab(img):
    # Simulated Colab preprocessing
    TARGET_SIZE = (128, 128)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized.flatten()

# Buat dummy image (random face)
img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
img2 = img1.copy()
img2[50:150, 50:150] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

f1_fastapi = preprocess_fastapi(img1).flatten()
f2_fastapi = preprocess_fastapi(img2).flatten()

print(f"\nFastAPI Inference Face min/max: {f1_fastapi.min()}, {f1_fastapi.max()}")

w1 = PRETRAINED["eigenfaces"] @ (f1_fastapi - PRETRAINED["mean_face"])
w2 = PRETRAINED["eigenfaces"] @ (f2_fastapi - PRETRAINED["mean_face"])

print(f"\nWeights W1 min/max/norm: {w1.min():.2f}, {w1.max():.2f}, {np.linalg.norm(w1):.2f}")
print(f"Weights W2 min/max/norm: {w2.min():.2f}, {w2.max():.2f}, {np.linalg.norm(w2):.2f}")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos = cosine_sim(w1, w2)
print(f"Cosine Eigenspace: {cos:.4f}")
print(f"Euc Dist: {np.linalg.norm(w1 - w2):.4f}")
print(f"Euc Sim: {1.0/(1.0 + np.linalg.norm(w1 - w2)):.4f}")
