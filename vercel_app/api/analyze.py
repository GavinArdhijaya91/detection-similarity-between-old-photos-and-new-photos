"""
api/analyze.py
==============
Vercel Python Serverless Function
Menerima dua gambar base64, jalankan PCA/SVD, return hasil JSON.
"""

import json
import base64
import sys
import os
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# ─────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────
TARGET_SIZE = (128, 128)
HAAR_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_image(image_b64: str) -> np.ndarray:
    """Decode base64 image → grayscale → resize → normalize."""
    img_bytes = base64.b64decode(image_b64.split(",")[-1])
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(pil_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    
    # Deteksi wajah
    face_cascade = cv2.CascadeClassifier(HAAR_FRONTAL)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    
    face_detected = len(faces) > 0
    if face_detected:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(min(w, h) * 0.2)
        H, W = gray.shape
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
        gray = gray[y1:y2, x1:x2]
    
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float64) / 255.0
    
    return normalized, face_detected


# ─────────────────────────────────────────────
# PCA / SVD Core
# ─────────────────────────────────────────────

def run_pca_svd(face1: np.ndarray, face2: np.ndarray):
    """Run full PCA/SVD pipeline on two faces."""
    f1 = face1.flatten()
    f2 = face2.flatten()
    
    # SVD per gambar
    U1, S1, Vt1 = np.linalg.svd(face1, full_matrices=False)
    U2, S2, Vt2 = np.linalg.svd(face2, full_matrices=False)
    
    # Stack → eigenfaces
    images_stack = np.stack([f1, f2], axis=0)
    mean_face = np.mean(images_stack, axis=0)
    centered = images_stack - mean_face
    
    U_joint, S_joint, Vt_joint = np.linalg.svd(centered, full_matrices=False)
    
    n_comp = min(2, len(S_joint))
    eigenfaces = Vt_joint[:n_comp]
    eigenvalues = (S_joint[:n_comp] ** 2) / 2
    
    # Proyeksi
    w1 = eigenfaces @ (f1 - mean_face)
    w2 = eigenfaces @ (f2 - mean_face)
    
    # Similarity metrics
    def cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))
    
    def euc_dist(a, b):
        return float(np.linalg.norm(a - b))
    
    def ssim_simple(img1, img2):
        a, b = img1.flatten(), img2.flatten()
        C1, C2 = 0.01**2, 0.03**2
        mu1, mu2 = np.mean(a), np.mean(b)
        s1, s2 = np.var(a), np.var(b)
        s12 = np.mean((a - mu1) * (b - mu2))
        num = (2*mu1*mu2 + C1) * (2*s12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
        return float(np.clip(num/den if den != 0 else 0, 0, 1))
    
    cos_eigen = cosine_sim(w1, w2)
    euc_d = euc_dist(w1, w2)
    euc_sim = 1.0 / (1.0 + euc_d)
    ssim = ssim_simple(face1, face2)
    cos_pixel = cosine_sim(f1, f2)
    composite = 0.45*max(0, cos_eigen) + 0.25*euc_sim + 0.20*ssim + 0.10*max(0, cos_pixel)
    
    # SVD info
    def sv_info(S):
        total = np.sum(S**2)
        return [
            {"rank": int(i+1), "value": float(S[i]), "variance_pct": float(S[i]**2/total*100)}
            for i in range(min(15, len(S)))
        ]
    
    return {
        "metrics": {
            "cosine_similarity_eigenspace": round(cos_eigen, 4),
            "euclidean_distance_eigenspace": round(euc_d, 4),
            "euclidean_similarity_norm": round(euc_sim, 4),
            "ssim_pixel": round(ssim, 4),
            "cosine_similarity_pixel": round(cos_pixel, 4),
            "composite_score": round(composite, 4),
        },
        "eigenvalues": [float(v) for v in eigenvalues],
        "weights_face1": [float(v) for v in w1],
        "weights_face2": [float(v) for v in w2],
        "singular_values_face1": sv_info(S1),
        "singular_values_face2": sv_info(S2),
        "singular_values_joint": [float(v) for v in S_joint],
        "mean_face_shape": list(mean_face.shape),
    }


def make_decision(composite_score: float, cos_eigen: float, threshold: float = 0.70):
    """Buat keputusan dari composite score."""
    is_same = composite_score >= threshold
    
    if cos_eigen >= 0.95:
        level = "Identik"; confidence = "Sangat Tinggi"; color = "#10b981"
    elif cos_eigen >= 0.85:
        level = "Sangat Mirip"; confidence = "Tinggi"; color = "#22c55e"
    elif cos_eigen >= 0.70:
        level = "Mirip"; confidence = "Sedang"; color = "#f59e0b"
    elif cos_eigen >= 0.55:
        level = "Kurang Mirip"; confidence = "Rendah"; color = "#f97316"
    else:
        level = "Tidak Mirip"; confidence = "Sangat Rendah"; color = "#ef4444"
    
    return {
        "is_same_person": bool(is_same),
        "verdict": "Orang yang Sama" if is_same else "Orang yang Berbeda",
        "verdict_icon": "✅" if is_same else "❌",
        "level": level,
        "confidence": confidence,
        "color": color,
        "threshold_used": threshold,
    }


# ─────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────

def handler(request, response):
    """Vercel Python serverless handler."""
    
    if request.method == "OPTIONS":
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response.send(200, "")
    
    if request.method != "POST":
        return response.send(405, json.dumps({"error": "Method not allowed"}))
    
    try:
        body = json.loads(request.body)
        image1_b64 = body.get("image1")
        image2_b64 = body.get("image2")
        threshold  = float(body.get("threshold", 0.70))
        
        if not image1_b64 or not image2_b64:
            return response.send(400, json.dumps({"error": "Kedua gambar diperlukan"}))
        
        face1, detected1 = preprocess_image(image1_b64)
        face2, detected2 = preprocess_image(image2_b64)
        
        result = run_pca_svd(face1, face2)
        decision = make_decision(
            result["metrics"]["composite_score"],
            result["metrics"]["cosine_similarity_eigenspace"],
            threshold,
        )
        
        output = {
            "success": True,
            "decision": decision,
            "metrics": result["metrics"],
            "math_data": {
                "eigenvalues": result["eigenvalues"],
                "weights_face1": result["weights_face1"],
                "weights_face2": result["weights_face2"],
                "singular_values_face1": result["singular_values_face1"],
                "singular_values_face2": result["singular_values_face2"],
                "singular_values_joint": result["singular_values_joint"],
            },
            "preprocessing": {
                "face1_detected": detected1,
                "face2_detected": detected2,
                "image_size": f"{TARGET_SIZE[0]}×{TARGET_SIZE[1]}",
            },
        }
        
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Content-Type"] = "application/json"
        return response.send(200, json.dumps(output))
    
    except Exception as e:
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response.send(500, json.dumps({"error": str(e), "success": False}))
