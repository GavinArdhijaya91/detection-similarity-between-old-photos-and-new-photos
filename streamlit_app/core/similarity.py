"""
core/similarity.py
==================
Metrik kemiripan antara dua representasi wajah dalam eigenspace.

Konsep Aljabar Linear:
- Cosine Similarity: cos(θ) = (a·b)/(‖a‖‖b‖)
- Euclidean Distance: d = ‖a - b‖₂
- Normalized Cross-Correlation
"""

import numpy as np
from typing import Tuple, Dict


# ─────────────────────────────────────────────
# Threshold & Level
# ─────────────────────────────────────────────

THRESHOLDS = {
    "identical"   : 0.95,  # Hampir pasti sama, foto digital identik
    "very_similar": 0.85,  # Sangat mirip, kemungkinan besar orang sama
    "similar"     : 0.70,  # Mirip, bisa orang yang sama
    "uncertain"   : 0.55,  # Tidak pasti
    # di bawah 0.55 → dianggap berbeda
}

DECISION_THRESHOLD = 0.70  # Default threshold untuk "orang yang sama"


# ─────────────────────────────────────────────
# Metrik Kemiripan
# ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine Similarity antara dua vektor.
    
    cos(θ) = (a · b) / (‖a‖₂ · ‖b‖₂)
    
    Nilai:
    - 1.0  → vektor identik (sudut 0°)
    - 0.0  → ortogonal (sudut 90°)
    - -1.0 → berlawanan arah (sudut 180°)
    
    Keunggulan: tidak sensitif terhadap magnitude (skala),
    hanya melihat arah vektor — cocok untuk eigenspace projection.
    """
    a_flat = a.flatten().astype(float)
    b_flat = b.flatten().astype(float)
    
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    dot_product = np.dot(a_flat, b_flat)
    similarity = dot_product / (norm_a * norm_b)
    
    # Clamp ke [-1, 1] untuk menghindari floating point error
    return float(np.clip(similarity, -1.0, 1.0))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean Distance antara dua vektor.
    
    d = ‖a - b‖₂ = sqrt(Σ(aᵢ - bᵢ)²)
    
    Catatan: makin kecil jarak, makin mirip.
    Perlu dinormalisasi untuk interpretasi yang bermakna.
    """
    a_flat = a.flatten().astype(float)
    b_flat = b.flatten().astype(float)
    return float(np.linalg.norm(a_flat - b_flat))


def normalized_euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Ubah Euclidean Distance menjadi similarity score [0, 1].
    
    similarity = 1 / (1 + distance)
    
    - distance = 0 → similarity = 1.0 (identik)
    - distance → ∞ → similarity → 0.0
    """
    dist = euclidean_distance(a, b)
    return float(1.0 / (1.0 + dist))


def structural_similarity_pixels(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity (SSIM) sederhana antara dua gambar yang sudah di-resize sama.
    Dihitung dari pixel langsung (bukan eigenspace).
    
    SSIM = (2μ₁μ₂ + C₁)(2σ₁₂ + C₂) / (μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂)
    """
    img1 = img1.flatten().astype(float)
    img2 = img2.flatten().astype(float)
    
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1    = np.mean(img1)
    mu2    = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    numerator   = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
    
    ssim = numerator / denominator if denominator != 0 else 0.0
    return float(np.clip(ssim, 0.0, 1.0))


# ─────────────────────────────────────────────
# Composite Score & Keputusan
# ─────────────────────────────────────────────

def compute_all_metrics(
    weights1: np.ndarray,
    weights2: np.ndarray,
    face1_pixel: np.ndarray,
    face2_pixel: np.ndarray,
) -> Dict[str, float]:
    """
    Hitung semua metrik kemiripan sekaligus.
    
    Parameters
    ----------
    weights1, weights2 : Proyeksi eigenspace (dari project_to_eigenspace)
    face1_pixel, face2_pixel : Gambar wajah preprocessed (pixel)
    """
    cos_sim    = cosine_similarity(weights1, weights2)
    euc_dist   = euclidean_distance(weights1, weights2)
    euc_sim    = normalized_euclidean_similarity(weights1, weights2)
    ssim_score = structural_similarity_pixels(face1_pixel, face2_pixel)
    pixel_cos  = cosine_similarity(face1_pixel, face2_pixel)
    
    # Composite score (weighted average)
    # Cosine similarity di eigenspace adalah metrik paling relevan untuk PCA
    # Clamp negatif ke 0 — cosine negatif berarti arah berlawanan (sangat berbeda)
    composite = (
        0.45 * max(0.0, cos_sim) +
        0.25 * euc_sim +
        0.20 * ssim_score +
        0.10 * max(0.0, pixel_cos)
    )
    
    return {
        "cosine_similarity_eigenspace" : round(cos_sim, 4),
        "euclidean_distance_eigenspace": round(euc_dist, 4),
        "euclidean_similarity_norm"    : round(euc_sim, 4),
        "ssim_pixel"                   : round(ssim_score, 4),
        "cosine_similarity_pixel"      : round(pixel_cos, 4),
        "composite_score"              : round(composite, 4),
    }


def make_decision(
    metrics: Dict[str, float],
    threshold: float = DECISION_THRESHOLD,
) -> Dict:
    """
    Buat keputusan apakah dua wajah adalah orang yang sama.
    
    Parameters
    ----------
    metrics   : dict dari compute_all_metrics()
    threshold : ambang batas composite score
    
    Returns
    -------
    dict berisi verdict, confidence, level, dan reasoning
    """
    score = metrics["composite_score"]
    cos   = metrics["cosine_similarity_eigenspace"]
    
    is_same = score >= threshold
    
    # Tentukan level kepercayaan
    if cos >= THRESHOLDS["identical"]:
        level      = "Identik"
        confidence = "Sangat Tinggi"
        color      = "#00ff88"
    elif cos >= THRESHOLDS["very_similar"]:
        level      = "Sangat Mirip"
        confidence = "Tinggi"
        color      = "#44cc88"
    elif cos >= THRESHOLDS["similar"]:
        level      = "Mirip"
        confidence = "Sedang"
        color      = "#ffcc00"
    elif cos >= THRESHOLDS["uncertain"]:
        level      = "Kurang Mirip"
        confidence = "Rendah"
        color      = "#ff8844"
    else:
        level      = "Tidak Mirip"
        confidence = "Sangat Rendah"
        color      = "#ff4444"
    
    # Reasoning matematis
    reasoning = []
    if cos >= 0.70:
        reasoning.append(f"Cosine similarity eigenspace tinggi ({cos:.2%}) → sudut kecil antara proyeksi kedua wajah")
    else:
        reasoning.append(f"Cosine similarity eigenspace rendah ({cos:.2%}) → proyeksi eigenspace berbeda signifikan")
    
    euc = metrics["euclidean_distance_eigenspace"]
    if euc < 1.0:
        reasoning.append(f"Jarak Euclidean di eigenspace kecil ({euc:.3f}) → representasi vektor berdekatan")
    else:
        reasoning.append(f"Jarak Euclidean di eigenspace besar ({euc:.3f}) → representasi vektor berjauhan")
    
    ssim = metrics["ssim_pixel"]
    reasoning.append(f"SSIM pixel: {ssim:.2%} → kemiripan struktur intensitas gambar")
    
    return {
        "is_same_person"   : is_same,
        "verdict"          : "[SAMA] Orang yang Sama" if is_same else "[BEDA] Orang yang Berbeda",
        "verdict_display"  : "\u2705 Orang yang Sama" if is_same else "\u274c Orang yang Berbeda",
        "verdict_en"       : "Same Person" if is_same else "Different Person",
        "score"            : score,
        "level"            : level,
        "confidence"       : confidence,
        "color"            : color,
        "threshold_used"   : threshold,
        "reasoning"        : reasoning,
        "metrics"          : metrics,
    }
