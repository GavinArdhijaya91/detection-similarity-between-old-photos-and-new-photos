import numpy as np
from typing import Dict, Any

DECISION_THRESHOLD = 0.68

THRESHOLDS = {
    "identical"   : 0.95,
    "very_similar": 0.85,
    "similar"     : 0.70,
    "uncertain"   : 0.55,
}

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(float)
    b_flat = b.flatten().astype(float)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a_flat, b_flat) / (norm_a * norm_b), -1.0, 1.0))

def custom_weighted_cosine_sim(w1: np.ndarray, w2: np.ndarray) -> float:
    weights = np.ones_like(w1)
    if len(weights) > 3:
        weights[:3] = 0.2
    if len(weights) > 15:
        weights[15:] = 0.5
        
    w1_scaled = w1 * weights
    w2_scaled = w2 * weights
    return cosine_similarity(w1_scaled, w2_scaled)

def ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index (SSIM) roughly for pixel-based comparison.
    """
    a, b = img1.flatten(), img2.flatten()
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a), np.var(b)
    s12 = np.mean((a - mu1) * (b - mu2))
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return float(np.clip(num / den if den != 0 else 0, 0, 1))

def compute_all_metrics(
    weights1: np.ndarray,
    weights2: np.ndarray,
    face1_display: np.ndarray,
    face2_display: np.ndarray,
    S_joint: np.ndarray,
    penalty_factor: float = 0.05,
    weights1_lbp: np.ndarray = None,
    weights2_lbp: np.ndarray = None,
    weights1_hog: np.ndarray = None,
    weights2_hog: np.ndarray = None,
    alpha: float = 0.35,
    beta: float = 0.50,
    gamma: float = 0.15,
) -> Dict[str, float]:
    """
    Menghitung semua metrik kemiripan.

    Mode Pixel-only (backward compatible):
      composite = cos_pixel_weighted * exp(-penalty * d_euc)

    Mode LBP+HOG+Pixel Fusion:
      score_lbp = cos_lbp * exp(-penalty * d_lbp)
      score_hog = cos_hog * exp(-penalty * d_hog)
      score_pix = cos_pix * exp(-penalty * d_pix)
      composite = (alpha*score_lbp + beta*score_hog + gamma*score_pix) / (alpha+beta+gamma)

    Default bobot: alpha=0.35 (LBP), beta=0.50 (HOG), gamma=0.15 (Pixel)
    """
    prio = np.ones_like(weights1)
    if len(prio) > 3:
        prio[:3] = 0.2
    if len(prio) > 15:
        prio[15:] = 0.5

    w1s, w2s  = weights1 * prio, weights2 * prio
    cos_eigen = cosine_similarity(w1s, w2s)
    euc_d     = float(np.linalg.norm(w1s - w2s))
    euc_sim   = float(np.exp(-penalty_factor * euc_d))
    score_pix = float(max(0, cos_eigen)) * euc_sim

    ssim      = ssim_simple(face1_display, face2_display)
    cos_pixel = cosine_similarity(face1_display.flatten(), face2_display.flatten())

    result = {
        "cosine_similarity_eigenspace" : round(cos_eigen, 4),
        "euclidean_distance_eigenspace": round(euc_d, 4),
        "euclidean_similarity_norm"    : round(euc_sim, 4),
        "ssim_pixel"                   : round(ssim, 4),
        "cosine_similarity_pixel"      : round(cos_pixel, 4),
        "feature_mode"                 : "pixel",
    }

    is_fusion = (
        weights1_lbp is not None and weights2_lbp is not None and
        weights1_hog is not None and weights2_hog is not None
    )

    if is_fusion:
        cos_lbp   = float(cosine_similarity(weights1_lbp, weights2_lbp))
        d_lbp     = float(np.linalg.norm(weights1_lbp - weights2_lbp))
        score_lbp = float(max(0, cos_lbp)) * float(np.exp(-penalty_factor * d_lbp))

        cos_hog   = float(cosine_similarity(weights1_hog, weights2_hog))
        d_hog     = float(np.linalg.norm(weights1_hog - weights2_hog))
        score_hog = float(max(0, cos_hog)) * float(np.exp(-penalty_factor * d_hog))

        total_w   = alpha + beta + gamma
        composite = (alpha * score_lbp + beta * score_hog + gamma * score_pix) / total_w

        result.update({
            "cosine_lbp"     : round(cos_lbp, 4),
            "score_lbp"      : round(score_lbp, 4),
            "cosine_hog"     : round(cos_hog, 4),
            "score_hog"      : round(score_hog, 4),
            "score_pix"      : round(score_pix, 4),
            "composite_score": round(composite, 4),
            "feature_mode"   : "fusion",
        })
    else:
        result["composite_score"] = round(score_pix, 4)

    return result



def make_decision(
    metrics: Dict[str, float],
    threshold: float = DECISION_THRESHOLD,
) -> Dict[str, Any]:
    score   = metrics["composite_score"]
    cos     = metrics["cosine_similarity_eigenspace"]
    is_same = score >= threshold

    if cos >= THRESHOLDS["identical"]:
        level, confidence, color = "Identik", "Sangat Tinggi", "#10b981"
    elif cos >= THRESHOLDS["very_similar"]:
        level, confidence, color = "Sangat Mirip", "Tinggi", "#22c55e"
    elif cos >= THRESHOLDS["similar"]:
        level, confidence, color = "Mirip", "Sedang", "#f59e0b"
    elif cos >= THRESHOLDS["uncertain"]:
        level, confidence, color = "Kurang Mirip", "Rendah", "#f97316"
    else:
        level, confidence, color = "Tidak Mirip", "Sangat Rendah", "#ef4444"

    euc  = metrics["euclidean_distance_eigenspace"]
    ssim = metrics["ssim_pixel"]
    reasoning = [
        f"Cosine similarity eigenspace: {cos:.2%} ({'tinggi' if cos >= 0.70 else 'rendah'})",
        f"Euclidean distance: {euc:.3f} ({'berdekatan' if euc < 1.0 else 'berjauhan'})",
        f"SSIM pixel: {ssim:.2%}",
    ]

    return {
        "is_same_person" : is_same,
        "verdict"        : "[SAMA] Orang yang Sama" if is_same else "[BEDA] Orang yang Berbeda",
        "verdict_display": "\u2705 Orang yang Sama" if is_same else "\u274c Orang yang Berbeda",
        "verdict_en"     : "Same Person" if is_same else "Different Person",
        "score"          : score,
        "level"          : level,
        "confidence"     : confidence,
        "color"          : color,
        "threshold_used" : threshold,
        "reasoning"      : reasoning,
        "metrics"        : metrics,
    }
