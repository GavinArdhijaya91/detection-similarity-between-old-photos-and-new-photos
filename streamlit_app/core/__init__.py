"""
core/__init__.py
"""
from .pca_svd import (
    svd_decompose,
    compute_eigenfaces,
    project_to_eigenspace,
    reconstruct_from_eigenspace,
    analyze_two_faces,
    get_singular_values_info,
)
from .face_utils import (
    load_image_from_bytes,
    load_image_from_pil,
    preprocess_face,
    detect_face,
    draw_face_box,
)
from .similarity import (
    cosine_similarity,
    euclidean_distance,
    compute_all_metrics,
    make_decision,
)

__all__ = [
    "svd_decompose", "compute_eigenfaces", "project_to_eigenspace",
    "reconstruct_from_eigenspace", "analyze_two_faces", "get_singular_values_info",
    "load_image_from_bytes", "load_image_from_pil", "preprocess_face",
    "detect_face", "draw_face_box",
    "cosine_similarity", "euclidean_distance", "compute_all_metrics", "make_decision",
]
