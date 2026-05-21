"""
core/pca_svd.py
===============
Implementasi PCA & SVD untuk deteksi kemiripan wajah.
Menggunakan NumPy murni untuk operasi aljabar linear.

Konsep Aljabar Linear yang diimplementasikan:
- SVD: A = U Σ Vᵀ
- Eigenvalue & Eigenvector dari matriks kovarians
- Proyeksi ke ruang eigen (eigenspace)
"""

import numpy as np
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────
# SVD Decomposition
# ─────────────────────────────────────────────

def svd_decompose(image_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dekomposisi SVD: A = U Σ Vᵀ
    
    Parameters
    ----------
    image_matrix : np.ndarray, shape (n_pixels,) atau (n_images, n_pixels)
        Vektor atau matriks gambar yang sudah di-flatten.
    
    Returns
    -------
    U  : Left singular vectors  (eigenfaces basis)
    S  : Singular values        (diagonal Σ)
    Vt : Right singular vectors (Vᵀ)
    """
    if image_matrix.ndim == 1:
        image_matrix = image_matrix.reshape(1, -1)
    
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    return U, S, Vt


def get_singular_values_info(S: np.ndarray) -> dict:
    """
    Analisis singular values untuk visualisasi.
    
    Singular values menunjukkan seberapa penting setiap komponen.
    Makin besar nilai, makin banyak informasi yang dikandung.
    """
    total_energy = np.sum(S ** 2)
    explained_variance = (S ** 2) / total_energy * 100  # dalam persen
    cumulative_variance = np.cumsum(explained_variance)
    
    return {
        "singular_values": S,
        "explained_variance_pct": explained_variance,
        "cumulative_variance_pct": cumulative_variance,
        "total_energy": total_energy,
    }


# ─────────────────────────────────────────────
# PCA via Eigenvalue Decomposition
# ─────────────────────────────────────────────

def compute_mean_face(images: np.ndarray) -> np.ndarray:
    """
    Hitung wajah rata-rata (mean face).
    
    Dalam PCA, data harus dicentrasi terlebih dahulu:
    A_centered = A - mean(A)
    """
    return np.mean(images, axis=0)


def compute_covariance_matrix(images_centered: np.ndarray) -> np.ndarray:
    """
    Hitung matriks kovarians: C = (1/n) * Aᵀ A
    
    Matriks kovarians adalah inti dari PCA.
    Eigenvalue & eigenvector dari C adalah komponen utama.
    
    Note: Kita pakai 'economy' trick untuk efisiensi —
    eigenfaces dihitung dari AᵀA bukan AAᵀ.
    """
    n = images_centered.shape[0]
    # C = (1/n) Aᵀ A  — matriks n×n, lebih efisien dari d×d
    C = (images_centered.T @ images_centered) / n
    return C


def compute_eigenfaces(
    images: np.ndarray,
    n_components: int = 50,
    use_svd: bool = True
) -> dict:
    """
    Hitung eigenfaces dari kumpulan gambar.
    
    Eigenfaces = eigenvector dari matriks kovarians gambar-gambar wajah.
    Setiap eigenface adalah "pola dasar" yang membentuk wajah manusia.
    
    Parameters
    ----------
    images       : np.ndarray, shape (n_images, n_pixels)
    n_components : int, jumlah eigenfaces yang diambil
    use_svd      : bool, gunakan SVD (lebih stabil) atau eig() langsung
    
    Returns
    -------
    dict berisi eigenfaces, eigenvalues, mean_face, dan explained_variance
    """
    # Pastikan format (n_images, n_pixels)
    if images.ndim == 3:
        n = images.shape[0]
        images = images.reshape(n, -1)
    
    # Langkah 1: Hitung mean face dan centrasi
    mean_face = compute_mean_face(images)
    images_centered = images - mean_face
    
    n_components = min(n_components, images.shape[0], images.shape[1])
    
    if use_svd:
        # Metode SVD (lebih numeris stabil)
        # A = U Σ Vᵀ  →  eigenfaces = Vᵀ[:n_components]
        U, S, Vt = np.linalg.svd(images_centered, full_matrices=False)
        
        eigenfaces = Vt[:n_components]           # shape (n_comp, n_pixels)
        eigenvalues = (S[:n_components] ** 2) / images.shape[0]
        
        explained_variance = (S ** 2)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        
    else:
        # Metode Eigenvalue langsung (ilustratif)
        C = compute_covariance_matrix(images_centered)
        eigenvalues_all, eigenvectors_all = np.linalg.eig(C)
        
        # Urutkan dari terbesar ke terkecil
        idx = np.argsort(eigenvalues_all)[::-1]
        eigenvalues_all = eigenvalues_all[idx].real
        eigenvectors_all = eigenvectors_all[:, idx].real
        
        eigenfaces = eigenvectors_all[:, :n_components].T
        eigenvalues = eigenvalues_all[:n_components]
        
        total = np.sum(eigenvalues_all)
        explained_variance_ratio = eigenvalues_all / total
    
    return {
        "eigenfaces": eigenfaces,            # (n_comp, n_pixels) — setiap baris = satu eigenface
        "eigenvalues": eigenvalues,          # (n_comp,) — seberapa penting tiap eigenface
        "mean_face": mean_face,              # (n_pixels,) — wajah rata-rata
        "explained_variance_ratio": explained_variance_ratio[:n_components],
        "n_components": n_components,
        "singular_values": S if use_svd else None,
    }


# ─────────────────────────────────────────────
# Proyeksi ke Eigenspace
# ─────────────────────────────────────────────

def project_to_eigenspace(
    face: np.ndarray,
    eigenfaces: np.ndarray,
    mean_face: np.ndarray
) -> np.ndarray:
    """
    Proyeksikan wajah ke ruang eigen.
    
    Operasi:
        face_centered = face - mean_face
        weights = eigenfaces @ face_centered
    
    Hasilnya adalah "representasi" wajah dalam ruang eigen —
    vektor yang jauh lebih kecil dari dimensi asli gambar.
    
    Ini adalah operasi proyeksi vektor ke subspace (Aljabar Linear bab 6).
    """
    if face.ndim > 1:
        face = face.flatten()
    
    face_centered = face.astype(float) - mean_face
    
    # weights[i] = seberapa besar kontribusi eigenface ke-i
    weights = eigenfaces @ face_centered  # shape (n_components,)
    
    return weights


def reconstruct_from_eigenspace(
    weights: np.ndarray,
    eigenfaces: np.ndarray,
    mean_face: np.ndarray
) -> np.ndarray:
    """
    Rekonstruksi wajah dari representasi eigenspace-nya.
    
    face_reconstructed = mean_face + Σ(weights[i] * eigenface[i])
    
    Berguna untuk visualisasi — melihat seberapa baik representasi eigenspace
    menangkap informasi asli gambar.
    """
    reconstructed = mean_face + weights @ eigenfaces
    return reconstructed


# ─────────────────────────────────────────────
# Two-Image Pipeline (utama untuk tugas ini)
# ─────────────────────────────────────────────

def analyze_two_faces(
    face1: np.ndarray,
    face2: np.ndarray,
    n_components: int = 30,
) -> dict:
    """
    Pipeline lengkap: analisis dua gambar wajah.
    
    Karena hanya ada 2 gambar, kita stack keduanya dan jalankan SVD
    untuk mendapat eigenspace dari kedua gambar tersebut.
    
    Parameters
    ----------
    face1, face2 : np.ndarray, shape (H, W) grayscale
    n_components : int, jumlah komponen PCA yang dipakai
    
    Returns
    -------
    dict berisi semua informasi analitis + skor kemiripan
    """
    # Flatten kedua gambar
    f1 = face1.flatten().astype(float)
    f2 = face2.flatten().astype(float)
    
    # Stack jadi matriks 2×n_pixels
    images_stack = np.stack([f1, f2], axis=0)
    
    # Hitung eigenfaces dari gabungan kedua gambar
    n_comp = min(n_components, 2)  # max 2 komponen kalau cuma 2 gambar
    eigenface_data = compute_eigenfaces(images_stack, n_components=n_comp)
    
    # Proyeksikan masing-masing ke eigenspace
    w1 = project_to_eigenspace(f1, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    w2 = project_to_eigenspace(f2, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    
    # Rekonstruksi untuk visualisasi
    r1 = reconstruct_from_eigenspace(w1, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    r2 = reconstruct_from_eigenspace(w2, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
    
    # SVD individual untuk visualisasi singular values
    U1, S1, Vt1 = svd_decompose(face1.astype(float))
    U2, S2, Vt2 = svd_decompose(face2.astype(float))
    
    return {
        "face1_flat": f1,
        "face2_flat": f2,
        "eigenface_data": eigenface_data,
        "weights_face1": w1,
        "weights_face2": w2,
        "reconstructed_face1": r1,
        "reconstructed_face2": r2,
        "svd_face1": {"U": U1, "S": S1, "Vt": Vt1},
        "svd_face2": {"U": U2, "S": S2, "Vt": Vt2},
    }
