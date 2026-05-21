"""
app.py — Streamlit App
=======================
Deteksi Kemiripan Foto Lama vs Foto Baru
Menggunakan PCA & SVD (Eigenfaces)

Mata Kuliah: Aljabar Linear — Semester 2
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cv2
import sys
import os

# Tambahkan direktori parent ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pca_svd import (
    svd_decompose, compute_eigenfaces,
    project_to_eigenspace, reconstruct_from_eigenspace,
    get_singular_values_info,
)
from core.face_utils import (
    load_image_from_pil, preprocess_face, detect_face, draw_face_box,
)
from core.similarity import compute_all_metrics, make_decision
from PIL import Image
import io

# ─────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FaceMatch PCA/SVD — Aljabar Linear",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card2: #1a2235;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-yellow: #f59e0b;
    --accent-pink: #ec4899;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: rgba(255,255,255,0.08);
    --glow-blue: 0 0 20px rgba(59,130,246,0.3);
    --glow-purple: 0 0 20px rgba(139,92,246,0.3);
    --glow-green: 0 0 20px rgba(16,185,129,0.3);
}

/* ── Base ── */
html, body, .stApp { background: var(--bg-primary) !important; font-family: 'Inter', sans-serif; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5), var(--glow-purple);
}
.app-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 30% 50%, rgba(139,92,246,0.15) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(59,130,246,0.1) 0%, transparent 60%);
}
.app-header-content { position: relative; z-index: 1; }
.app-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e2e8f0, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.app-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}
.badge-row { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 50px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
}
.badge-blue  { border-color: rgba(59,130,246,0.4); color: #93c5fd; }
.badge-purple{ border-color: rgba(139,92,246,0.4); color: #c4b5fd; }
.badge-cyan  { border-color: rgba(6,182,212,0.4);  color: #67e8f9; }

/* ── Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue));
    opacity: 0.7;
}
.metric-card:hover {
    border-color: rgba(139,92,246,0.4);
    box-shadow: var(--glow-purple);
    transform: translateY(-2px);
}

/* ── Upload Zone ── */
.upload-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.8) !important;
    border: 2px dashed rgba(139,92,246,0.4) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,92,246,0.8) !important;
    background: rgba(139,92,246,0.05) !important;
}

/* ── Result Card ── */
.result-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 1.5rem 0;
}
.result-same {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border: 2px solid rgba(16,185,129,0.5);
    box-shadow: 0 0 40px rgba(16,185,129,0.2);
}
.result-diff {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.08));
    border: 2px solid rgba(239,68,68,0.5);
    box-shadow: 0 0 40px rgba(239,68,68,0.2);
}
.result-score {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-label {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-sublabel {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

/* ── Math Box ── */
.math-box {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid var(--accent-purple);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin: 0.75rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #c4b5fd;
}

/* ── Section Title ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.5rem;
}

/* ── Metrics Row ── */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-name { color: var(--text-secondary); }
.metric-value { font-weight: 600; font-family: 'JetBrains Mono', monospace; color: var(--text-primary); }

/* ── Progress Bar ── */
.progress-container { margin: 1rem 0; }
.progress-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.5s ease;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080d1a !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }

/* ── Misc ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
    box-shadow: 0 4px 15px rgba(139,92,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139,92,246,0.5) !important;
}
h1,h2,h3,h4 { color: var(--text-primary) !important; }
p { color: var(--text-secondary) !important; }
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div class="app-header-content">
    <div class="app-title">🔬 FaceMatch — PCA & SVD</div>
    <div class="app-subtitle">Deteksi Kemiripan Foto Lama vs Foto Baru · Implementasi Eigenfaces</div>
    <div class="badge-row">
      <span class="badge badge-purple">📐 Aljabar Linear</span>
      <span class="badge badge-blue">🧮 Eigenvalue & Eigenvector</span>
      <span class="badge badge-cyan">🔢 SVD Decomposition</span>
      <span class="badge badge-blue">📊 PCA Projection</span>
      <span class="badge badge-purple">📏 Cosine Similarity</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar — Parameter & Info
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parameter")
    
    threshold = st.slider(
        "Ambang Batas Kemiripan",
        min_value=0.40,
        max_value=0.95,
        value=0.70,
        step=0.01,
        help="Jika composite score ≥ threshold → dianggap orang yang sama",
    )
    
    detect_faces_opt = st.toggle("🧠 Auto Deteksi Wajah (Haar Cascade)", value=True)
    show_math = st.toggle("📐 Tampilkan Penjelasan Matematis", value=True)
    show_eigenfaces = st.toggle("👁️ Visualisasi Eigenfaces", value=True)
    show_svd = st.toggle("📉 Grafik Singular Values", value=True)
    
    st.divider()
    st.markdown("## 📚 Konsep Matematika")
    st.markdown("""
**SVD (Singular Value Decomposition)**
```
A = U Σ Vᵀ
```
- **U** = Left singular vectors
- **Σ** = Matriks diagonal (singular values)
- **Vᵀ** = Right singular vectors (eigenfaces)

**PCA (Principal Component Analysis)**
```
C = (1/n) AᵀA
C·v = λ·v
```
- **λ** = Eigenvalue (kepentingan)
- **v** = Eigenvector (arah komponen)

**Cosine Similarity**
```
cos(θ) = (a·b)/(‖a‖·‖b‖)
```

**Euclidean Distance**
```
d = ‖a − b‖₂
```
""")
    
    st.divider()
    st.caption("🎓 Mata Kuliah Aljabar Linear — Semester 2")
    st.caption("Implementasi: NumPy · OpenCV · Streamlit")


# ─────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────

st.markdown('<div class="section-title">📸 Upload Foto</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="upload-label">📷 Foto Lama (Masa Kecil)</div>', unsafe_allow_html=True)
    file1 = st.file_uploader(
        "Upload foto pertama",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="photo_old",
        label_visibility="collapsed",
    )

with col2:
    st.markdown('<div class="upload-label">📱 Foto Baru (Saat Ini)</div>', unsafe_allow_html=True)
    file2 = st.file_uploader(
        "Upload foto kedua",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="photo_new",
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_matplotlib_dark():
    """Set matplotlib ke dark theme konsisten."""
    plt.rcParams.update({
        "figure.facecolor" : "#111827",
        "axes.facecolor"   : "#1a2235",
        "axes.edgecolor"   : "#2d3748",
        "axes.labelcolor"  : "#94a3b8",
        "xtick.color"      : "#94a3b8",
        "ytick.color"      : "#94a3b8",
        "text.color"       : "#f1f5f9",
        "grid.color"       : "#1e293b",
        "grid.alpha"       : 0.5,
        "font.family"      : "DejaVu Sans",
    })


def score_to_color_gradient(score: float) -> str:
    """Konversi score 0-1 ke warna CSS."""
    if score >= 0.85:   return "#10b981"
    elif score >= 0.70: return "#22c55e"
    elif score >= 0.55: return "#f59e0b"
    elif score >= 0.40: return "#f97316"
    else:               return "#ef4444"


def render_progress_bar(score: float, label: str, color: str):
    pct = score * 100
    st.markdown(f"""
    <div class="progress-container">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;font-size:0.8rem;">
        <span style="color:#94a3b8">{label}</span>
        <span style="color:{color};font-weight:700;font-family:'JetBrains Mono',monospace">{pct:.1f}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill"
             style="width:{pct}%;background:linear-gradient(90deg,{color}88,{color})">
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────

if file1 and file2:
    with st.spinner("⚙️ Memproses gambar & menjalankan SVD/PCA..."):
        
        # Load images
        pil1 = Image.open(file1)
        pil2 = Image.open(file2)
        
        gray1 = load_image_from_pil(pil1)
        gray2 = load_image_from_pil(pil2)
        
        # Preprocessing
        face1_proc, info1 = preprocess_face(gray1, detect=detect_faces_opt)
        face2_proc, info2 = preprocess_face(gray2, detect=detect_faces_opt)
        
        # SVD per gambar (untuk visualisasi individual)
        # Gambar sebagai matriks 2D → SVD langsung
        face1_matrix = face1_proc  # 128×128
        face2_matrix = face2_proc

        U1, S1, Vt1 = np.linalg.svd(face1_matrix, full_matrices=False)
        U2, S2, Vt2 = np.linalg.svd(face2_matrix, full_matrices=False)
        
        # Stack kedua gambar → hitung eigenfaces gabungan
        images_stack = np.stack([face1_proc.flatten(), face2_proc.flatten()], axis=0)
        eigenface_data = compute_eigenfaces(images_stack, n_components=2)
        
        # Proyeksi ke eigenspace
        w1 = project_to_eigenspace(face1_proc.flatten(), eigenface_data["eigenfaces"], eigenface_data["mean_face"])
        w2 = project_to_eigenspace(face2_proc.flatten(), eigenface_data["eigenfaces"], eigenface_data["mean_face"])
        
        # Rekonstruksi
        r1 = reconstruct_from_eigenspace(w1, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
        r2 = reconstruct_from_eigenspace(w2, eigenface_data["eigenfaces"], eigenface_data["mean_face"])
        
        r1_img = r1.reshape(128, 128)
        r2_img = r2.reshape(128, 128)
        
        # Hitung semua metrik
        metrics = compute_all_metrics(w1, w2, face1_proc, face2_proc)
        decision = make_decision(metrics, threshold=threshold)

    # ── Preview Foto ──────────────────────────────────────────
    st.markdown('<div class="section-title">🖼️ Pratinjau Foto</div>', unsafe_allow_html=True)
    
    pcol1, pcol2 = st.columns(2, gap="large")
    
    with pcol1:
        st.markdown("**📷 Foto Lama**")
        display1 = np.array(pil1.convert("RGB"))
        if info1["bbox"] is not None:
            display1 = draw_face_box(display1, info1["bbox"])
        st.image(display1, use_container_width=True)
        
        detect_status1 = "✅ Wajah terdeteksi" if info1["face_detected"] else "⚠️ Wajah tidak terdeteksi — pakai gambar penuh"
        st.caption(detect_status1)
    
    with pcol2:
        st.markdown("**📱 Foto Baru**")
        display2 = np.array(pil2.convert("RGB"))
        if info2["bbox"] is not None:
            display2 = draw_face_box(display2, info2["bbox"])
        st.image(display2, use_container_width=True)
        
        detect_status2 = "✅ Wajah terdeteksi" if info2["face_detected"] else "⚠️ Wajah tidak terdeteksi — pakai gambar penuh"
        st.caption(detect_status2)
    
    # ── Hasil Utama ──────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Hasil Analisis</div>', unsafe_allow_html=True)
    
    score = decision["score"]
    is_same = decision["is_same_person"]
    result_class = "result-same" if is_same else "result-diff"
    score_color  = "#10b981" if is_same else "#ef4444"
    
    verdict_display = decision.get("verdict_display", decision["verdict"])
    st.markdown(f"""
    <div class="result-card {result_class}">
      <div style="font-size:3rem;margin-bottom:0.5rem">{'\u2705' if is_same else '\u274c'}</div>
      <div class="result-label" style="color:{score_color}">{verdict_display}</div>
      <div class="result-score" style="color:{score_color}">{score:.1%}</div>
      <div class="result-sublabel">Composite Similarity Score</div>
      <div style="margin-top:1rem;padding:0.5rem 1rem;background:rgba(0,0,0,0.3);border-radius:8px;
                  display:inline-block;font-size:0.85rem;color:#94a3b8">
        Tingkat Kepercayaan: <strong style="color:{decision['color']}">{decision['level']} ({decision['confidence']})</strong>
        &nbsp;|&nbsp; Threshold: <strong style="color:#94a3b8">{threshold:.0%}</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Metrik Detail ──────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Metrik Kemiripan Detail</div>', unsafe_allow_html=True)
    
    m = metrics
    mcol1, mcol2 = st.columns(2, gap="large")
    
    with mcol1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔢 Metrik Eigenspace (PCA)**")
        
        cos_score = m["cosine_similarity_eigenspace"]
        render_progress_bar(max(0, cos_score), "Cosine Similarity (Eigenspace)", score_to_color_gradient(max(0, cos_score)))
        
        euc_sim = m["euclidean_similarity_norm"]
        render_progress_bar(euc_sim, "Euclidean Similarity (Normalized)", score_to_color_gradient(euc_sim))
        
        st.markdown(f"""
        <div class="metric-row">
          <span class="metric-name">Euclidean Distance</span>
          <span class="metric-value">{m['euclidean_distance_eigenspace']:.4f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with mcol2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🖼️ Metrik Pixel**")
        
        ssim = m["ssim_pixel"]
        render_progress_bar(ssim, "SSIM (Structural Similarity)", score_to_color_gradient(ssim))
        
        pcos = m["cosine_similarity_pixel"]
        render_progress_bar(max(0, pcos), "Cosine Similarity (Pixel)", score_to_color_gradient(max(0, pcos)))
        
        render_progress_bar(score, "Composite Score (Weighted)", score_to_color_gradient(score))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Penjelasan Matematis ──────────────────────────────────────────
    if show_math:
        st.markdown('<div class="section-title">📐 Penjelasan Matematis</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔢 SVD Step-by-Step", "📊 Eigenvalue & Eigenface", "📏 Similarity Calculation"])
        
        with tab1:
            st.markdown("#### SVD Decomposition: A = U Σ Vᵀ")
            st.markdown("""
            Setiap gambar wajah diperlakukan sebagai **matriks piksel** (128×128), lalu didekomposisi:
            """)
            
            col_svd1, col_svd2 = st.columns(2)
            with col_svd1:
                st.markdown("**📷 Foto Lama — Singular Values (10 teratas)**")
                sv_data1 = {"Rank": list(range(1, 11)), "Singular Value": [f"{v:.2f}" for v in S1[:10]],
                            "Variance (%)": [f"{(v**2/np.sum(S1**2)*100):.2f}%" for v in S1[:10]]}
                st.dataframe(sv_data1, use_container_width=True, hide_index=True)
            
            with col_svd2:
                st.markdown("**📱 Foto Baru — Singular Values (10 teratas)**")
                sv_data2 = {"Rank": list(range(1, 11)), "Singular Value": [f"{v:.2f}" for v in S2[:10]],
                            "Variance (%)": [f"{(v**2/np.sum(S2**2)*100):.2f}%" for v in S2[:10]]}
                st.dataframe(sv_data2, use_container_width=True, hide_index=True)
            
            st.markdown(f"""
            <div class="math-box">
# Dimensi matriks SVD untuk gambar 128×128:
A  shape = (128, 128)
U  shape = (128, 128)   ← left singular vectors
Σ  shape = (128,)       ← singular values (diagonal)
Vt shape = (128, 128)   ← right singular vectors (eigenfaces)

# Rekonstruksi dengan k=20 komponen saja:
A_approx = U[:, :20] @ diag(Σ[:20]) @ Vt[:20, :]
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("#### Eigenvalue & Eigenface dari Matriks Kovarians")
            
            ev = eigenface_data["eigenvalues"]
            ef = eigenface_data["eigenfaces"]
            evr = eigenface_data["explained_variance_ratio"]
            
            st.markdown(f"""
            **Eigenvalue gabungan 2 gambar:**
            - λ₁ = `{ev[0]:.6f}` — Komponen utama (menyimpan {evr[0]*100:.2f}% informasi)
            - λ₂ = `{ev[1]:.6f}` — Komponen sekunder (menyimpan {evr[1]*100:.2f}% informasi)
            """)
            
            st.markdown(f"""
            <div class="math-box">
# Proyeksi ke eigenspace:
# w = eigenfaces @ (face - mean_face)

Foto Lama → w₁ = [{w1[0]:.4f}, {w1[1]:.4f}]
Foto Baru → w₂ = [{w2[0]:.4f}, {w2[1]:.4f}]

# w adalah representasi wajah di ruang 2D
# (jauh lebih kecil dari 16384 dimensi asli!)
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("#### Perhitungan Similarity")
            
            cos_val = m["cosine_similarity_eigenspace"]
            euc_val = m["euclidean_distance_eigenspace"]
            
            st.markdown(f"""
            <div class="math-box">
# Cosine Similarity di Eigenspace:
a = w₁ = [{w1[0]:.4f}, {w1[1]:.4f}]
b = w₂ = [{w2[0]:.4f}, {w2[1]:.4f}]

a · b    = {np.dot(w1, w2):.6f}
‖a‖      = {np.linalg.norm(w1):.6f}
‖b‖      = {np.linalg.norm(w2):.6f}

cos(θ) = {np.dot(w1,w2):.6f} / ({np.linalg.norm(w1):.6f} × {np.linalg.norm(w2):.6f})
       = {cos_val:.6f}  →  {cos_val*100:.2f}%

# Euclidean Distance:
‖w₁ - w₂‖ = {euc_val:.6f}

# Composite Score (weighted):
score = 0.45×{cos_val:.4f} + 0.25×{m['euclidean_similarity_norm']:.4f}
      + 0.20×{m['ssim_pixel']:.4f} + 0.10×{m['cosine_similarity_pixel']:.4f}
      = {score:.4f}  →  {score*100:.2f}%

# Threshold = {threshold:.0%}  →  {'✅ SAMA' if is_same else '❌ BERBEDA'}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Reasoning:**")
            for r in decision["reasoning"]:
                st.markdown(f"- {r}")
    
    # ── Visualisasi Eigenfaces & SVD ──────────────────────────────
    if show_eigenfaces or show_svd:
        st.markdown('<div class="section-title">📈 Visualisasi</div>', unsafe_allow_html=True)
        
        make_matplotlib_dark()
        
        if show_eigenfaces:
            # Visualisasi: processed faces + rekonstruksi + eigenfaces
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            fig.patch.set_facecolor("#111827")
            
            titles = [
                ["Foto Lama\n(Preprocessed)", "Foto Baru\n(Preprocessed)",
                 "Mean Face\n(Rata-rata)", "Eigenface #1"],
                ["Rekonstruksi\nFoto Lama", "Rekonstruksi\nFoto Baru",
                 "SVD Approx\nFoto Lama (k=20)", "SVD Approx\nFoto Baru (k=20)"],
            ]
            
            # Rekonstruksi SVD dengan k=20 komponen
            k = min(20, len(S1))
            svd_approx1 = (U1[:, :k] * S1[:k]) @ Vt1[:k, :]
            svd_approx1 = np.clip(svd_approx1, 0, 1)
            
            k2 = min(20, len(S2))
            svd_approx2 = (U2[:, :k2] * S2[:k2]) @ Vt2[:k2, :]
            svd_approx2 = np.clip(svd_approx2, 0, 1)
            
            mean_img = eigenface_data["mean_face"].reshape(128, 128)
            ef1_img  = eigenface_data["eigenfaces"][0].reshape(128, 128)
            
            # Normalisasi eigenface untuk tampilan
            ef1_display = (ef1_img - ef1_img.min()) / (ef1_img.max() - ef1_img.min() + 1e-8)
            mean_display = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)
            
            images_row0 = [face1_proc, face2_proc, mean_display, ef1_display]
            images_row1 = [r1_img, r2_img, svd_approx1, svd_approx2]
            
            custom_cmap = LinearSegmentedColormap.from_list(
                "eigen", ["#0a0e1a", "#3b82f6", "#8b5cf6", "#f1f5f9"]
            )
            
            for col_i, (img, title) in enumerate(zip(images_row0, titles[0])):
                ax = axes[0, col_i]
                cmap = custom_cmap if col_i == 3 else "gray"
                ax.imshow(np.clip(img, 0, 1), cmap=cmap, vmin=0, vmax=1)
                ax.set_title(title, fontsize=9, color="#94a3b8", pad=6)
                ax.axis("off")
                ax.set_facecolor("#1a2235")
            
            for col_i, (img, title) in enumerate(zip(images_row1, titles[1])):
                ax = axes[1, col_i]
                ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
                ax.set_title(title, fontsize=9, color="#94a3b8", pad=6)
                ax.axis("off")
                ax.set_facecolor("#1a2235")
            
            plt.suptitle("Visualisasi Eigenfaces & Rekonstruksi SVD", color="#f1f5f9", fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        if show_svd:
            # Grafik Singular Values
            fig2, (ax_sv, ax_var) = plt.subplots(1, 2, figsize=(13, 4.5))
            fig2.patch.set_facecolor("#111827")
            
            k_show = min(30, len(S1), len(S2))
            x_range = np.arange(1, k_show + 1)
            
            # Singular Values
            ax_sv.plot(x_range, S1[:k_show], "o-", color="#3b82f6", linewidth=2, markersize=5, label="Foto Lama", alpha=0.9)
            ax_sv.plot(x_range, S2[:k_show], "s-", color="#8b5cf6", linewidth=2, markersize=5, label="Foto Baru", alpha=0.9)
            ax_sv.fill_between(x_range, S1[:k_show], alpha=0.15, color="#3b82f6")
            ax_sv.fill_between(x_range, S2[:k_show], alpha=0.15, color="#8b5cf6")
            ax_sv.set_facecolor("#1a2235")
            ax_sv.set_title("Singular Values (σ)", color="#f1f5f9", fontweight="bold")
            ax_sv.set_xlabel("Rank")
            ax_sv.set_ylabel("Nilai Singular (σ)")
            ax_sv.legend(framealpha=0.3)
            ax_sv.grid(True, alpha=0.3)
            ax_sv.spines["bottom"].set_color("#2d3748")
            ax_sv.spines["left"].set_color("#2d3748")
            ax_sv.spines["top"].set_visible(False)
            ax_sv.spines["right"].set_visible(False)
            
            # Cumulative Variance
            cum1 = np.cumsum(S1[:k_show]**2) / np.sum(S1**2) * 100
            cum2 = np.cumsum(S2[:k_show]**2) / np.sum(S2**2) * 100
            
            ax_var.plot(x_range, cum1, "o-", color="#10b981", linewidth=2, markersize=5, label="Foto Lama", alpha=0.9)
            ax_var.plot(x_range, cum2, "s-", color="#f59e0b", linewidth=2, markersize=5, label="Foto Baru", alpha=0.9)
            ax_var.axhline(y=95, color="#ef4444", linestyle="--", alpha=0.6, linewidth=1.5, label="95% threshold")
            ax_var.fill_between(x_range, cum1, alpha=0.1, color="#10b981")
            ax_var.fill_between(x_range, cum2, alpha=0.1, color="#f59e0b")
            ax_var.set_facecolor("#1a2235")
            ax_var.set_title("Cumulative Explained Variance (%)", color="#f1f5f9", fontweight="bold")
            ax_var.set_xlabel("Jumlah Komponen")
            ax_var.set_ylabel("Variance Explained (%)")
            ax_var.set_ylim(0, 105)
            ax_var.legend(framealpha=0.3)
            ax_var.grid(True, alpha=0.3)
            ax_var.spines["bottom"].set_color("#2d3748")
            ax_var.spines["left"].set_color("#2d3748")
            ax_var.spines["top"].set_visible(False)
            ax_var.spines["right"].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
    
    # ── Eigenspace 2D Plot ──────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Proyeksi Eigenspace 2D</div>', unsafe_allow_html=True)
    
    make_matplotlib_dark()
    fig3, ax = plt.subplots(figsize=(8, 5))
    fig3.patch.set_facecolor("#111827")
    ax.set_facecolor("#1a2235")
    
    # Plot proyeksi kedua wajah
    ax.scatter(w1[0], w1[1], s=200, c="#3b82f6", zorder=5, label="Foto Lama", edgecolors="#93c5fd", linewidths=2)
    ax.scatter(w2[0], w2[1], s=200, c="#8b5cf6", zorder=5, label="Foto Baru", edgecolors="#c4b5fd", linewidths=2)
    
    # Garis penghubung
    ax.plot([w1[0], w2[0]], [w1[1], w2[1]], "--", color="#64748b", alpha=0.6, linewidth=1.5)
    
    # Anotasi
    ax.annotate("Foto Lama", (w1[0], w1[1]), textcoords="offset points", xytext=(10, 10),
                color="#93c5fd", fontsize=9, fontweight="bold")
    ax.annotate("Foto Baru", (w2[0], w2[1]), textcoords="offset points", xytext=(10, -18),
                color="#c4b5fd", fontsize=9, fontweight="bold")
    
    # Jarak
    mid_x = (w1[0] + w2[0]) / 2
    mid_y = (w1[1] + w2[1]) / 2
    ax.annotate(f"d = {m['euclidean_distance_eigenspace']:.3f}", (mid_x, mid_y),
                textcoords="offset points", xytext=(8, 0),
                color="#f59e0b", fontsize=8, ha="left")
    
    ax.set_title("Proyeksi Wajah di Ruang Eigen 2D\n(PC1 vs PC2)", color="#f1f5f9", fontweight="bold")
    ax.set_xlabel("Eigenface Component 1 (PC1)")
    ax.set_ylabel("Eigenface Component 2 (PC2)")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.spines["bottom"].set_color("#2d3748")
    ax.spines["left"].set_color("#2d3748")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

else:
    # ── Placeholder / Instructions ──────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;background:#111827;border:2px dashed rgba(139,92,246,0.3);
                border-radius:20px;margin-top:2rem">
      <div style="font-size:3rem;margin-bottom:1rem">🔬</div>
      <h3 style="color:#e2e8f0;margin-bottom:0.5rem">Upload Dua Foto untuk Memulai</h3>
      <p style="color:#64748b;max-width:500px;margin:0 auto">
        Upload <strong style="color:#a78bfa">Foto Lama</strong> (masa kecil) dan 
        <strong style="color:#60a5fa">Foto Baru</strong> (saat ini) untuk mendeteksi 
        kemiripan menggunakan algoritma <strong style="color:#6ee7b7">PCA & SVD (Eigenfaces)</strong>.
      </p>
      <div style="margin-top:2rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
        <div style="text-align:center">
          <div style="font-size:1.5rem">📐</div>
          <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.3rem">SVD Decomposition</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:1.5rem">🧮</div>
          <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.3rem">Eigenvalue & Eigenvector</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:1.5rem">📊</div>
          <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.3rem">PCA Projection</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:1.5rem">📏</div>
          <div style="color:#94a3b8;font-size:0.8rem;margin-top:0.3rem">Cosine Similarity</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:4rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.06)">
  <p style="color:#334155;font-size:0.8rem">
    🎓 Tugas Mata Kuliah Aljabar Linear · Semester 2 · 
    Implementasi PCA & SVD (Eigenfaces) · NumPy · OpenCV · Streamlit
  </p>
</div>
""", unsafe_allow_html=True)
