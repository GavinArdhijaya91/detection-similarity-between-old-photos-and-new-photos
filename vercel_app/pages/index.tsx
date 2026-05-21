// pages/index.tsx
// ──────────────────────────────────────────────────────────────
// FaceMatch PCA/SVD — Main Page (Next.js + Vercel)
// Deteksi Kemiripan Foto Lama vs Foto Baru
// ──────────────────────────────────────────────────────────────

import Head from 'next/head';
import { useState, useCallback, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ReferenceLine, Legend,
} from 'recharts';

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface AnalysisResult {
  success: boolean;
  decision: {
    is_same_person: boolean;
    verdict: string;
    verdict_icon: string;
    level: string;
    confidence: string;
    color: string;
    threshold_used: number;
  };
  metrics: {
    cosine_similarity_eigenspace: number;
    euclidean_distance_eigenspace: number;
    euclidean_similarity_norm: number;
    ssim_pixel: number;
    cosine_similarity_pixel: number;
    composite_score: number;
  };
  math_data: {
    eigenvalues: number[];
    weights_face1: number[];
    weights_face2: number[];
    singular_values_face1: Array<{ rank: number; value: number; variance_pct: number }>;
    singular_values_face2: Array<{ rank: number; value: number; variance_pct: number }>;
    singular_values_joint: number[];
  };
  preprocessing: {
    face1_detected: boolean;
    face2_detected: boolean;
    image_size: string;
  };
}

// ─────────────────────────────────────────────
// Components
// ─────────────────────────────────────────────

function ProgressBar({ value, color, label }: { value: number; color: string; label: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const displayColor = pct >= 70 ? '#10b981' : pct >= 55 ? '#f59e0b' : '#ef4444';
  const c = color || displayColor;

  return (
    <div className="progress-container">
      <div className="progress-label">
        <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>{label}</span>
        <span style={{ color: c, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: '0.8rem' }}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${c}88, ${c})` }}
        />
      </div>
    </div>
  );
}

function DropZone({
  label,
  photo,
  onPhoto,
}: {
  label: string;
  photo: string | null;
  onPhoto: (src: string) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = (e) => onPhoto(e.target?.result as string);
      reader.readAsDataURL(file);
    },
    [onPhoto]
  );

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b' }}>
        {label}
      </div>

      {photo ? (
        <div style={{ position: 'relative', borderRadius: 16, overflow: 'hidden', border: '2px solid rgba(139,92,246,0.5)', cursor: 'pointer' }}
             onClick={() => inputRef.current?.click()}>
          <img src={photo} alt={label} style={{ width: '100%', height: 220, objectFit: 'cover', display: 'block' }} />
          <div style={{
            position: 'absolute', inset: 0, background: 'rgba(0,0,0,0)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.3s',
          }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(0,0,0,0.5)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(0,0,0,0)')}
          >
            <span style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600, opacity: 0, transition: 'opacity 0.3s' }}
              onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
              onMouseLeave={(e) => (e.currentTarget.style.opacity = '0')}
            >
              Ganti Foto
            </span>
          </div>
        </div>
      ) : (
        <div
          className={`dropzone ${dragging ? 'drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          style={{ minHeight: 220 }}
        >
          <div className="dropzone-icon">📷</div>
          <div className="dropzone-title">Klik atau Drag & Drop</div>
          <div className="dropzone-hint">PNG, JPG, WEBP • Maks 10MB</div>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
    </div>
  );
}


// ─────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────

const DARK_CHART = {
  backgroundColor: '#111827',
  gridColor: '#1e293b',
  textColor: '#94a3b8',
};

export default function Home() {
  const [photo1, setPhoto1] = useState<string | null>(null);
  const [photo2, setPhoto2] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.70);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'metrics' | 'svd' | 'math'>('metrics');

  const handleAnalyze = async () => {
    if (!photo1 || !photo2) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image1: photo1, image2: photo2, threshold }),
      });

      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Analisis gagal');
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Terjadi kesalahan');
    } finally {
      setLoading(false);
    }
  };

  const isSame = result?.decision.is_same_person;
  const score  = result?.metrics.composite_score ?? 0;
  const scoreColor = score >= 0.70 ? '#10b981' : score >= 0.55 ? '#f59e0b' : '#ef4444';

  return (
    <>
      <Head>
        <title>FaceMatch PCA/SVD — Deteksi Kemiripan Foto | Aljabar Linear</title>
        <meta name="description" content="Deteksi kemiripan antara foto lama dan foto baru menggunakan PCA dan SVD (Eigenfaces). Implementasi konsep Aljabar Linear: eigenvalue, eigenvector, cosine similarity." />
        <meta property="og:title" content="FaceMatch PCA/SVD — Deteksi Kemiripan Foto" />
        <meta property="og:description" content="Implementasi Eigenfaces menggunakan PCA & SVD untuk deteksi kemiripan wajah" />
      </Head>

      <div className="page-wrapper">
        {/* ── Header ── */}
        <header style={{
          padding: '2rem 1.5rem',
          background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)',
          borderBottom: '1px solid rgba(139,92,246,0.2)',
          position: 'relative',
          overflow: 'hidden',
        }}>
          {/* Glow effects */}
          <div style={{
            position: 'absolute', inset: 0,
            background: 'radial-gradient(circle at 30% 50%, rgba(139,92,246,0.12) 0%, transparent 60%), radial-gradient(circle at 70% 50%, rgba(59,130,246,0.08) 0%, transparent 60%)',
            pointerEvents: 'none',
          }} />

          <div className="container" style={{ position: 'relative', zIndex: 1 }}>
            <h1 className="gradient-text" style={{ marginBottom: '0.4rem' }}>
              🔬 FaceMatch — PCA &amp; SVD
            </h1>
            <p style={{ fontSize: '1rem', marginBottom: '1rem', color: '#94a3b8' }}>
              Deteksi Kemiripan Foto Lama vs Foto Baru · Implementasi Eigenfaces Aljabar Linear
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {['📐 Aljabar Linear', '🧮 Eigenvalue & Eigenvector', '🔢 SVD Decomposition', '📊 PCA Projection', '📏 Cosine Similarity'].map((b, i) => (
                <span key={i} className={`badge ${['badge-purple','badge-blue','badge-cyan','badge-blue','badge-purple'][i]}`}>{b}</span>
              ))}
            </div>
          </div>
        </header>

        <main className="container" style={{ padding: '2rem 1.5rem', flex: 1 }}>
          {/* ── Upload Section ── */}
          <section style={{ marginBottom: '2rem' }}>
            <div className="section-title">📸 Upload Foto</div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              <DropZone label="📷 Foto Lama (Masa Kecil)" photo={photo1} onPhoto={setPhoto1} />
              <DropZone label="📱 Foto Baru (Saat Ini)"   photo={photo2} onPhoto={setPhoto2} />
            </div>

            {/* Threshold */}
            <div className="glass-card" style={{ padding: '1.25rem 1.5rem', marginTop: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#94a3b8' }}>
                  ⚖️ Ambang Batas Kemiripan (Threshold)
                </span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#a78bfa', fontWeight: 700, fontSize: '0.9rem' }}>
                  {(threshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range" min={0.40} max={0.95} step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#8b5cf6', cursor: 'pointer', height: 4 }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#475569', marginTop: '0.3rem' }}>
                <span>40% (Longgar)</span><span>70% (Default)</span><span>95% (Ketat)</span>
              </div>
            </div>

            {/* Analyze Button */}
            <button
              className="btn btn-primary"
              style={{ marginTop: '1rem', width: '100%', fontSize: '1rem', padding: '0.8rem' }}
              onClick={handleAnalyze}
              disabled={!photo1 || !photo2 || loading}
            >
              {loading ? (
                <>
                  <div className="spinner" />
                  Menganalisis SVD & PCA...
                </>
              ) : '🔍 Analisis Kemiripan'}
            </button>

            {error && (
              <div style={{
                marginTop: '1rem', padding: '0.75rem 1rem', borderRadius: 12,
                background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                color: '#fca5a5', fontSize: '0.85rem',
              }}>
                ❌ {error}
              </div>
            )}
          </section>

          {/* ── Results ── */}
          {result && (
            <div className="fade-in">
              {/* Main verdict */}
              <section style={{ marginBottom: '2rem' }}>
                <div className="section-title">🎯 Hasil Analisis</div>

                <div className={`glass-card ${isSame ? 'result-same pulse-green' : 'result-diff pulse-red'}`}
                     style={{ padding: '2.5rem', textAlign: 'center', borderRadius: 20 }}>
                  <div style={{ fontSize: '3rem', marginBottom: '0.75rem' }}>
                    {result.decision.verdict_icon}
                  </div>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, color: isSame ? '#10b981' : '#ef4444', marginBottom: '0.3rem' }}>
                    {result.decision.verdict}
                  </div>
                  <div style={{ fontSize: '3.5rem', fontWeight: 900, color: scoreColor, lineHeight: 1, margin: '0.5rem 0' }}>
                    {(score * 100).toFixed(1)}%
                  </div>
                  <div style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: '1rem' }}>Composite Similarity Score</div>

                  <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <span className={`badge ${isSame ? 'badge-green' : 'badge-red'}`}>
                      {result.decision.level}
                    </span>
                    <span className="badge badge-purple">
                      Kepercayaan: {result.decision.confidence}
                    </span>
                    <span className="badge badge-blue">
                      Threshold: {(result.decision.threshold_used * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Deteksi info */}
                  <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '1rem', fontSize: '0.78rem', color: '#475569' }}>
                    <span>{result.preprocessing.face1_detected ? '✅' : '⚠️'} Foto Lama: {result.preprocessing.face1_detected ? 'Wajah terdeteksi' : 'Gambar penuh'}</span>
                    <span>{result.preprocessing.face2_detected ? '✅' : '⚠️'} Foto Baru: {result.preprocessing.face2_detected ? 'Wajah terdeteksi' : 'Gambar penuh'}</span>
                    <span>📐 Ukuran: {result.preprocessing.image_size}</span>
                  </div>
                </div>
              </section>

              {/* Detail Metrics Tabs */}
              <section style={{ marginBottom: '2rem' }}>
                <div className="section-title">📊 Detail Analisis</div>

                <div className="glass-card" style={{ padding: '1.5rem' }}>
                  <div className="tabs">
                    {(['metrics', 'svd', 'math'] as const).map((tab) => (
                      <button
                        key={tab}
                        className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab)}
                      >
                        {{ metrics: '📏 Metrik Kemiripan', svd: '📉 Singular Values', math: '🔢 Matematika' }[tab]}
                      </button>
                    ))}
                  </div>

                  {/* Tab: Metrics */}
                  {activeTab === 'metrics' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                      <div>
                        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Metrik Eigenspace (PCA)
                        </div>
                        <ProgressBar value={Math.max(0, result.metrics.cosine_similarity_eigenspace)} label="Cosine Similarity (Eigenspace)" color="" />
                        <ProgressBar value={result.metrics.euclidean_similarity_norm} label="Euclidean Similarity (Normalized)" color="" />
                        <div className="metric-row">
                          <span className="metric-name">Euclidean Distance</span>
                          <span className="metric-value">{result.metrics.euclidean_distance_eigenspace.toFixed(4)}</span>
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Metrik Pixel
                        </div>
                        <ProgressBar value={result.metrics.ssim_pixel} label="SSIM (Structural Similarity)" color="" />
                        <ProgressBar value={Math.max(0, result.metrics.cosine_similarity_pixel)} label="Cosine Similarity (Pixel)" color="" />
                        <ProgressBar value={result.metrics.composite_score} label="Composite Score (Final)" color={scoreColor} />
                      </div>

                      {/* Eigenspace 2D scatter */}
                      <div style={{ gridColumn: '1/-1' }}>
                        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Proyeksi Eigenspace 2D
                        </div>
                        <div style={{ background: '#111827', borderRadius: 12, padding: '1.25rem', height: 260, position: 'relative' }}>
                          {/* SVG mini scatter plot */}
                          <svg width="100%" height="100%" viewBox="0 0 500 220" style={{ overflow: 'visible' }}>
                            {/* Grid */}
                            {[0.2,0.4,0.6,0.8].map(r => (
                              <line key={r} x1={r*500} y1={0} x2={r*500} y2={220} stroke="#1e293b" strokeWidth={1} />
                            ))}
                            {[0.25,0.5,0.75].map(r => (
                              <line key={r} x1={0} y1={r*220} x2={500} y2={r*220} stroke="#1e293b" strokeWidth={1} />
                            ))}

                            {/* Axis labels */}
                            <text x={250} y={216} textAnchor="middle" fill="#475569" fontSize={11}>Eigenface Component 1 (PC1)</text>
                            <text x={10} y={110} textAnchor="middle" fill="#475569" fontSize={11} transform="rotate(-90,10,110)">PC2</text>

                            {(() => {
                              const w1 = result.math_data.weights_face1;
                              const w2 = result.math_data.weights_face2;
                              if (!w1 || !w2 || w1.length < 2) return null;

                              const allX = [w1[0], w2[0]];
                              const allY = [w1[1], w2[1]];
                              const minX = Math.min(...allX); const maxX = Math.max(...allX);
                              const minY = Math.min(...allY); const maxY = Math.max(...allY);
                              const rangeX = maxX - minX || 1; const rangeY = maxY - minY || 1;
                              const pad = 0.2;

                              const px = (v: number) => ((v - minX) / rangeX * (1 - 2*pad) + pad) * 470 + 15;
                              const py = (v: number) => (1 - (v - minY) / rangeY * (1 - 2*pad) - pad) * 190 + 10;

                              const x1 = px(w1[0]); const y1s = py(w1[1]);
                              const x2 = px(w2[0]); const y2s = py(w2[1]);

                              return (
                                <>
                                  {/* Connection line */}
                                  <line x1={x1} y1={y1s} x2={x2} y2={y2s} stroke="#475569" strokeWidth={1.5} strokeDasharray="5,3" />

                                  {/* Points */}
                                  <circle cx={x1} cy={y1s} r={9} fill="#3b82f6" stroke="#93c5fd" strokeWidth={2} />
                                  <circle cx={x2} cy={y2s} r={9} fill="#8b5cf6" stroke="#c4b5fd" strokeWidth={2} />

                                  {/* Labels */}
                                  <text x={x1+13} y={y1s+4} fill="#93c5fd" fontSize={11} fontWeight={700}>Foto Lama</text>
                                  <text x={x2+13} y={y2s+4} fill="#c4b5fd" fontSize={11} fontWeight={700}>Foto Baru</text>

                                  {/* Coords */}
                                  <text x={x1} y={y1s-14} fill="#64748b" fontSize={9} textAnchor="middle">
                                    ({w1[0].toFixed(3)}, {w1[1].toFixed(3)})
                                  </text>
                                  <text x={x2} y={y2s+20} fill="#64748b" fontSize={9} textAnchor="middle">
                                    ({w2[0].toFixed(3)}, {w2[1].toFixed(3)})
                                  </text>
                                </>
                              );
                            })()}
                          </svg>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Tab: SVD */}
                  {activeTab === 'svd' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                      {/* Chart singular values */}
                      <div style={{ gridColumn: '1/-1' }}>
                        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Singular Values σ per Gambar
                        </div>
                        <div style={{ background: '#111827', borderRadius: 12, padding: '1rem', height: 260 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.math_data.singular_values_face1.map((d, i) => ({
                              rank: d.rank,
                              foto_lama: d.value,
                              foto_baru: result.math_data.singular_values_face2[i]?.value ?? 0,
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                              <XAxis dataKey="rank" stroke="#475569" tick={{ fill: '#475569', fontSize: 11 }} label={{ value: 'Rank', position: 'insideBottom', fill: '#475569', fontSize: 11, offset: -2 }} />
                              <YAxis stroke="#475569" tick={{ fill: '#475569', fontSize: 11 }} />
                              <Tooltip
                                contentStyle={{ background: '#1a2235', border: '1px solid #2d3748', borderRadius: 8, color: '#f1f5f9', fontSize: 12 }}
                                formatter={(v: any, name: string) => [typeof v === 'number' ? v.toFixed(2) : v, name === 'foto_lama' ? '📷 Foto Lama' : '📱 Foto Baru']}
                              />
                              <Legend formatter={(v) => v === 'foto_lama' ? '📷 Foto Lama' : '📱 Foto Baru'} />
                              <Line type="monotone" dataKey="foto_lama" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3, fill: '#3b82f6' }} name="foto_lama" />
                              <Line type="monotone" dataKey="foto_baru" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3, fill: '#8b5cf6' }} name="foto_baru" />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      {/* Cumulative variance */}
                      <div style={{ gridColumn: '1/-1' }}>
                        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Cumulative Explained Variance (%)
                        </div>
                        <div style={{ background: '#111827', borderRadius: 12, padding: '1rem', height: 240 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.math_data.singular_values_face1.map((d, i) => {
                              const sv1 = result.math_data.singular_values_face1;
                              const sv2 = result.math_data.singular_values_face2;
                              const total1 = sv1.reduce((s, x) => s + x.value**2, 0);
                              const total2 = sv2.reduce((s, x) => s + x.value**2, 0);
                              const cum1 = sv1.slice(0, i+1).reduce((s, x) => s + x.value**2/total1*100, 0);
                              const cum2 = sv2.slice(0, i+1).reduce((s, x) => s + x.value**2/total2*100, 0);
                              return { rank: d.rank, foto_lama: cum1, foto_baru: cum2 };
                            })}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                              <XAxis dataKey="rank" stroke="#475569" tick={{ fill: '#475569', fontSize: 11 }} />
                              <YAxis stroke="#475569" tick={{ fill: '#475569', fontSize: 11 }} domain={[0, 100]} />
                              <Tooltip contentStyle={{ background: '#1a2235', border: '1px solid #2d3748', borderRadius: 8, color: '#f1f5f9', fontSize: 12 }}
                                formatter={(v: any) => [`${typeof v === 'number' ? v.toFixed(1) : v}%`]} />
                              <Legend formatter={(v) => v === 'foto_lama' ? '📷 Foto Lama' : '📱 Foto Baru'} />
                              <ReferenceLine y={95} stroke="#ef4444" strokeDasharray="4 3" label={{ value: '95%', fill: '#ef4444', fontSize: 10 }} />
                              <Line type="monotone" dataKey="foto_lama" stroke="#10b981" strokeWidth={2} dot={false} name="foto_lama" />
                              <Line type="monotone" dataKey="foto_baru" stroke="#f59e0b" strokeWidth={2} dot={false} name="foto_baru" />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      {/* Top 10 table */}
                      {[
                        { title: '📷 Foto Lama — Top 10 Singular Values', data: result.math_data.singular_values_face1.slice(0,10) },
                        { title: '📱 Foto Baru — Top 10 Singular Values', data: result.math_data.singular_values_face2.slice(0,10) },
                      ].map(({ title, data }) => (
                        <div key={title}>
                          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#64748b', marginBottom: '0.5rem' }}>{title}</div>
                          <div style={{ background: '#111827', borderRadius: 12, overflow: 'hidden' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
                              <thead>
                                <tr style={{ borderBottom: '1px solid #1e293b' }}>
                                  {['Rank', 'σ (Singular Value)', 'Variance %'].map(h => (
                                    <th key={h} style={{ padding: '0.6rem 0.75rem', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {data.map((row) => (
                                  <tr key={row.rank} style={{ borderBottom: '1px solid #1a2235' }}>
                                    <td style={{ padding: '0.5rem 0.75rem', color: '#94a3b8' }}>{row.rank}</td>
                                    <td style={{ padding: '0.5rem 0.75rem', fontFamily: "'JetBrains Mono', monospace", color: '#c4b5fd' }}>{row.value.toFixed(3)}</td>
                                    <td style={{ padding: '0.5rem 0.75rem', color: '#6ee7b7' }}>{row.variance_pct.toFixed(2)}%</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Tab: Math */}
                  {activeTab === 'math' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                      <div>
                        <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '0.5rem' }}>
                          🔢 SVD Decomposition: A = U Σ Vᵀ
                        </div>
                        <div className="math-block">{`# Gambar wajah direpresentasikan sebagai matriks 128×128
# lalu didekomposisi menggunakan SVD:

A (128×128) = U (128×128) × Σ (128×128) × Vᵀ (128×128)

# U  = left singular vectors (kolom = eigenfaces kiri)
# Σ  = matriks diagonal berisi singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ
# Vᵀ = right singular vectors (baris = eigenfaces)

# Singular values foto lama (σ₁, σ₂, σ₃):
σ = [${result.math_data.singular_values_face1.slice(0,5).map(d => d.value.toFixed(2)).join(', ')}, ...]

# Singular values foto baru (σ₁, σ₂, σ₃):
σ = [${result.math_data.singular_values_face2.slice(0,5).map(d => d.value.toFixed(2)).join(', ')}, ...]`}</div>
                      </div>

                      <div>
                        <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '0.5rem' }}>
                          📊 PCA — Eigenvalue & Eigenface
                        </div>
                        <div className="math-block">{`# Matriks kovarians gabungan kedua gambar:
# C = (1/n) AᵀA

# Eigenvalue (λ) menunjukkan seberapa penting setiap komponen:
λ₁ = ${result.math_data.eigenvalues[0]?.toFixed(6) ?? 'N/A'}   ← Komponen utama
λ₂ = ${result.math_data.eigenvalues[1]?.toFixed(6) ?? 'N/A'}   ← Komponen sekunder

# Eigenface (v) adalah eigenvector dari matriks kovarians
# Setiap eigenface adalah "pola dasar" wajah manusia
# Shape eigenface: (16384,) → direshape ke (128, 128) untuk visualisasi`}</div>
                      </div>

                      <div>
                        <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '0.5rem' }}>
                          🎯 Proyeksi & Similarity
                        </div>
                        <div className="math-block">{`# Proyeksi ke eigenspace:
# w = eigenfaces @ (face - mean_face)

w₁ (Foto Lama) = [${result.math_data.weights_face1.map(v => v.toFixed(4)).join(', ')}]
w₂ (Foto Baru) = [${result.math_data.weights_face2.map(v => v.toFixed(4)).join(', ')}]

# Cosine Similarity di Eigenspace:
# cos(θ) = (w₁ · w₂) / (‖w₁‖ · ‖w₂‖)
cos_sim  = ${result.metrics.cosine_similarity_eigenspace.toFixed(6)}  → ${(result.metrics.cosine_similarity_eigenspace * 100).toFixed(2)}%

# Euclidean Distance:
# d = ‖w₁ - w₂‖₂
euc_dist = ${result.metrics.euclidean_distance_eigenspace.toFixed(6)}

# Composite Score (weighted average):
# score = 0.45×cos_eigen + 0.25×euc_sim + 0.20×ssim + 0.10×cos_pixel
score = 0.45×${result.metrics.cosine_similarity_eigenspace.toFixed(4)} + 0.25×${result.metrics.euclidean_similarity_norm.toFixed(4)}
      + 0.20×${result.metrics.ssim_pixel.toFixed(4)} + 0.10×${result.metrics.cosine_similarity_pixel.toFixed(4)}
      = ${result.metrics.composite_score.toFixed(4)}  →  ${(result.metrics.composite_score * 100).toFixed(2)}%

# Threshold = ${(threshold * 100).toFixed(0)}%  →  ${isSame ? '✅ ORANG YANG SAMA' : '❌ ORANG YANG BERBEDA'}`}</div>
                      </div>
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          {/* ── Placeholder ── */}
          {!result && !loading && (
            <div className="fade-in" style={{
              textAlign: 'center', padding: '4rem 2rem',
              background: '#111827', borderRadius: 20,
              border: '2px dashed rgba(139,92,246,0.25)',
            }}>
              <div style={{ fontSize: '3.5rem', marginBottom: '1rem' }}>🔬</div>
              <h2 style={{ color: '#e2e8f0', marginBottom: '0.5rem', fontSize: '1.3rem' }}>
                Upload Dua Foto untuk Memulai
              </h2>
              <p style={{ maxWidth: 450, margin: '0 auto', color: '#475569', fontSize: '0.9rem' }}>
                Upload <strong style={{ color: '#a78bfa' }}>Foto Lama</strong> (masa kecil) dan{' '}
                <strong style={{ color: '#60a5fa' }}>Foto Baru</strong> (saat ini) untuk mendeteksi
                kemiripan menggunakan algoritma <strong style={{ color: '#6ee7b7' }}>PCA & SVD (Eigenfaces)</strong>.
              </p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '2rem', flexWrap: 'wrap' }}>
                {[['📐', 'SVD Decomposition'], ['🧮', 'Eigenvalue & Eigenvector'], ['📊', 'PCA Projection'], ['📏', 'Cosine Similarity']].map(([icon, text]) => (
                  <div key={text} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.8rem' }}>{icon}</div>
                    <div style={{ fontSize: '0.75rem', color: '#475569', marginTop: '0.3rem' }}>{text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>

        {/* ── Footer ── */}
        <footer style={{
          textAlign: 'center', padding: '1.5rem',
          borderTop: '1px solid rgba(255,255,255,0.05)',
          color: '#334155', fontSize: '0.8rem',
        }}>
          🎓 Tugas Mata Kuliah Aljabar Linear · Semester 2 · Implementasi PCA &amp; SVD (Eigenfaces) · NumPy · OpenCV
        </footer>
      </div>
    </>
  );
}
