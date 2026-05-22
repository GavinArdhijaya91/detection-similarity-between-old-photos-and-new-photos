import Head from 'next/head';
import { useState, useCallback, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ReferenceLine, Legend,
} from 'recharts';

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

function ProgressBar({ value, color, label }: { value: number; color: string; label: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const displayColor = pct >= 70 ? 'bg-emerald-500' : pct >= 55 ? 'bg-amber-500' : 'bg-red-500';
  const textDisplayColor = pct >= 70 ? 'text-emerald-500' : pct >= 55 ? 'text-amber-500' : 'text-red-500';
  
  return (
    <div className="my-3">
      <div className="flex justify-between items-center mb-1.5 text-xs">
        <span className="text-gray-400">{label}</span>
        <span className={`${textDisplayColor} font-mono font-bold`}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-gray-800/50 overflow-hidden border border-gray-800">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${displayColor}`}
          style={{ width: `${pct}%` }}
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
    <div className="flex flex-col gap-2">
      <div className="text-xs font-bold tracking-widest uppercase text-indigo-200/65">
        {label}
      </div>

      {photo ? (
        <div 
          className="relative rounded-2xl overflow-hidden border-2 border-indigo-500/50 cursor-pointer group"
          onClick={() => inputRef.current?.click()}
        >
          <img src={photo} alt={label} className="w-full h-[220px] object-cover block" />
          <div className="absolute inset-0 bg-transparent flex items-center justify-center transition-colors duration-300 group-hover:bg-black/50">
            <span className="text-white text-sm font-semibold opacity-0 transition-opacity duration-300 group-hover:opacity-100">
              Ganti Foto
            </span>
          </div>
        </div>
      ) : (
        <div
          className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-6 min-h-[220px] text-center select-none cursor-pointer transition-all duration-300
            ${dragging 
              ? 'border-indigo-500 bg-indigo-500/10 shadow-[0_0_20px_rgba(99,102,241,0.1)]' 
              : 'border-indigo-500/30 bg-gray-900/40 hover:border-indigo-500/60 hover:bg-indigo-500/5'}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <div className="text-4xl mb-3 opacity-70">📷</div>
          <div className="text-sm font-semibold text-gray-200 mb-1">Klik atau Drag & Drop</div>
          <div className="text-xs text-gray-400">PNG, JPG, WEBP • Maks 10MB</div>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
    </div>
  );
}

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
  const scoreColor = score >= 0.70 ? 'text-emerald-500' : score >= 0.55 ? 'text-amber-500' : 'text-red-500';

  return (
    <>
      <Head>
        <title>FaceMatch PCA/SVD — Deteksi Kemiripan Foto</title>
      </Head>

      <div className="min-h-screen flex flex-col font-inter">
        
        {/* ── Header ── */}
        <header className="relative pt-16 pb-12 overflow-hidden border-b [border-image:linear-gradient(to_right,transparent,--theme(--color-indigo-500/.25),transparent)1]">
          {/* Background Elements */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] opacity-20 pointer-events-none blur-[100px] bg-indigo-600 rounded-full mix-blend-screen" />
          
          <div className="max-w-6xl mx-auto px-4 sm:px-6 relative z-10 text-center">
            <h1 className="text-4xl md:text-5xl lg:text-6xl animate-[gradient_6s_linear_infinite] bg-[linear-gradient(to_right,var(--color-gray-200),var(--color-indigo-200),var(--color-gray-50),var(--color-indigo-300),var(--color-gray-200))] bg-[length:200%_auto] bg-clip-text text-transparent font-bold tracking-tight mb-4">
              FaceMatch <span className="opacity-50">/</span> PCA & SVD
            </h1>
            <p className="text-lg text-indigo-200/65 max-w-2xl mx-auto mb-8 font-medium">
              Deteksi Kemiripan Foto Lama vs Foto Baru menggunakan implementasi Eigenfaces Aljabar Linear.
            </p>
            
            <div className="flex flex-wrap justify-center gap-2">
              {['📐 Aljabar Linear', '🧮 Eigenvalue', '🔢 SVD', '📊 PCA'].map((b, i) => (
                <span key={i} className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                  {b}
                </span>
              ))}
            </div>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-4 sm:px-6 py-12 flex-1 w-full">
          
          {/* ── Upload Section ── */}
          <section className="mb-12 max-w-4xl mx-auto">
            <div className="flex items-center gap-3 text-lg font-bold text-gray-200 mb-6 after:content-[''] after:flex-1 after:h-px after:bg-gray-800">
              📸 Upload Foto
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DropZone label="Foto Lama (Masa Kecil)" photo={photo1} onPhoto={setPhoto1} />
              <DropZone label="Foto Baru (Saat Ini)"   photo={photo2} onPhoto={setPhoto2} />
            </div>

            {/* Threshold */}
            <div className="mt-6 p-6 rounded-2xl bg-gray-900/50 border border-gray-800 backdrop-blur-sm">
              <div className="flex justify-between items-center mb-3">
                <span className="text-sm font-semibold text-gray-400">
                  ⚖️ Ambang Batas Kemiripan (Threshold)
                </span>
                <span className="font-mono text-indigo-300 font-bold text-sm">
                  {(threshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range" min={0.40} max={0.95} step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-2 font-medium">
                <span>40% (Longgar)</span><span>70% (Default)</span><span>95% (Ketat)</span>
              </div>
            </div>

            {/* Analyze Button */}
            <div className="mt-6">
              <button
                className="w-full inline-flex items-center justify-center whitespace-nowrap rounded-xl text-base font-medium transition-all px-6 py-4 group bg-linear-to-t from-indigo-600 to-indigo-500 bg-[length:100%_100%] bg-[bottom] text-white shadow-[inset_0px_1px_0px_0px_--theme(--color-white/.16)] hover:bg-[length:100%_150%] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={handleAnalyze}
                disabled={!photo1 || !photo2 || loading}
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Menganalisis SVD & PCA...
                  </span>
                ) : (
                  <span className="relative inline-flex items-center">
                    🔍 Analisis Kemiripan
                    <span className="ml-1 tracking-normal text-white/50 transition-transform group-hover:translate-x-0.5">-&gt;</span>
                  </span>
                )}
              </button>
            </div>

            {error && (
              <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm font-medium">
                ❌ {error}
              </div>
            )}
          </section>

          {/* ── Divider ── */}
          {(result || (!result && !loading)) && (
            <div className="border-t py-6 [border-image:linear-gradient(to_right,transparent,--theme(--color-gray-800),transparent)1] md:py-10"></div>
          )}

          {/* ── Results ── */}
          {result && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              
              {/* Main verdict */}
              <section className="mb-12 max-w-4xl mx-auto">
                <div className="flex items-center gap-3 text-lg font-bold text-gray-200 mb-6 after:content-[''] after:flex-1 after:h-px after:bg-gray-800">
                  🎯 Hasil Analisis
                </div>

                <div className={`p-10 text-center rounded-3xl border bg-gray-900/40 backdrop-blur-md transition-all duration-700
                    ${isSame 
                      ? 'border-emerald-500/30 shadow-[0_0_40px_-10px_rgba(16,185,129,0.2)]' 
                      : 'border-red-500/30 shadow-[0_0_40px_-10px_rgba(239,68,68,0.2)]'}`}
                >
                  <div className="text-6xl mb-4 drop-shadow-md">
                    {result.decision.verdict_icon}
                  </div>
                  <div className={`text-2xl md:text-3xl font-black mb-2 tracking-tight ${isSame ? 'text-emerald-400' : 'text-red-400'}`}>
                    {result.decision.verdict}
                  </div>
                  <div className={`text-6xl md:text-7xl font-black my-4 leading-none tracking-tighter ${scoreColor}`}>
                    {(score * 100).toFixed(1)}%
                  </div>
                  <div className="text-gray-400 text-sm font-medium mb-6 uppercase tracking-widest">
                    Composite Similarity Score
                  </div>

                  <div className="flex justify-center gap-3 flex-wrap">
                    <span className={`inline-flex items-center px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border ${isSame ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
                      {result.decision.level}
                    </span>
                    <span className="inline-flex items-center px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                      Kepercayaan: {result.decision.confidence}
                    </span>
                  </div>
                </div>
              </section>

              {/* Detail Metrics Tabs */}
              <section className="mb-12 max-w-5xl mx-auto">
                <div className="flex items-center gap-3 text-lg font-bold text-gray-200 mb-6 after:content-[''] after:flex-1 after:h-px after:bg-gray-800">
                  📊 Detail Analisis
                </div>

                <div className="p-6 md:p-8 rounded-3xl bg-gray-900/50 border border-gray-800/80 backdrop-blur-sm">
                  
                  {/* Tabs using Secondary Button Style */}
                  <div className="flex gap-3 border-b border-gray-800 pb-5 mb-8 overflow-x-auto">
                    {(['metrics', 'svd', 'math'] as const).map((tab) => {
                      const isActive = activeTab === tab;
                      return (
                        <button
                          key={tab}
                          className={`inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-semibold transition-all px-5 py-2.5 relative 
                            ${isActive 
                              ? 'bg-linear-to-b from-gray-800 to-gray-800/60 bg-[length:100%_100%] bg-[bottom] text-gray-100 before:pointer-events-none before:absolute before:inset-0 before:rounded-[inherit] before:border before:border-transparent before:[background:linear-gradient(to_right,var(--color-gray-700),var(--color-gray-600),var(--color-gray-700))_border-box] before:[mask-composite:exclude_!important] before:[mask:linear-gradient(white_0_0)_padding-box,_linear-gradient(white_0_0)]' 
                              : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                            }`}
                          onClick={() => setActiveTab(tab)}
                        >
                          {{ metrics: '📏 Metrik Kemiripan', svd: '📉 Singular Values', math: '🔢 Matematika' }[tab]}
                        </button>
                      );
                    })}
                  </div>

                  {/* Tab: Metrics */}
                  {activeTab === 'metrics' && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                      <div>
                        <div className="text-xs font-bold text-indigo-300 mb-4 uppercase tracking-widest flex items-center gap-2">
                          <svg className="fill-indigo-400" xmlns="http://www.w3.org/2000/svg" width={16} height={16} viewBox="0 0 24 24"><path d="M0 0h14v17H0V0Zm2 2v13h10V2H2Z" /><path fillOpacity=".48" d="m16.295 5.393 7.528 2.034-4.436 16.412L5.87 20.185l.522-1.93 11.585 3.132 3.392-12.55-5.597-1.514.522-1.93Z" /></svg>
                          Metrik Eigenspace (PCA)
                        </div>
                        <ProgressBar value={Math.max(0, result.metrics.cosine_similarity_eigenspace)} label="Cosine Similarity (Eigenspace)" color="" />
                        <ProgressBar value={result.metrics.euclidean_similarity_norm} label="Euclidean Similarity (Normalized)" color="" />
                        
                        <div className="flex justify-between items-center py-3 border-b border-gray-800/50 mt-4 text-sm">
                          <span className="text-gray-400 font-medium">Euclidean Distance</span>
                          <span className="font-mono text-gray-200 font-bold">{result.metrics.euclidean_distance_eigenspace.toFixed(4)}</span>
                        </div>
                      </div>

                      <div>
                        <div className="text-xs font-bold text-indigo-300 mb-4 uppercase tracking-widest flex items-center gap-2">
                          <svg className="fill-indigo-400" xmlns="http://www.w3.org/2000/svg" width={16} height={16} viewBox="0 0 24 24"><path d="M0 0h14v17H0V0Zm2 2v13h10V2H2Z" /><path fillOpacity=".48" d="m16.295 5.393 7.528 2.034-4.436 16.412L5.87 20.185l.522-1.93 11.585 3.132 3.392-12.55-5.597-1.514.522-1.93Z" /></svg>
                          Metrik Pixel
                        </div>
                        <ProgressBar value={result.metrics.ssim_pixel} label="SSIM (Structural Similarity)" color="" />
                        <ProgressBar value={Math.max(0, result.metrics.cosine_similarity_pixel)} label="Cosine Similarity (Pixel)" color="" />
                        <ProgressBar value={result.metrics.composite_score} label="Composite Score (Final)" color="" />
                      </div>
                    </div>
                  )}

                  {/* Tab: SVD */}
                  {activeTab === 'svd' && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div className="col-span-1 lg:col-span-2">
                        <div className="text-xs font-bold text-indigo-300 mb-4 uppercase tracking-widest">
                          Singular Values σ per Gambar
                        </div>
                        <div className="bg-gray-950 rounded-2xl p-4 h-[300px] border border-gray-800/60">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.math_data.singular_values_face1.map((d, i) => ({
                              rank: d.rank,
                              foto_lama: d.value,
                              foto_baru: result.math_data.singular_values_face2[i]?.value ?? 0,
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                              <XAxis dataKey="rank" stroke="#6b7280" tick={{ fill: '#6b7280', fontSize: 11 }} />
                              <YAxis stroke="#6b7280" tick={{ fill: '#6b7280', fontSize: 11 }} />
                              <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 12, color: '#f3f4f6' }} />
                              <Legend />
                              <Line type="monotone" dataKey="foto_lama" stroke="#8b5cf6" strokeWidth={2.5} dot={{ r: 2, fill: '#8b5cf6' }} />
                              <Line type="monotone" dataKey="foto_baru" stroke="#3b82f6" strokeWidth={2.5} dot={{ r: 2, fill: '#3b82f6' }} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Tab: Math */}
                  {activeTab === 'math' && (
                    <div className="flex flex-col gap-6">
                      {[
                        { title: 'SVD Decomposition: A = U Σ Vᵀ', code: `A (128×128) = U × Σ × Vᵀ\n\nσ (Lama) = [${result.math_data.singular_values_face1.slice(0,4).map(d => d.value.toFixed(2)).join(', ')}, ...]\nσ (Baru) = [${result.math_data.singular_values_face2.slice(0,4).map(d => d.value.toFixed(2)).join(', ')}, ...]` },
                        { title: 'Cosine Similarity di Eigenspace', code: `cos_sim = ${result.metrics.cosine_similarity_eigenspace.toFixed(6)}\nScore   = ${(result.metrics.composite_score * 100).toFixed(2)}%\nVerdict = ${isSame ? '✅ SAMA' : '❌ BEDA'}` }
                      ].map((block, i) => (
                        <div key={i}>
                          <div className="text-sm font-bold text-gray-200 mb-2">{block.title}</div>
                          <div className="bg-gray-950 border border-gray-800/80 border-l-4 border-l-indigo-500 rounded-lg p-5 font-mono text-xs text-indigo-200/80 whitespace-pre overflow-x-auto">
                            {block.code}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          {/* ── Placeholder ── */}
          {!result && !loading && (
            <div className="py-20 text-center animate-in fade-in duration-700">
              <div className="text-5xl mb-6 opacity-80 drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]">🧬</div>
              <h2 className="text-2xl font-bold text-gray-200 mb-3 tracking-tight">
                Upload Dua Foto untuk Memulai
              </h2>
              <p className="max-w-md mx-auto text-indigo-200/65 text-sm font-medium leading-relaxed">
                Platform akan mendeteksi kemiripan wajah menggunakan algoritma <strong className="text-indigo-300 font-semibold">PCA & SVD (Eigenfaces)</strong> yang canggih.
              </p>
            </div>
          )}
        </main>
        
        {/* Footer */}
        <footer className="py-6 text-center text-xs font-medium text-gray-500 border-t border-gray-800/50">
          Implementasi Aljabar Linear · Next.js · Tailwind CSS v4
        </footer>
      </div>
    </>
  );
}
