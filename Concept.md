Konsep Dasar dan Pendekatan Eigenfaces Deteksi kemiripan wajah bertujuan untuk menentukan apakah dua citra wajah memiliki karakteristik yang mirip

- Dalam metode ini, wajah dipandang sebagai data dengan dimensi yang sangat tinggi.

Sebagai contoh, sebuah gambar grayscale berukuran 100 × 100 piksel akan direpresentasikan sebagai satu titik vektor yang memiliki 10.000 fitur (100 × 100 = 10.000).

Karena pemrosesan ruang berdimensi 10.000 sangat sulit dan berat secara komputasi, PCA digunakan untuk mereduksi dimensi gambar tersebut ke ruang yang jauh lebih kecil (misalnya menjadi 50 atau 100 dimensi) namun tetap mempertahankan informasi visual yang paling penting. Pendekatan ini dikenal dengan metode Eigenfaces, di mana PCA mencari arah variasi terbesar dari sekumpulan data wajah untuk membentuk pola utama, atau "wajah dasar" yang disebut eigenfaces. Nantinya, setiap wajah baru akan diproyeksikan ke dalam ruang eigenfaces ini, dan tingkat kemiripan antar wajah dihitung berdasarkan jarak representasi tersebut. Alur Implementasi Lengkap Secara praktik, sistem dibangun dengan tahapan yang sistematis, baik untuk melatih data (training) maupun untuk menguji data wajah baru (testing)

1. Pengumpulan dan Deteksi Wajah Langkah pertama adalah mengumpulkan dataset gambar wajah sebagai data latih yang dikelompokkan dalam folder berdasarkan identitas orangnya

Pada sistem dunia nyata, gambar tidak hanya berisi wajah, sehingga proses Face Detection (seperti menggunakan algoritma Haar Cascade pada OpenCV) sangat disarankan untuk mendeteksi dan memotong (crop) area wajah dari latar belakang

2. Preprocessing Gambar Agar data dapat diproses oleh PCA, semua gambar wajah harus diseragamkan

- Tahapan preprocessing ini meliputi: Mengubah gambar menjadi skala abu-abu (grayscale)

- Melakukan resize ke ukuran standar yang seragam, contohnya 100 × 100 piksel

- Melakukan normalisasi nilai piksel (biasanya ke rentang 0-1 dengan membaginya dengan 255)

- Melakukan flatten, yaitu meratakan matriks gambar (100 × 100) menjadi satu vektor panjang (10.000 elemen)

3. Pembentukan Matriks dan Centering Data Vektor-vektor wajah tersebut kemudian disusun menjadi sebuah matriks data X, di mana jumlah baris adalah total gambar dan jumlah kolom adalah total piksel (misalnya X∈R 200×10000 untuk 200 gambar)

- Sebelum menerapkan PCA, data wajib melalui proses centering, yaitu mengurangkan setiap fitur gambar dengan wajah rata-rata (mean face), sehingga didapatkan matriks data yang terpusat (Xc)

4. Proses PCA menggunakan Singular Value Decomposition (SVD) Proses inti dari PCA dilakukan menggunakan dekomposisi SVD pada data yang sudah di-centering: Xc=UΣVT

- Dalam konteks ini:
Kolom-kolom pada matriks V adalah komponen utama yang bertindak sebagai eigenfaces atau arah utama variasi wajah. Nilai singular pada Σ menunjukkan seberapa krusial peran masing-masing eigenface. Jika kita hanya ingin menggunakan k fitur terpenting (misalnya k=50), maka matriks dapat dipangkas, sehingga wajah berdimensi 10.000 sukses direduksi menjadi representasi ringkas berupa 50 angka penting saja

5. Proyeksi dan Menghitung Kemiripan Wajah yang telah diproses kemudian diproyeksikan ke ruang PCA

- Untuk membandingkan dua wajah, sistem akan mengukur seberapa dekat representasi vektor dari dua wajah tersebut menggunakan perhitungan jarak atau kesamaan

Ada dua metode utama:
- Euclidean Distance: Mengukur jarak garis lurus antar dua vektor wajah. Jarak yang kecil (< ambang batas/ threshold) menandakan wajah mirip

- Cosine Similarity: Mengukur kesamaan arah dari dua vektor. Nilai metrik ini berkisar dari -1 hingga 1. Jika nilainya mendekati 1 (dan berada di atas threshold, contohnya > 0,80), maka kedua wajah dianggap mirip. Cosine similarity lebih sering digunakan dalam deteksi wajah karena lebih fokus pada arah vektor ketimbang jarak absolut.

Dalam implementasinya dengan bahasa Python, proses ini biasanya memanfaatkan pustaka OpenCV (untuk pemrosesan gambar), NumPy (untuk operasi matriks), dan Scikit-Learn (untuk PCA dan menghitung cosine similarity). Sistem juga bisa diterapkan untuk mengenali identitas seseorang dengan cara mencocokkan wajah input dengan seluruh database dan mencari nilai kesamaan tertinggi. Kelebihan dan Keterbatasan Sistem Metode PCA/SVD untuk pengenalan wajah memiliki daya tarik tersendiri karena sangat ringan secara komputasi dibandingkan membandingkan piksel mentah satu per satu, mampu mereduksi dimensi data secara drastis, serta secara matematis sangat mudah dipahami dan divisualisasikan.

Meskipun begitu, metode klasik ini memiliki beberapa keterbatasan signifikan. Pendekatan Eigenfaces sangat sensitif terhadap variasi pencahayaan, posisi wajah, ekspresi, serta sudut pandang (angle) wajah. Metode ini juga mewajibkan wajah harus sudah dalam keadaan sejajar (aligned). Oleh karena itu, tingkat akurasinya tidak sekuat model Deep Learning modern seperti FaceNet atau CNN, dan seringkali membutuhkan tahapan preprocessing tambahan agar bisa bekerja optimal di dunia nyata


## Contoh Implementasi

1. Impor Pustaka dan Fungsi Pra-pemrosesan Gambar Bagian ini menyiapkan pustaka yang dibutuhkan dan membuat fungsi untuk mengubah gambar asli menjadi vektor berdimensi 10.000 yang telah dinormalisasi.

import cv2 
import numpy as np 
from sklearn.decomposition import PCA 
from sklearn.metrics.pairwise import cosine_similarity
import os

# Ukuran standar gambar wajah
IMG_SIZE = (100, 100)  

def load_face_image(path):     
    """     
    Membaca gambar wajah, mengubah ke grayscale,     
    resize, lalu flatten menjadi vektor.     
    """     
    img = cv2.imread(path)      
    if img is None:         
        raise ValueError(f"Gambar tidak ditemukan: {path}")      
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
    resized = cv2.resize(gray, IMG_SIZE)      
    
    # Normalisasi nilai piksel ke rentang 0-1     
    normalized = resized / 255.0      
    
    # Ubah matriks 100x100 menjadi vektor 10000     
    vector = normalized.flatten()      
    
    return vector

2. Memuat Dataset dan Melatih Model PCA Kode ini membaca seluruh gambar wajah yang ada di direktori dataset, lalu menggunakan fungsi PCA(n_components=50) untuk mereduksi dimensi data wajah menjadi 50 fitur utama.

def load_dataset(dataset_path):     
    """     
    Membaca seluruh gambar wajah dari folder dataset.     
    """     
    X = []     
    labels = []      
    
    for person_name in os.listdir(dataset_path):         
        person_folder = os.path.join(dataset_path, person_name)          
        if not os.path.isdir(person_folder):             
            continue          
            
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):                 
                image_path = os.path.join(person_folder, filename)                  
                vector = load_face_image(image_path)                  
                X.append(vector)                 
                labels.append(person_name)      
                
    return np.array(X), np.array(labels)   

# 1. Load dataset 
X, labels = load_dataset("dataset")  

# 2. Latih model PCA 
pca = PCA(n_components=50) 
X_pca = pca.fit_transform(X)  

print("Ukuran data asli:", X.shape) 
print("Ukuran data setelah PCA:", X_pca.shape) 
print("Total explained variance:", np.sum(pca.explained_variance_ratio_))

3. Fungsi Membandingkan Dua Wajah Fungsi ini memproyeksikan dua gambar wajah ke ruang PCA, kemudian menghitung kesamaannya menggunakan cosine similarity untuk memutuskan apakah kedua wajah tersebut mirip berdasarkan threshold tertentu.

def compare_faces(image_path_1, image_path_2, pca, threshold=0.80):     
    """     
    Membandingkan dua gambar wajah menggunakan PCA dan cosine similarity.     
    """     
    face_1 = load_face_image(image_path_1)     
    face_2 = load_face_image(image_path_2)      
    
    # Ubah menjadi bentuk 2D karena PCA membutuhkan input 2D     
    face_1 = face_1.reshape(1, -1)
    face_2 = face_2.reshape(1, -1)      
    
    # Proyeksi ke ruang PCA     
    face_1_pca = pca.transform(face_1)     
    face_2_pca = pca.transform(face_2)      
    
    # Hitung cosine similarity     
    similarity = cosine_similarity(face_1_pca, face_2_pca)      
    
    if similarity >= threshold:         
        result = "Mirip"     
    else:         
        result = "Tidak mirip"      
        
    return similarity, result   

# Contoh penggunaan
similarity, result = compare_faces(     
    "test/wajah_1.jpg",     
    "test/wajah_2.jpg",     
    pca,     
    threshold=0.80 
)  
print("Similarity:", similarity) 
print("Hasil:", result)

4. Implementasi Identifikasi Wajah di Database Untuk mencari identitas suatu wajah, fungsi ini akan membandingkan satu wajah input dengan seluruh representasi wajah yang telah disimpan pada matriks database (X_pca).

def recognize_face(image_path, pca, X_pca, labels, threshold=0.80):     
    """     
    Mencari wajah paling mirip dari database.     
    """     
    face = load_face_image(image_path)     
    face = face.reshape(1, -1)      
    
    # Proyeksi wajah input ke ruang PCA     
    face_pca = pca.transform(face)

    # Hitung cosine similarity terhadap semua wajah di database     
    similarities = cosine_similarity(face_pca, X_pca)      
    
    # Ambil indeks wajah paling mirip     
    best_index = np.argmax(similarities)     
    best_similarity = similarities[best_index]     
    best_label = labels[best_index]      
    
    if best_similarity >= threshold:         
        return best_label, best_similarity     
    else:         
        return "Tidak dikenal", best_similarity   

# Contoh penggunaan
name, score = recognize_face(     
    "test/wajah_baru.jpg",     
    pca,     
    X_pca,     
    labels,     
    threshold=0.80 
)  
print("Hasil identifikasi:", name) 
print("Similarity:", score)

5. Deteksi Wajah (Tahap Opsional Namun Penting) Mengingat PCA sangat sensitif terhadap posisi wajah, sumber menyertakan sebuah fungsi face detection menggunakan model Haar Cascade dari OpenCV. Fungsi ini digunakan untuk mendeteksi dan memotong area wajah sebelum diproses, menggantikan pemanggilan gambar biasa.

def detect_and_crop_face(image_path):     
    """     
    Mendeteksi wajah dari gambar, lalu mengembalikan crop wajah.     
    """     
    img = cv2.imread(image_path)      
    if img is None:         
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")      
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
    face_cascade = cv2.CascadeClassifier(         
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"     
    )      
    
    faces = face_cascade.detectMultiScale(         
        gray,         
        scaleFactor=1.1,         
        minNeighbors=5     
    )      
    
    if len(faces) == 0:         
        raise ValueError("Wajah tidak terdeteksi")      
        
    # Ambil wajah pertama     
    x, y, w, h = faces      
    face_crop = gray[y:y+h, x:x+w]     
    face_resized = cv2.resize(face_crop, IMG_SIZE)     
    face_normalized = face_resized / 255.0      
    
    return face_normalized.flatten()