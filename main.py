# Lokasi: SKRIPSI1/ai-backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import warnings

# Mengabaikan warning versi scikit-learn agar terminal lebih bersih
warnings.filterwarnings("ignore", category=UserWarning)

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Konfigurasi CORS untuk mengizinkan koneksi dari frontend Next.js Anda
origins = [
    "http://localhost:3000",  # Alamat default Next.js
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Muat model AI satu kali saja saat server pertama kali dijalankan
try:
    model_path = 'model/rf_bisindo_99.pkl'
    clf = joblib.load(model_path)
except FileNotFoundError:
    print(f"FATAL ERROR: File model tidak ditemukan di {model_path}")
    print("Pastikan folder 'model' dan file '.pkl' sudah ada di dalam 'ai-backend'.")
    exit()

# Definisikan struktur data yang akan diterima dari frontend
class LandmarkData(BaseModel):
    landmarks: list[float]

# Buat API endpoint untuk prediksi
@app.post("/predict_landmarks")
async def predict_from_landmarks(data: LandmarkData):
    # Cek apakah data landmark valid (2 tangan = 21*3 + 21*3 = 126 koordinat)
    if data.landmarks and len(data.landmarks) == 126:
        # Ubah data list menjadi numpy array yang bisa dibaca model
        landmarks_np = np.array(data.landmarks).reshape(1, -1)
        
        # Lakukan prediksi menggunakan model yang sudah dimuat
        prediction = clf.predict(landmarks_np)
        predicted_label = prediction[0]
        
        # Kirim hasilnya kembali ke frontend dalam format JSON
        return {"prediction": predicted_label}
        
    return {"prediction": "Tangan Tidak Valid"}

# Pesan saat server berhasil berjalan (opsional)
@app.get("/")
def read_root():
    return {"status": "AI Backend is running"}