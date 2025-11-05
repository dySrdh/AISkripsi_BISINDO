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

# ==============================
# 🔧 KONFIGURASI CORS
# ==============================
origins = [
    "http://localhost:3000",              # untuk development lokal
    "https://cerdas-isyarat.vercel.app",  # frontend production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex="https://.*\.vercel\.app",  # izinkan semua subdomain vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🤖 MUAT MODEL MACHINE LEARNING
# ==============================
try:
    model_path = 'model/rf_bisindo_99.pkl'
    clf = joblib.load(model_path)
    print(f"✅ Model berhasil dimuat dari: {model_path}")
except FileNotFoundError:
    print(f"❌ FATAL ERROR: File model tidak ditemukan di {model_path}")
    print("Pastikan folder 'model' dan file '.pkl' sudah ada di dalam 'ai-backend'.")
    exit()

# ==============================
# 🧠 STRUKTUR DATA INPUT DARI FRONTEND
# ==============================
class LandmarkData(BaseModel):
    landmarks: list[float]

# ==============================
# 🔮 ENDPOINT UNTUK PREDIKSI
# ==============================
@app.post("/predict_landmarks")
async def predict_from_landmarks(data: LandmarkData):
    # Validasi jumlah landmark (2 tangan = 21*3 + 21*3 = 126 koordinat)
    if data.landmarks and len(data.landmarks) == 126:
        landmarks_np = np.array(data.landmarks).reshape(1, -1)
        prediction = clf.predict(landmarks_np)
        predicted_label = prediction[0]
        return {"prediction": predicted_label}
    else:
        return {"prediction": "Tangan Tidak Valid"}

# ==============================
# 🌐 ENDPOINT UTAMA UNTUK CEK STATUS SERVER
# ==============================
@app.get("/")
def read_root():
    return {"status": "AI Backend is running 🚀"}
