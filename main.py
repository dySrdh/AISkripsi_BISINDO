from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import joblib
import warnings

# ==============================
# ⚙️ Konfigurasi Dasar
# ==============================
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="BISINDO AI Backend")

# ==============================
# 🔒 FIX CORS (untuk Vercel & Railway)
# ==============================
origins = [
    "http://localhost:3000",               # Development lokal
    "https://cerdas-isyarat.vercel.app",   # Frontend production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # hanya domain yang diizinkan
    allow_credentials=True,
    allow_methods=["*"],           # izinkan semua method (GET, POST, OPTIONS, dll)
    allow_headers=["*"],           # izinkan semua header
)

# Tambahan explicit handler untuk OPTIONS (fix Railway 502)
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """Menangani preflight OPTIONS agar tidak 502 di Railway"""
    response = JSONResponse(content={"message": "CORS preflight OK"})
    response.headers["Access-Control-Allow-Origin"] = "https://cerdas-isyarat.vercel.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ==============================
# 🧠 Muat Model Machine Learning
# ==============================
try:
    model_path = "model/rf_bisindo_99.pkl"
    clf = joblib.load(model_path)
    print(f"✅ Model berhasil dimuat dari: {model_path}")
except FileNotFoundError:
    print(f"❌ FATAL ERROR: File model tidak ditemukan di {model_path}")
    print("Pastikan folder 'model' dan file '.pkl' sudah ada di dalam 'ai-backend'.")
    clf = None


# ==============================
# 📦 Struktur Data Input
# ==============================
class LandmarkData(BaseModel):
    landmarks: list[float]


# ==============================
# 🔮 Endpoint Prediksi
# ==============================
@app.post("/predict_landmarks")
async def predict_from_landmarks(data: LandmarkData):
    if not clf:
        return {"error": "Model belum dimuat di server"}

    # Validasi input
    if not data.landmarks or len(data.landmarks) != 126:
        return {"prediction": "Tangan Tidak Valid"}

    landmarks_np = np.array(data.landmarks).reshape(1, -1)
    prediction = clf.predict(landmarks_np)
    predicted_label = prediction[0]
    return {"prediction": predicted_label}


# ==============================
# 🌐 Endpoint Utama
# ==============================
@app.get("/")
def read_root():
    return {"status": "AI Backend is running 🚀"}


# ==============================
# 🚀 Entry Point (optional jika dijalankan manual)
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
