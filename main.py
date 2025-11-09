from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import warnings

# ==============================
# 🧹 Bersihkan warning
# ==============================
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# 🚀 Inisialisasi FastAPI
# ==============================
app = FastAPI(title="AI BISINDO Backend", version="1.0")

# ==============================
# 🌐 Konfigurasi CORS
# ==============================
origins = [
    "http://localhost:3000",             # untuk development
    "https://cerdas-isyarat.vercel.app", # frontend di Vercel
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🧠 PRELOAD MODEL SEKALI SAJA
# ==============================
try:
    MODEL_PATH = "model/rf_bisindo_99.pkl"
    clf = joblib.load(MODEL_PATH)
    print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model tidak ditemukan di {MODEL_PATH}")
    raise SystemExit("File model tidak ditemukan — pastikan path benar.")

# ==============================
# 📦 Struktur Input
# ==============================
class LandmarkData(BaseModel):
    landmarks: list[float]

# ==============================
# 🔮 Endpoint Prediksi
# ==============================
@app.post("/predict_landmarks")
async def predict_from_landmarks(data: LandmarkData):
    if not data.landmarks or len(data.landmarks) != 126:
        return {"prediction": "Tangan Tidak Valid"}

    landmarks_np = np.array(data.landmarks).reshape(1, -1)
    prediction = clf.predict(landmarks_np)
    return {"prediction": prediction[0]}

# ==============================
# 🌍 Endpoint Root
# ==============================
@app.get("/")
def root():
    return {"status": "AI Backend is running 🚀"}
