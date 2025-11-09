from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="BISINDO AI Backend")

# ==============================
# ✅ FIX CORS GLOBAL (Railway + Vercel)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cerdas-isyarat.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ini opsional, tapi bantu banget Railway
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
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

    if not data.landmarks or len(data.landmarks) != 126:
        return {"prediction": "Tangan Tidak Valid"}

    landmarks_np = np.array(data.landmarks).reshape(1, -1)
    prediction = clf.predict(landmarks_np)
    return {"prediction": prediction[0]}


# ==============================
# 🌐 Endpoint Utama
# ==============================
@app.get("/")
def read_root():
    return {"status": "AI Backend is running 🚀"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
