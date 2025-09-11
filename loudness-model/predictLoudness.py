from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path

app = FastAPI()

# CORS similar to other services
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try loading model from common locations/names
_BASE_DIR = Path(__file__).resolve().parent
_CANDIDATES = [
    _BASE_DIR / "loudness_model.pkl",
    _BASE_DIR / "loudess_model.pkl",
    _BASE_DIR / "models" / "loudness_model.pkl",
    _BASE_DIR / "models" / "loudess_model.pkl",
]

model = None
for _p in _CANDIDATES:
    try:
        if _p.exists():
            model = joblib.load(str(_p))
            break
    except Exception as _e:
        print(f"⚠️ Failed to load model at {_p}: {_e}")
if model is None:
    print(f"❌ Could not find/load loudness model in: {[str(p) for p in _CANDIDATES]}")

def extract_features(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) > 8000:
        y = y[:8000]

    peak = np.max(np.abs(y))
    lufs = 20 * np.log10(np.mean(np.abs(y)) + 1e-6)
    zcr = np.mean(np.abs(np.diff(np.sign(y)))) / 2
    centroid = np.sum(np.arange(len(y)) * np.abs(y)) / (np.sum(np.abs(y)) + 1e-6)
    rms = np.sqrt(np.mean(y**2))

    return [peak, lufs, zcr, centroid], rms

def rms_to_category(rms, prediction):
    if rms < 0.05:
        return "Low / Silent"
    elif prediction < 0.15:
        return "Acceptable"
    else:
        return "Too Loud"

@app.post("/predict-loudness")
async def predict_loudness(file: UploadFile = File(...)):
    try:
        if model is None:
            return {
                "status": "error",
                "error": "Model not loaded. Ensure loudness_model.pkl is present and compatible.",
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        features, rms = extract_features(tmp_path)
        pred = model.predict([features])[0]
        category = rms_to_category(rms, pred)

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {
            "status": "success",
            "category": category,
            "rms": float(rms),
            "prediction": float(pred) if hasattr(pred, "__float__") else pred,
            "features": {
                "peak": float(features[0]),
                "lufs": float(features[1]),
                "zcr": float(features[2]),
                "centroid": float(features[3]),
            },
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
