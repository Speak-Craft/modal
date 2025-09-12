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

print(f"🔍 Looking for model in: {[str(p) for p in _CANDIDATES]}")

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
    """
    Read audio and compute simple features. Prefer librosa for container formats
    like WebM/MP4; fall back to soundfile for WAV/FLAC.

    Uses up to the last 1.0 second of audio to better reflect speaking volume
    in short streaming chunks.
    """
    y = None
    sr = None

    # Decide reader order based on extension
    ext = str(path).lower()
    prefer_librosa = any(x in ext for x in [".webm", ".mp4", ".m4a", ".ogg"])  # container formats

    def _read_with_librosa(p):
        import librosa  # local import to avoid mandatory dependency at import time
        yy, ssr = librosa.load(p, sr=16000, mono=True)  # resample to stable rate
        return yy.astype(np.float32), ssr

    def _read_with_sf(p):
        yy, ssr = sf.read(p, dtype="float32")
        # If multi-channel, downmix
        if hasattr(yy, "ndim") and yy.ndim > 1:
            yy = yy.mean(axis=1)
        # Resample if needed to 16k for stable window sizing
        try:
            if ssr != 16000:
                import librosa
                yy = librosa.resample(yy, orig_sr=ssr, target_sr=16000)
                ssr = 16000
        except Exception:
            pass
        return yy, ssr

    try:
        if prefer_librosa:
            y, sr = _read_with_librosa(path)
        else:
            y, sr = _read_with_sf(path)
    except Exception as e_first:
        print(f"⚠️ Primary reader failed for {path}: {e_first}")
        # Fallback to the other reader
        try:
            if prefer_librosa:
                y, sr = _read_with_sf(path)
            else:
                y, sr = _read_with_librosa(path)
        except Exception as e_second:
            print(f"⚠️ Secondary reader also failed: {e_second}")
            raise e_first

    # Ensure up to a 3-second analysis window from the tail (matches client cadence)
    if sr is None or sr <= 0:
        sr = 16000
    window_samples = int(sr * 3.0)
    if len(y) > window_samples:
        y = y[-window_samples:]

    # Guard against silence/NaNs
    if not np.any(np.isfinite(y)):
        y = np.zeros(window_samples, dtype=np.float32)

    # Feature calculations
    peak = float(np.max(np.abs(y)) + 1e-8)
    mean_abs = float(np.mean(np.abs(y)))
    lufs_like = 20 * np.log10(mean_abs + 1e-8)
    zcr = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2)
    # Simple spectral centroid proxy in time domain (not true centroid)
    centroid = float(np.sum(np.arange(len(y)) * np.abs(y)) / (np.sum(np.abs(y)) + 1e-8))
    rms = float(np.sqrt(np.mean(y ** 2)))

    return [peak, lufs_like, zcr, centroid], rms

def rms_to_category(rms, prediction):
    """
    Match labels/thresholds from utills/audio_features.py:
    - if rms < 0.05 => "Low / Silent"
    - else model prediction < 0.15 => "Acceptable" else "Too Loud"
    """
    if rms < 0.05:
        return "Low / Silent"
    # Fallback: treat very high RMS as "Too Loud" even if model underestimates
    # This helps surface all labels in real-time usage.
    if rms >= 0.20:
        return "Too Loud"
    try:
        pred_value = float(prediction) if hasattr(prediction, "__float__") else prediction
    except Exception:
        pred_value = prediction
    return "Acceptable" if pred_value < 0.15 else "Too Loud"

@app.post("/predict-loudness")
async def predict_loudness(file: UploadFile = File(...)):
    try:
        if model is None:
            return {
                "status": "error",
                "error": "Model not loaded. Ensure loudness_model.pkl is present and compatible.",
            }

        # Determine file extension from content type or filename
        file_extension = ".wav"
        if file.content_type:
            if "webm" in file.content_type:
                file_extension = ".webm"
            elif "mp4" in file.content_type:
                file_extension = ".mp4"
            elif "ogg" in file.content_type:
                file_extension = ".ogg"
        elif file.filename:
            if file.filename.endswith(".webm"):
                file_extension = ".webm"
            elif file.filename.endswith(".mp4"):
                file_extension = ".mp4"
            elif file.filename.endswith(".ogg"):
                file_extension = ".ogg"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
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
