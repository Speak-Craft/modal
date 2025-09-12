#!/usr/bin/env python3

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
import uvicorn

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the loudness model
_BASE_DIR = Path(__file__).resolve().parent
model_path = _BASE_DIR / "loudness-model" / "models" / "loudness_model.pkl"

print(f"🔍 Looking for model at: {model_path}")

model = None
if model_path.exists():
    try:
        model = joblib.load(str(model_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
else:
    print(f"❌ Model file not found at: {model_path}")

def extract_features(path):
    try:
        y, sr = sf.read(path, dtype="float32")
    except Exception as e:
        print(f"⚠️ Error reading audio file {path}: {e}")
        # Try with librosa as fallback
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=True)
        except Exception as e2:
            print(f"⚠️ Error with librosa fallback: {e2}")
            raise e
    
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

@app.get("/")
def root():
    return {"message": "Loudness Analysis API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict-loudness")
async def predict_loudness(file: UploadFile = File(...)):
    try:
        print(f"🔍 Received file: {file.filename}, content_type: {file.content_type}, size: {file.size}")
        
        if model is None:
            return {
                "status": "error",
                "error": "Model not loaded. Ensure loudness_model.pkl is present and compatible.",
            }

        # Check if file is empty
        content = await file.read()
        if len(content) == 0:
            return {
                "status": "error",
                "error": "Empty audio file received. Please record some audio first.",
            }
        
        print(f"📁 File size: {len(content)} bytes")

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
        
        print(f"🎵 Using file extension: {file_extension}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        print(f"💾 Saved temporary file: {tmp_path}")
        
        features, rms = extract_features(tmp_path)
        print(f"📊 Extracted features: {features}, RMS: {rms}")
        
        pred = model.predict([features])[0]
        category = rms_to_category(rms, pred)
        
        print(f"🎯 Prediction: {pred}, Category: {category}")

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
        print(f"❌ Error in predict_loudness: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Loudness Analysis Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
