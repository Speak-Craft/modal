from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import librosa, joblib, numpy as np
from feature_extractor import extract_features
import tempfile, os
from pathlib import Path

app = FastAPI()

# CORS similar to test.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent / "filler_detector_model.pkl"

# Load model safely so the server can still start even if loading fails
try:
    model = joblib.load(str(MODEL_PATH))
except Exception as e:
    print(f"❌ Failed to load model at {MODEL_PATH}: {e}")
    model = None

@app.post("/predict-filler-words")
async def predict(audio: UploadFile = File(...)):
    try:
        if model is None:
            return {"error": "Model not loaded. Please verify filler_detector_model.pkl compatibility with the current Python/libs."}

        print("✅ Request received at /predict-filler-words")

        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            temp_path = tmp.name
        print(f"✅ Saved file to {temp_path}")

        y, sr = librosa.load(temp_path, sr=16000)
        chunk_duration_sec = 1.0
        chunk_size = int(chunk_duration_sec * sr)

        filler_count = 0
        total_chunks = 0

        # Split into chunks and predict on each
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]

            # Skip too short chunks (less than 0.3 sec)
            if len(chunk) < int(0.3 * sr):
                continue

            features = extract_features(chunk, sr)
            if features is None:
                continue

            prediction = model.predict([features])[0]
            total_chunks += 1
            if prediction == 1:
                filler_count += 1

        print(f"\n✅ Total chunks analyzed: {total_chunks}")
        print(f"🗣️ Estimated filler words in clip: {filler_count}")

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        return {
            "filler_prediction": filler_count,
            "total_chunks": total_chunks,
            "message": f"Filler word count detected: {filler_count}",
        }

    except Exception as e:
        print("❌ Error in /predict-filler-words:", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
