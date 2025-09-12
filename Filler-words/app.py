from fastapi import FastAPI, File, UploadFile, Form
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
async def predict(audio: UploadFile = File(...), transcript: str | None = Form(None)):
    try:
        print("✅ Request received at /predict-filler-words")

        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            temp_path = tmp.name
        print(f"✅ Saved file to {temp_path}")

        y, sr = librosa.load(temp_path, sr=16000)

        # Whole-clip feature extraction (no chunking)
        features = extract_features(y, sr)
        filler_prediction = 0
        filler_probability = None
        if model is not None and features is not None:
            try:
                filler_prediction = int(model.predict([features])[0])
                # Optionally include probability if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([features])[0]
                    filler_probability = float(max(proba))
            except Exception as _:
                pass

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        # Optional text-based filler counting using provided transcript
        detected_fillers = []
        filler_text_count = 0
        if transcript:
            fillers = [
                "um", "uh", "erm", "hmm",
                "like", "you know", "i mean",
                "sort of", "kind of", "basically",
                "actually", "literally", "so", "well", "okay", "right"
            ]
            normalized = f" {transcript.lower()} ".replace("\n", " ")
            import re
            normalized = re.sub(r"\s+", " ", normalized)
            normalized = re.sub(r"[\.,!?;:()\"']", " ", normalized)

            for fw in fillers:
                pattern = rf" {re.escape(fw)} "
                matches = re.findall(pattern, normalized)
                if matches:
                    detected_fillers.append({"word": fw, "count": len(matches)})
                    filler_text_count += len(matches)

        # Return success payload
        return {
            "status": "success",
            "modelLoaded": model is not None,
            "filler_prediction": filler_prediction if model is not None else 0,
            "filler_probability": filler_probability,
            "detected_fillers": detected_fillers,
            "filler_text_count": filler_text_count,
            "message": (f"Filler present: {bool(filler_prediction)}" if model is not None else "No model prediction; using transcript-based counts if provided."),
        }

    except Exception as e:
        print("❌ Error in /predict-filler-words:", str(e))
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
