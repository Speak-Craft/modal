from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import librosa, joblib, numpy as np
from feature_extractor import extract_features
import tempfile, os

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = joblib.load("filler_detector_model.pkl")
    print("✅ Model loaded successfully")
    print(f"📊 Model type: {type(model)}")
    if hasattr(model, 'classes_'):
        print(f"📊 Model classes: {model.classes_}")
    if hasattr(model, 'feature_importances_'):
        print(f"📊 Feature importances shape: {model.feature_importances_.shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.post("/predict-filler-words/")
async def predict(audio: UploadFile = File(...)):
    try:
        print("✅ Request received at /predict-filler-words/")

        if not audio.filename:
            return {"error": "No audio file found"}, 400

        print(f"📄 File received: {audio.filename}")
        print(f"📄 Content type: {audio.content_type}")

        temp_path = "temp_audio.wav"
        content = await audio.read()
        with open(temp_path, "wb") as f:
            f.write(content)
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

            try:
                features = extract_features(chunk, sr)
                if features is None:
                    continue
                
                if model is None:
                    continue
                
                prediction_proba = model.predict_proba([features])[0] if hasattr(model, 'predict_proba') else None
                
                # Use a more reasonable threshold for filler detection
                # Instead of using the hard prediction, use probability threshold
                filler_probability = prediction_proba[1] if prediction_proba is not None else 0
                filler_threshold = 0.3  # More lenient threshold
                
                prediction = 1 if filler_probability > filler_threshold else 0
                
                total_chunks += 1
                if prediction == 1:
                    filler_count += 1
                    print(f"✅ FILLER DETECTED! (prob: {filler_probability:.3f})")
                    
            except Exception as e:
                print(f"❌ Error processing chunk: {e}")
                continue

        print(f"\n✅ Total chunks analyzed: {total_chunks}")
        print(f"🗣️ Estimated filler words in clip: {filler_count}")

        return {
            "filler_prediction": filler_count,
            "total_chunks": total_chunks,
            "message": f"Filler word count detected: {filler_count}"
        }

    except Exception as e:
        print("❌ Error in Flask /predict:", str(e))
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)