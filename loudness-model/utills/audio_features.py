import sys
import joblib
import numpy as np
import librosa

# Load the trained multi-feature regression model
model = joblib.load("loudness_model.pkl")

def extract_features(path):
    """
    Extract features required by the model:
    - peak, lufs, zero-crossing rate, spectral centroid
    - RMS (for silence detection)
    """
    y, sr = librosa.load(path, sr=None)
    
    peak = np.max(np.abs(y))
    lufs = 20 * np.log10(np.mean(np.abs(y)) + 1e-6)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    
    return [peak, lufs, zcr, centroid], rms

if __name__ == "__main__":
    audio_path = sys.argv[1]
    try:
        features, rms = extract_features(audio_path)
        # Optional: show category for quick testing
        if rms < 0.05:
            category = "Low / Silent"
        else:
            prediction = model.predict([features])[0]
            category = "Acceptable" if prediction < 0.15 else "Too Loud"
        print(category)
    except Exception as e:
        print("error:", str(e))
