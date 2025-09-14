import librosa
import numpy as np
from joblib import load
import noisereduce as nr
import soundfile as sf
from typing import Tuple, List, Dict, Any

model = None
clf = load("./mlp_classifier.pkl")
scaler = load("./scaler.pkl")


def denoise_wav_to_path(input_path: str) -> str:
    y, sr = librosa.load(input_path, sr=16000)
    noise_sample = y[0:int(0.5 * sr)] if len(y) > int(0.5 * sr) else y
    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    out_path = input_path.replace(".wav", "_denoised.wav")
    sf.write(out_path, y_denoised, sr)
    return out_path


def transcribe_and_pacing(path: str) -> Tuple[str, List[Dict[str, float]], List[float]]:
    """Transcribe with Whisper and compute per-segment WPM bins and pacing curve."""
    global model
    if model is None:
        import whisper
        model = whisper.load_model("base")
    result = model.transcribe(path, word_timestamps=True)
    transcription = result.get("text", "").strip()
    segments = result.get("segments", [])

    time_bins: List[float] = []
    wpm_bins: List[float] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        duration = max(1e-9, end - start)
        words = str(seg.get("text", "")).strip().split()
        if duration > 0:
            wpm = len(words) / (duration / 60.0)
            time_bins.append((start + end) / 2.0)
            wpm_bins.append(float(wpm))

    pacing_curve = [{"time": round(t, 2), "wpm": round(w, 2)} for t, w in zip(time_bins, wpm_bins)]
    return transcription, pacing_curve, wpm_bins


def extract_mfcc(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def classify_rate(path: str) -> Tuple[str, float]:
    feats = extract_mfcc(path).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    pred_idx = int(clf.predict(feats_scaled)[0])
    label = ["Fast", "Ideal", "Slow"][pred_idx]
    probs = clf.predict_proba(feats_scaled)[0]
    conf = float(probs[pred_idx])
    return label, conf



