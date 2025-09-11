import librosa
import numpy as np

def extract_features(y, sr):
    """Extracts audio features (MFCC, chroma, spectral, etc.) for filler detection."""
    # Skip clips shorter than 0.3s or silent
    if len(y) < int(0.3 * sr) or np.all(y == 0):
        raise ValueError("Audio too short or silent")

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        # Validate features
        if any(f.shape[1] == 0 for f in [mfcc, delta, chroma]) or spec_centroid.shape[1] == 0:
            raise ValueError("One or more features are empty")

        # Combine all features into a single vector
        feature_vector = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            [np.mean(spec_centroid), np.std(spec_centroid)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)]
        ])
        return feature_vector

    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}")
