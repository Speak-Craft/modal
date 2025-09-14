import librosa
import numpy as np

def extract_features(y, sr):
    """Extracts audio features (MFCC, chroma, spectral, etc.) for filler detection."""
    # Skip clips shorter than 0.3s or silent
    if len(y) < int(0.3 * sr) or np.all(y == 0):
        print(f"⚠️ Audio too short ({len(y)/sr:.2f}s) or silent")
        return None

    try:
        print(f"🔧 Extracting features for {len(y)/sr:.2f}s audio chunk")
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Extract spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        print(f"📊 Feature shapes - MFCC: {mfcc.shape}, Delta: {delta.shape}, Chroma: {chroma.shape}")
        print(f"📊 Spectral shapes - Centroid: {spec_centroid.shape}, ZCR: {zcr.shape}, RMS: {rms.shape}")

        # Validate features
        if any(f.shape[1] == 0 for f in [mfcc, delta, chroma]) or spec_centroid.shape[1] == 0:
            print("❌ One or more features are empty")
            return None

        # Combine all features into a single vector
        feature_vector = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            [np.mean(spec_centroid), np.std(spec_centroid)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)]
        ])
        
        print(f"✅ Feature vector created: {len(feature_vector)} features")
        print(f"📈 Feature vector stats - min: {np.min(feature_vector):.3f}, max: {np.max(feature_vector):.3f}, mean: {np.mean(feature_vector):.3f}")
        
        return feature_vector

    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
