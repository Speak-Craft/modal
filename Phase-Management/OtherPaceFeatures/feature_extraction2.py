# feature_extraction2.py - Enhanced Feature Extraction for Speech Pace Management
import os, json, joblib, numpy as np, pandas as pd
import librosa, whisper
import parselmouth
from parselmouth.praat import call
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

AUDIO_DIR = "../WAVs"
MODEL_NAME = "medium"  # whisper
CHUNK_SIZE = 30.0      # seconds for distribution bins

# ---------- Industry-Standard Pause Values (Toastmasters/Presentations) ----------
MICRO_PAUSE = 0.1      # Breathing, emphasis (0.1-0.3s)
SHORT_PAUSE = 0.3      # Natural flow, comma pauses (0.3-1.0s)
MED_PAUSE = 1.0        # Sentence breaks, transitions (1.0-2.5s)
LONG_PAUSE = 2.5       # Paragraph breaks, audience engagement (2.5-5.0s)
EXCESSIVE_PAUSE = 5.0  # Problematic, needs improvement (>5.0s)

# Toastmasters Standards
OPTIMAL_PAUSE_RATIO_MIN = 0.08   # 8% minimum
OPTIMAL_PAUSE_RATIO_MAX = 0.12   # 12% maximum
TRANSITION_PAUSE_MIN = 1.5        # 1.5s minimum for transitions
TRANSITION_PAUSE_MAX = 2.0        # 2.0s maximum for transitions
EMPHASIS_PAUSE_MIN = 0.5          # 0.5s minimum for emphasis
EMPHASIS_PAUSE_MAX = 1.0          # 1.0s maximum for emphasis
AUDIENCE_PAUSE_MIN = 2.0          # 2.0s minimum for audience processing
AUDIENCE_PAUSE_MAX = 3.0          # 3.0s maximum for audience processing

# ---------- Feature Lists ----------
CORE_FEATURES = [
    "short_count","med_count","long_count","excessive_count",
    "short_time","med_time","long_time","excessive_time","total_pause_time","pause_ratio",
    "short_rate_pm","med_rate_pm","long_rate_pm","excessive_rate_pm","all_rate_pm",
    "pause_p50","pause_p90","pause_p95","pause_std","pause_min","pause_max",
    "max_long_streak","seg_duration_std","rhythm_outliers",
    "wpm_mean","wpm_std","wpm_delta_mean","wpm_delta_std",
    "wpm_cv","wpm_range","gap_clustering","wpm_acceleration","wpm_jerk",
    "pause_pattern_regularity","speech_continuity","pause_spacing_consistency",
    "speaking_efficiency","pause_efficiency","rhythm_regularity"
]

NOVEL_PAUSE_FEATURES = [
    "contextual_pause_score","transition_pause_count","emphasis_pause_count",
    "optimal_transition_ratio","optimal_emphasis_ratio","pause_rhythm_consistency",
    "golden_ratio_pauses","pause_entropy","pause_autocorrelation",
    "cognitive_pause_score","memory_retrieval_pauses","confidence_score",
    "optimal_cognitive_pause_ratio","pause_fractal_dimension","pause_spectral_density",
    "pause_trend_analysis","pause_volatility","toastmasters_compliance_score"
]

ADVANCED_FEATURES = [
    "spectral_centroid","spectral_rolloff","spectral_bandwidth","zero_crossing_rate","tempo",
    "pitch_mean","pitch_std","pitch_min","pitch_max",
    "jitter_local","jitter_rap","jitter_ppq5",
    "shimmer_local","shimmer_apq3","shimmer_apq5","hnr_mean",
    "f1_mean","f2_mean","f3_mean"
]

MFCC_FEATURES = [f"mfcc_{i}_mean" for i in range(13)] + [f"mfcc_{i}_std" for i in range(13)]

# ---------- Transcription ----------
def transcribe_with_words(path: str):
    model = whisper.load_model(MODEL_NAME)
    res = model.transcribe(path, word_timestamps=True, verbose=False)
    segs = res.get("segments", [])
    # Normalize: ensure each segment has word list (can be absent rarely)
    for s in segs:
        if "words" not in s or s["words"] is None:
            # Approximate words by splitting text evenly across segment
            txt = s.get("text","").strip()
            if not txt:
                s["words"] = []
            else:
                words = [w for w in txt.split() if w.strip()]
                if words:
                    dur = s["end"] - s["start"]
                    step = dur / len(words)
                    s["words"] = [{"word": w, "start": s["start"] + i*step, "end": s["start"] + (i+1)*step}
                                  for i, w in enumerate(words)]
                else:
                    s["words"] = []
    return segs, res.get("text","")

# ---------- Advanced Audio Feature Extraction ----------
def extract_advanced_features(path: str) -> Dict[str, float]:
    """Extract advanced audio features using Parselmouth and Librosa"""
    try:
        # Load audio with librosa
        y, sr = librosa.load(path, sr=16000)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Parselmouth features (if available)
        try:
            import parselmouth
            from parselmouth.praat import call
            
            sound = parselmouth.Sound(path)
            
            # Pitch features
            pitch = sound.to_pitch()
            pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            pitch_std_dev = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            pitch_min = call(pitch, "Get minimum", 0, 0, "Hertz")
            pitch_max = call(pitch, "Get maximum", 0, 0, "Hertz")
            
            # Voice Quality Features
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Harmonicity
            hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_mean = call(hnr, "Get mean", 0, 0)
            
            # Formants
            formant = sound.to_formant_burg()
            f1_mean = call(formant, "Get mean", 1, 0, 0, "hertz")
            f2_mean = call(formant, "Get mean", 2, 0, 0, "hertz")
            f3_mean = call(formant, "Get mean", 3, 0, 0, "hertz")
            
        except ImportError:
            # Fallback values when parselmouth is not available
            pitch_mean = 150.0
            pitch_std_dev = 20.0
            pitch_min = 80.0
            pitch_max = 300.0
            jitter_local = 0.01
            jitter_rap = 0.01
            jitter_ppq5 = 0.01
            shimmer_local = 0.05
            shimmer_apq3 = 0.05
            shimmer_apq5 = 0.05
            hnr_mean = 15.0
            f1_mean = 500.0
            f2_mean = 1500.0
            f3_mean = 2500.0
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        features = {}
        
        # Add MFCC features
        for i in range(13):
            features[f"mfcc_{i}_mean"] = float(mfcc_mean[i])
            features[f"mfcc_{i}_std"] = float(mfcc_std[i])
        
        # Add spectral features
        features.update({
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_bandwidth": float(spectral_bandwidth),
            "zero_crossing_rate": float(zcr),
            "tempo": float(tempo)
        })
        
        # Add pitch features
        features.update({
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std_dev),
            "pitch_min": float(pitch_min),
            "pitch_max": float(pitch_max)
        })
        

        
        # Add voice quality features
        features.update({
            "jitter_local": float(jitter_local),
            "jitter_rap": float(jitter_rap),
            "jitter_ppq5": float(jitter_ppq5),
            "shimmer_local": float(shimmer_local),
            "shimmer_apq3": float(shimmer_apq3),
            "shimmer_apq5": float(shimmer_apq5),
            "hnr_mean": float(hnr_mean)
        })
        
        # Add formant features
        features.update({
            "f1_mean": float(f1_mean),
            "f2_mean": float(f2_mean),
            "f3_mean": float(f3_mean)
        })
        
        return features
        
    except Exception as e:
        print(f"Error extracting advanced features: {e}")
        # Return default values if extraction fails
        default_features = {}
        for i in range(13):
            default_features[f"mfcc_{i}_mean"] = 0.0
            default_features[f"mfcc_{i}_std"] = 0.0
        
        default_features.update({
            "spectral_centroid": 0.0, "spectral_rolloff": 0.0, "spectral_bandwidth": 0.0,
            "zero_crossing_rate": 0.0, "tempo": 0.0, "pitch_mean": 0.0, "pitch_std": 0.0,
            "pitch_min": 0.0, "pitch_max": 0.0, "jitter_local": 0.0, "jitter_rap": 0.0,
            "jitter_ppq5": 0.0, "shimmer_local": 0.0, "shimmer_apq3": 0.0, "shimmer_apq5": 0.0,
            "hnr_mean": 0.0, "f1_mean": 0.0, "f2_mean": 0.0, "f3_mean": 0.0
        })
        return default_features

# ---------- Enhanced Pause extraction ----------
def extract_word_gaps(segments: List[Dict[str,Any]]) -> List[Tuple[float,float,float]]:
    """
    Returns list of (gap_start, gap_end, gap_dur) across the entire file
    computed between end of previous word and start of next word.
    Falls back to segment boundaries when needed.
    """
    words = []
    for s in segments:
        for w in s.get("words", []):
            if w and isinstance(w.get("start"), (int,float)) and isinstance(w.get("end"), (int,float)):
                words.append((w["start"], w["end"]))
    words.sort(key=lambda x: x[0])

    gaps = []
    if words:
        prev_end = words[0][1]
        for (ws, we) in words[1:]:
            gap = ws - prev_end
            if gap >= MICRO_PAUSE:
                gaps.append((prev_end, ws, gap))
            prev_end = max(prev_end, we)
    else:
        # fallback to segments
        prev_end = None
        for s in segments:
            if prev_end is not None:
                g = s["start"] - prev_end
                if g >= MICRO_PAUSE:
                    gaps.append((prev_end, s["start"], g))
            prev_end = s["end"]
    return gaps

def bin_long_pause_distribution(gaps: List[Tuple[float,float,float]], total_dur: float, chunk_size: float = CHUNK_SIZE):
    n = int(np.ceil(total_dur / chunk_size))
    dist = np.zeros(n, dtype=float)
    for (gs, ge, gd) in gaps:
        if gd > MED_PAUSE:  # long only
            idx = int(gs // chunk_size)
            if 0 <= idx < n:
                dist[idx] += gd
    return dist.tolist()

# ---------- Enhanced Feature engineering ----------
def pause_features_for_file(path: str) -> Dict[str, Any]:
    y, sr = librosa.load(path, sr=16000)
    total_dur = librosa.get_duration(y=y, sr=sr)

    segments, _ = transcribe_with_words(path)
    gaps = extract_word_gaps(segments)
    
    # Extract advanced features
    advanced_features = extract_advanced_features(path)

    short = [g for (_,_,g) in gaps if MICRO_PAUSE <= g <= SHORT_PAUSE]
    med   = [g for (_,_,g) in gaps if SHORT_PAUSE < g <= MED_PAUSE]
    longg = [g for (_,_,g) in gaps if MED_PAUSE < g <= LONG_PAUSE]
    excessive = [g for (_,_,g) in gaps if g > EXCESSIVE_PAUSE]

    # speaking segments for pace/rhythm (using whisper segments)
    wpm_bins, seg_durs = [], []
    for s in segments:
        dur = max(1e-9, s["end"] - s["start"])
        word_count = sum(1 for w in s.get("words", []) if w.get("word","").strip())
        wpm_bins.append(60.0 * word_count / dur)
        seg_durs.append(dur)

    # deltas
    wpm_bins = np.array(wpm_bins) if len(wpm_bins) else np.array([0.0])
    wpm_delta = np.diff(wpm_bins) if len(wpm_bins) > 1 else np.array([0.0])
    seg_durs = np.array(seg_durs) if len(seg_durs) else np.array([0.0])

    short_time, med_time, long_time, excessive_time = sum(short), sum(med), sum(longg), sum(excessive)
    total_pause_time = short_time + med_time + long_time + excessive_time
    pause_ratio = (total_pause_time / total_dur) if total_dur > 0 else 0.0

    # rates per minute
    dur_min = max(1e-9, total_dur / 60.0)
    short_rate_pm = len(short) / dur_min
    med_rate_pm   = len(med)   / dur_min
    long_rate_pm  = len(longg) / dur_min
    excessive_rate_pm = len(excessive) / dur_min
    all_rate_pm   = (len(short) + len(med) + len(longg) + len(excessive)) / dur_min

    # Enhanced statistics
    def safe_stats(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        a = np.array(arr)
        return (float(np.median(a)), float(np.percentile(a,90)), float(np.percentile(a,95)), 
                float(np.std(a)), float(np.min(a)), float(np.max(a)))

    p50, p90, p95, pause_std, pause_min, pause_max = safe_stats([g for (_,_,g) in gaps])

    # run-length of long pauses (including excessive)
    long_streak = 0
    max_long_streak = 0
    for (_,_,g) in gaps:
        if g > MED_PAUSE:  # This includes both long and excessive pauses
            long_streak += 1
            max_long_streak = max(max_long_streak, long_streak)
        else:
            long_streak = 0

    # Enhanced rhythm analysis
    seg_std = float(np.std(seg_durs)) if len(seg_durs) > 1 else 0.0
    rhythm_outliers = int(np.sum(np.abs(seg_durs - seg_durs.mean()) > 1.5 * (seg_std if seg_std>0 else 1)))
    
    # Pace consistency metrics
    wpm_mean = float(wpm_bins.mean())
    wpm_std  = float(wpm_bins.std())
    wpm_delta_mean = float(wpm_delta.mean()) if wpm_delta.size > 0 else 0.0
    wpm_delta_std  = float(wpm_delta.std())  if wpm_delta.size > 0 else 0.0
    
    # Additional pace management features
    wpm_cv = (wpm_std / wpm_mean) if wpm_mean > 0 else 0.0  # Coefficient of variation
    wpm_range = float(wpm_bins.max() - wpm_bins.min()) if len(wpm_bins) > 0 else 0.0
    
    # Pause clustering analysis
    if len(gaps) > 1:
        gap_intervals = [gaps[i+1][0] - gaps[i][1] for i in range(len(gaps)-1)]
        gap_clustering = np.std(gap_intervals) if gap_intervals else 0.0
    else:
        gap_clustering = 0.0
    
    # Advanced pace stability metrics
    wpm_acceleration = np.mean(np.diff(wpm_delta)) if len(wpm_delta) > 1 else 0.0
    wpm_jerk = np.mean(np.abs(np.diff(wpm_delta))) if len(wpm_delta) > 1 else 0.0
    
    # Pause pattern analysis
    pause_pattern_regularity = 1.0 / (1.0 + gap_clustering) if gap_clustering > 0 else 1.0
    
    # Speech flow metrics
    speech_continuity = 1.0 - (len(gaps) / max(1, len(segments)))
    pause_spacing_consistency = 1.0 / (1.0 + np.std([g[2] for g in gaps])) if len(gaps) > 1 else 1.0
    
    # Speaking efficiency metrics
    speaking_efficiency = (total_dur - total_pause_time) / total_dur if total_dur > 0 else 0.0
    pause_efficiency = total_pause_time / len(gaps) if len(gaps) > 0 else 0.0
    
    # Rhythm regularity
    rhythm_regularity = 1.0 / (1.0 + seg_std) if seg_std > 0 else 1.0
    
    # ---------- NOVEL PAUSE ANALYSIS FEATURES ----------
    
    # 1. 🎭 Contextual Pause Analysis
    def analyze_pause_context(gaps, segments):
        """Analyze pauses based on surrounding content context"""
        contextual_scores = []
        transition_pauses = []
        emphasis_pauses = []
        
        for gap_start, gap_end, gap_dur in gaps:
            # Find segments before and after pause
            prev_seg = None
            next_seg = None
            
            for seg in segments:
                if abs(seg["end"] - gap_start) < 0.1:  # Pause starts after this segment
                    prev_seg = seg
                if abs(seg["start"] - gap_end) < 0.1:  # Pause ends before this segment
                    next_seg = seg
                    break
            
            # Contextual scoring
            score = 0.0
            
            # Transition pause detection (between different topics)
            if prev_seg and next_seg:
                prev_text = prev_seg.get("text", "").lower()
                next_text = next_seg.get("text", "").lower()
                
                # Check for topic transitions (simple heuristics)
                transition_indicators = ["however", "but", "on the other hand", "meanwhile", 
                                      "furthermore", "additionally", "in conclusion", "finally"]
                
                is_transition = any(indicator in prev_text or indicator in next_text 
                                   for indicator in transition_indicators)
                
                if is_transition and TRANSITION_PAUSE_MIN <= gap_dur <= TRANSITION_PAUSE_MAX:
                    score += 2.0  # Good transition pause
                    transition_pauses.append(gap_dur)
                elif gap_dur > TRANSITION_PAUSE_MAX:
                    score += 1.0  # Transition pause too long
                else:
                    score += 0.5  # Regular pause
            
            # Emphasis pause detection
            if EMPHASIS_PAUSE_MIN <= gap_dur <= EMPHASIS_PAUSE_MAX:
                score += 1.5  # Good emphasis pause
                emphasis_pauses.append(gap_dur)
            
            # Audience processing pause
            if AUDIENCE_PAUSE_MIN <= gap_dur <= AUDIENCE_PAUSE_MAX:
                score += 1.0  # Good audience pause
            
            # Excessive pause penalty
            if gap_dur > EXCESSIVE_PAUSE:
                score -= 2.0  # Heavy penalty for excessive pauses
            
            contextual_scores.append(score)
        
        return {
            "contextual_pause_score": np.mean(contextual_scores) if contextual_scores else 0.0,
            "transition_pause_count": len(transition_pauses),
            "emphasis_pause_count": len(emphasis_pauses),
            "optimal_transition_ratio": len([g for g in transition_pauses if TRANSITION_PAUSE_MIN <= g <= TRANSITION_PAUSE_MAX]) / max(1, len(transition_pauses)),
            "optimal_emphasis_ratio": len([g for g in emphasis_pauses if EMPHASIS_PAUSE_MIN <= g <= EMPHASIS_PAUSE_MAX]) / max(1, len(emphasis_pauses))
        }
    
    # 2. 🎵 Rhythm Pattern Analysis
    def analyze_pause_rhythm(gaps, total_dur):
        """Analyze pause rhythm patterns and consistency"""
        if len(gaps) < 2:
            return {
                "pause_rhythm_consistency": 1.0,
                "golden_ratio_pauses": 0.0,
                "pause_entropy": 0.0,
                "pause_autocorrelation": 0.0
            }
        
        # Pause intervals (time between pauses)
        pause_intervals = [gaps[i+1][0] - gaps[i][1] for i in range(len(gaps)-1)]
        
        # Rhythm consistency (regularity of pause spacing)
        rhythm_consistency = 1.0 / (1.0 + np.std(pause_intervals)) if pause_intervals else 1.0
        
        # Golden ratio pauses (pauses that follow natural speech rhythms)
        # Golden ratio ≈ 1.618, we look for pauses that are roughly 1.618x longer than average
        avg_pause_dur = np.mean([g[2] for g in gaps])
        golden_ratio_target = avg_pause_dur * 1.618
        golden_ratio_pauses = sum(1 for g in gaps if 0.8 * golden_ratio_target <= g[2] <= 1.2 * golden_ratio_target)
        golden_ratio_ratio = golden_ratio_pauses / len(gaps)
        
        # Pause entropy (measure of randomness vs. intentionality)
        pause_durations = [g[2] for g in gaps]
        if len(set(pause_durations)) > 1:
            # Calculate entropy using histogram
            hist, _ = np.histogram(pause_durations, bins=min(10, len(pause_durations)))
            hist = hist[hist > 0]  # Remove zero bins
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            max_entropy = np.log2(len(prob))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 0.0
        
        # Pause autocorrelation (detect repeating patterns)
        if len(pause_durations) > 3:
            # Simple autocorrelation at lag 1
            autocorr = np.corrcoef(pause_durations[:-1], pause_durations[1:])[0, 1]
            autocorr = 0.0 if np.isnan(autocorr) else abs(autocorr)
        else:
            autocorr = 0.0
        
        return {
            "pause_rhythm_consistency": float(rhythm_consistency),
            "golden_ratio_pauses": float(golden_ratio_ratio),
            "pause_entropy": float(normalized_entropy),
            "pause_autocorrelation": float(autocorr)
        }
    
    # 3. 🧠 Cognitive Load Indicators
    def analyze_cognitive_load(gaps, segments):
        """Analyze pauses that suggest cognitive load or thinking"""
        cognitive_pauses = []
        memory_retrieval_pauses = []
        confidence_indicators = []
        
        for gap_start, gap_end, gap_dur in gaps:
            # Find surrounding segments for context
            prev_seg = None
            next_seg = None
            
            for seg in segments:
                if abs(seg["end"] - gap_start) < 0.1:
                    prev_seg = seg
                if abs(seg["start"] - gap_end) < 0.1:
                    next_seg = seg
                    break
            
            # Cognitive load indicators
            if prev_seg and next_seg:
                prev_text = prev_seg.get("text", "").lower()
                next_text = next_seg.get("text", "").lower()
                
                # Memory retrieval pauses (longer pauses after complex sentences)
                complex_indicators = ["because", "therefore", "consequently", "as a result", 
                                   "in addition", "furthermore", "moreover"]
                
                is_complex = any(indicator in prev_text for indicator in complex_indicators)
                if is_complex and gap_dur > 1.5:
                    memory_retrieval_pauses.append(gap_dur)
                
                # Confidence indicators (hesitation vs. intentional pauses)
                hesitation_indicators = ["um", "uh", "er", "ah", "like", "you know"]
                has_hesitation = any(indicator in prev_text or indicator in next_text 
                                   for indicator in hesitation_indicators)
                
                if has_hesitation:
                    confidence_indicators.append(1.0 if gap_dur < 1.0 else 0.5)
                else:
                    confidence_indicators.append(1.0)
                
                # Cognitive load based on pause duration
                if 1.0 <= gap_dur <= 3.0:  # Optimal thinking pause
                    cognitive_pauses.append(1.0)
                elif gap_dur > 3.0:  # Too long, might indicate struggle
                    cognitive_pauses.append(0.5)
                else:  # Too short for complex thinking
                    cognitive_pauses.append(0.0)
        
        return {
            "cognitive_pause_score": np.mean(cognitive_pauses) if cognitive_pauses else 0.0,
            "memory_retrieval_pauses": len(memory_retrieval_pauses),
            "confidence_score": np.mean(confidence_indicators) if confidence_indicators else 1.0,
            "optimal_cognitive_pause_ratio": len([g for g in cognitive_pauses if g >= 0.8]) / max(1, len(cognitive_pauses))
        }
    
    # 4. 📊 Advanced Statistical Features
    def calculate_advanced_statistics(gaps, total_dur):
        """Calculate advanced statistical features for pause analysis"""
        if len(gaps) < 2:
            return {
                "pause_fractal_dimension": 0.0,
                "pause_spectral_density": 0.0,
                "pause_trend_analysis": 0.0,
                "pause_volatility": 0.0
            }
        
        pause_durations = [g[2] for g in gaps]
        pause_timings = [g[0] for g in gaps]  # When pauses occur
        
        # Pause volatility (rate of change in pause durations)
        if len(pause_durations) > 1:
            volatility = np.std(np.diff(pause_durations))
        else:
            volatility = 0.0
        
        # Pause trend analysis (are pauses getting longer/shorter over time?)
        if len(pause_durations) > 2:
            # Linear trend
            x = np.arange(len(pause_durations))
            coeffs = np.polyfit(x, pause_durations, 1)
            trend_slope = coeffs[0]
            trend_strength = abs(trend_slope) / (np.std(pause_durations) + 1e-6)
        else:
            trend_strength = 0.0
        
        # Pause spectral density (frequency domain analysis)
        if len(pause_durations) > 4:
            # Simple FFT-based spectral density
            fft_vals = np.fft.fft(pause_durations)
            spectral_density = np.mean(np.abs(fft_vals[1:len(fft_vals)//2]))
        else:
            spectral_density = 0.0
        
        # Fractal dimension approximation (self-similarity across time scales)
        if len(pause_durations) > 8:
            # Box-counting dimension approximation
            scales = [1, 2, 4, 8]
            counts = []
            
            for scale in scales:
                if len(pause_durations) >= scale:
                    # Resample at different scales
                    resampled = [np.mean(pause_durations[i:i+scale]) 
                               for i in range(0, len(pause_durations), scale)]
                    counts.append(len(set(np.round(np.array(resampled), 1))))
                else:
                    counts.append(1)
            
            # Calculate fractal dimension
            if len(counts) > 1 and counts[0] > 0:
                log_counts = np.log(counts)
                log_scales = np.log(scales[:len(counts)])
                if len(log_counts) > 1:
                    fractal_dim = -np.polyfit(log_scales, log_counts, 1)[0]
                else:
                    fractal_dim = 0.0
            else:
                fractal_dim = 0.0
        else:
            fractal_dim = 0.0
        
        return {
            "pause_fractal_dimension": float(fractal_dim),
            "pause_spectral_density": float(spectral_density),
            "pause_trend_analysis": float(trend_strength),
            "pause_volatility": float(volatility)
        }
    
    # Apply novel analysis functions
    contextual_features = analyze_pause_context(gaps, segments)
    rhythm_features = analyze_pause_rhythm(gaps, total_dur)
    cognitive_features = analyze_cognitive_load(gaps, segments)
    advanced_stats = calculate_advanced_statistics(gaps, total_dur)
    
    # Toastmasters compliance scoring
    toastmasters_compliance = {
        "pause_ratio_compliance": 1.0 if OPTIMAL_PAUSE_RATIO_MIN <= pause_ratio <= OPTIMAL_PAUSE_RATIO_MAX else 0.0,
        "transition_pause_compliance": contextual_features["optimal_transition_ratio"],
        "emphasis_pause_compliance": contextual_features["optimal_emphasis_ratio"],
        "overall_compliance_score": 0.0
    }
    
    # Calculate overall compliance score
    compliance_scores = [
        toastmasters_compliance["pause_ratio_compliance"],
        toastmasters_compliance["transition_pause_compliance"],
        toastmasters_compliance["emphasis_pause_compliance"]
    ]
    toastmasters_compliance["overall_compliance_score"] = np.mean(compliance_scores)
    
    long_dist = bin_long_pause_distribution(gaps, total_dur, CHUNK_SIZE)

    # Combine all features
    feats = {
        "total_duration": total_dur,
        "short_count": len(short), "med_count": len(med), "long_count": len(longg), "excessive_count": len(excessive),
        "short_time": short_time, "med_time": med_time, "long_time": long_time, "excessive_time": excessive_time,
        "total_pause_time": total_pause_time, "pause_ratio": pause_ratio,
        "short_rate_pm": short_rate_pm, "med_rate_pm": med_rate_pm,
        "long_rate_pm": long_rate_pm, "excessive_rate_pm": excessive_rate_pm, "all_rate_pm": all_rate_pm,
        "pause_p50": p50, "pause_p90": p90, "pause_p95": p95, "pause_std": pause_std,
        "pause_min": pause_min, "pause_max": pause_max,
        "max_long_streak": max_long_streak,
        "seg_duration_std": seg_std, "rhythm_outliers": rhythm_outliers,
        "wpm_mean": wpm_mean, "wpm_std": wpm_std,
        "wpm_delta_mean": wpm_delta_mean, "wpm_delta_std": wpm_delta_std,
        "wpm_cv": wpm_cv, "wpm_range": wpm_range,
        "gap_clustering": gap_clustering,
        "wpm_acceleration": wpm_acceleration, "wpm_jerk": wpm_jerk,
        "pause_pattern_regularity": pause_pattern_regularity,
        "speech_continuity": speech_continuity, "pause_spacing_consistency": pause_spacing_consistency,
        "speaking_efficiency": speaking_efficiency,
        "pause_efficiency": pause_efficiency,
        "rhythm_regularity": rhythm_regularity,
        "long_pause_distribution": long_dist,
        "contextual_pause_score": contextual_features["contextual_pause_score"],
        "transition_pause_count": contextual_features["transition_pause_count"],
        "emphasis_pause_count": contextual_features["emphasis_pause_count"],
        "optimal_transition_ratio": contextual_features["optimal_transition_ratio"],
        "optimal_emphasis_ratio": contextual_features["optimal_emphasis_ratio"],
        "pause_rhythm_consistency": rhythm_features["pause_rhythm_consistency"],
        "golden_ratio_pauses": rhythm_features["golden_ratio_pauses"],
        "pause_entropy": rhythm_features["pause_entropy"],
        "pause_autocorrelation": rhythm_features["pause_autocorrelation"],
        "cognitive_pause_score": cognitive_features["cognitive_pause_score"],
        "memory_retrieval_pauses": cognitive_features["memory_retrieval_pauses"],
        "confidence_score": cognitive_features["confidence_score"],
        "optimal_cognitive_pause_ratio": cognitive_features["optimal_cognitive_pause_ratio"],
        "pause_fractal_dimension": advanced_stats["pause_fractal_dimension"],
        "pause_spectral_density": advanced_stats["pause_spectral_density"],
        "pause_trend_analysis": advanced_stats["pause_trend_analysis"],
        "pause_volatility": advanced_stats["pause_volatility"],
        "toastmasters_compliance_score": toastmasters_compliance["overall_compliance_score"]
    }
    
    # Add advanced features
    feats.update(advanced_features)
    
    return feats

# ---------- Enhanced labeling system ----------
def auto_label(row: pd.Series) -> str:
    """Enhanced labeling system using Toastmasters standards and novel pause analysis"""
    
    # Toastmasters compliance scoring
    toastmasters_score = row.get("toastmasters_compliance_score", 0.0)
    
    # Pause management quality using industry standards
    if (row["pause_ratio"] > 0.15 or row.get("excessive_count", 0) > 2 or 
        row.get("contextual_pause_score", 0.0) < -0.5):
        return "poor_pause_control"
    elif (row["pause_ratio"] > 0.12 or row.get("excessive_count", 0) > 1 or 
          row.get("contextual_pause_score", 0.0) < 0.0):
        return "needs_pause_improvement"
    
    # Pace consistency and stability
    if (row["wpm_cv"] > 0.4 or row["wpm_delta_std"] > 8 or 
        row["seg_duration_std"] > 2.5 or row["wpm_jerk"] > 10):
        return "inconsistent_pace"
    elif (row["wpm_cv"] > 0.25 or row["wpm_delta_std"] > 5 or 
          row["wpm_jerk"] > 5):
        return "moderate_pace_consistency"
    
    # Rhythm and flow quality using novel features
    rhythm_score = row.get("pause_rhythm_consistency", 1.0)
    golden_ratio_score = row.get("golden_ratio_pauses", 0.0)
    
    if (row["rhythm_outliers"] >= 6 or rhythm_score < 0.3 or
        row["speech_continuity"] < 0.6 or golden_ratio_score < 0.2):
        return "poor_rhythm"
    elif (row["rhythm_outliers"] >= 3 or rhythm_score < 0.5 or
          row["speech_continuity"] < 0.8 or golden_ratio_score < 0.4):
        return "moderate_rhythm"
    
    # Cognitive load and confidence analysis
    cognitive_score = row.get("cognitive_pause_score", 0.0)
    confidence_score = row.get("confidence_score", 1.0)
    
    if (cognitive_score < 0.3 or confidence_score < 0.6 or
        row.get("optimal_cognitive_pause_ratio", 1.0) < 0.5):
        return "cognitive_load_issues"
    
    # Overall speaking quality and efficiency
    if (row["speaking_efficiency"] < 0.7 or row["pause_efficiency"] > 3.0 or
        row["pause_spacing_consistency"] < 0.5):
        return "needs_overall_improvement"
    
    # Toastmasters excellence
    if (toastmasters_score >= 0.8 and row["pause_ratio"] >= 0.08 and 
        row["pause_ratio"] <= 0.12 and row.get("optimal_transition_ratio", 0.0) >= 0.7):
        return "toastmasters_excellent"
    
    return "good_pace_control"

# ---------- Dataset build ----------
def build_dataset(audio_dir=AUDIO_DIR) -> pd.DataFrame:
    rows = []
    for f in os.listdir(audio_dir):
        if not f.lower().endswith(".wav"):
            continue
        path = os.path.join(audio_dir, f)
        try:
            feats = pause_features_for_file(path)
            feats["filename"] = f
            rows.append(feats)
            print(f"OK: {f}")
        except Exception as e:
            print(f"FAIL {f}: {e}")
    df = pd.DataFrame(rows)

    # explode distribution into columns
    max_bins = max((len(x) for x in df["long_pause_distribution"]), default=0)
    for i in range(max_bins):
        df[f"long_dist_{i}"] = df["long_pause_distribution"].apply(lambda v: v[i] if i < len(v) else 0.0)
    df.drop(columns=["long_pause_distribution"], inplace=True)

    # auto labels
    df["label"] = df.apply(auto_label, axis=1)
    return df

if __name__ == "__main__":
    print("🚀 Enhanced Speech Pace Management - Feature Extraction")
    print("=" * 60)
    
    # Build dataset
    print("📊 Building enhanced dataset...")
    df = build_dataset(AUDIO_DIR)
    df.to_csv("enhanced_pause_features.csv", index=False)
    print(f"✅ Dataset: {len(df)} rows saved to enhanced_pause_features.csv")
    
    print("\n🎉 Feature extraction complete! Ready for model training.")
    print("\n📁 Next steps:")
    print("  1. Run model_training.py to train the model")
    print("  2. Use the trained model for real-time analysis")
    print("  3. Generate personalized feedback and suggestions")

# ---------- Enhanced Inference + Suggestions ----------
@dataclass
class EnhancedPrediction:
    label: str
    probs: Dict[str,float]
    suggestions: List[str]
    features: Dict[str,float]
    confidence: float
    improvement_areas: List[str]

def suggest_from_features(f: Dict[str, Any]) -> List[str]:
    """Enhanced suggestion system using Toastmasters standards and novel pause analysis"""
    tips = []
    
    # Toastmasters compliance suggestions
    toastmasters_score = f.get("toastmasters_compliance_score", 0.0)
    if toastmasters_score < 0.6:
        tips.append("🏆 TOASTMASTERS: Focus on industry-standard pause management (8-12% pause ratio).")
    elif toastmasters_score < 0.8:
        tips.append("🏆 TOASTMASTERS: Good progress! Aim for 8-12% pause ratio with optimal transition timing.")
    
    # Pause management suggestions using industry standards
    if f["pause_ratio"] > 0.15:
        tips.append("🚨 CRITICAL: Reduce total pause time significantly. Target pause ratio 8-12% (Toastmasters standard).")
    elif f["pause_ratio"] > 0.12:
        tips.append("⚠️  Reduce pause time. Aim for 8-12% pause ratio for natural flow (Toastmasters standard).")
    elif f["pause_ratio"] < 0.08:
        tips.append("💡 Consider adding more strategic pauses. Optimal range is 8-12% for audience engagement.")
    
    if f.get("excessive_count", 0) > 2:
        tips.append("🚨 CRITICAL: Too many excessive pauses (>5s). Eliminate all pauses longer than 5 seconds.")
    elif f.get("excessive_count", 0) > 0:
        tips.append("⚠️  Reduce excessive pauses. No pause should exceed 5 seconds.")
    
    if f["long_count"] > 6:
        tips.append("⚠️  Too many long pauses (2.5-5s). Limit to <4 long pauses per presentation.")
    elif f["long_count"] > 4:
        tips.append("💡 Consider reducing long pauses. Use transitions or signposting to avoid gaps.")
    
    # Contextual pause analysis suggestions
    contextual_score = f.get("contextual_pause_score", 0.0)
    if contextual_score < 0.0:
        tips.append("🎭 CONTEXT: Improve pause context. Use 1.5-2.0s for transitions, 0.5-1.0s for emphasis.")
    
    transition_ratio = f.get("optimal_transition_ratio", 1.0)
    if transition_ratio < 0.6:
        tips.append("🔄 TRANSITIONS: Improve transition pauses. Target 1.5-2.0s between topics for clarity.")
    
    emphasis_ratio = f.get("optimal_emphasis_ratio", 1.0)
    if emphasis_ratio < 0.6:
        tips.append("💪 EMPHASIS: Use 0.5-1.0s pauses for key points to enhance audience retention.")
    
    # Rhythm pattern suggestions
    rhythm_consistency = f.get("pause_rhythm_consistency", 1.0)
    if rhythm_consistency < 0.4:
        tips.append("🎵 RHYTHM: Work on consistent pause timing. Practice with metronome for rhythm consistency.")
    
    golden_ratio = f.get("golden_ratio_pauses", 0.0)
    if golden_ratio < 0.3:
        tips.append("✨ GOLDEN RATIO: Incorporate natural speech rhythms. Some pauses should be 1.618x longer than average.")
    
    # Cognitive load suggestions
    cognitive_score = f.get("cognitive_pause_score", 0.0)
    if cognitive_score < 0.4:
        tips.append("🧠 COGNITIVE: Optimize thinking pauses. Use 1.0-3.0s pauses for complex concepts.")
    
    confidence_score = f.get("confidence_score", 1.0)
    if confidence_score < 0.7:
        tips.append("💪 CONFIDENCE: Reduce hesitation. Practice content to minimize thinking pauses during delivery.")
    
    # Advanced statistical suggestions
    pause_entropy = f.get("pause_entropy", 0.0)
    if pause_entropy > 0.8:
        tips.append("📊 PATTERN: Pauses are too random. Develop more intentional, structured pause patterns.")
    
    pause_volatility = f.get("pause_volatility", 0.0)
    if pause_volatility > 2.0:
        tips.append("📈 STABILITY: Pause durations vary too much. Aim for consistent pause lengths within categories.")
    
    # Pace consistency suggestions
    if f["wpm_cv"] > 0.4:
        tips.append("🚨 CRITICAL: Very inconsistent pace. Practice with metronome at steady WPM.")
    elif f["wpm_cv"] > 0.25:
        tips.append("⚠️  Pace varies too much. Aim for ±15% WPM variation.")
    
    if f["wpm_delta_std"] > 8:
        tips.append("🚨 CRITICAL: Sudden pace changes. Practice smooth transitions between sections.")
    elif f["wpm_delta_std"] > 5:
        tips.append("⚠️  Pace changes too abruptly. Smooth out transitions.")
    
    # Advanced pace stability
    if f.get("wpm_jerk", 0) > 10:
        tips.append("🚨 CRITICAL: Very jerky pace changes. Practice gradual speed adjustments.")
    elif f.get("wpm_jerk", 0) > 5:
        tips.append("⚠️  Pace changes are jerky. Smooth out acceleration and deceleration.")
    
    # Rhythm and flow suggestions
    if f["rhythm_outliers"] >= 6:
        tips.append("🚨 CRITICAL: Inconsistent rhythm. Practice consistent segment lengths.")
    elif f["rhythm_outliers"] >= 3:
        tips.append("⚠️  Rhythm needs work. Aim for consistent speech segment durations.")
    
    if f.get("speech_continuity", 1.0) < 0.6:
        tips.append("🚨 CRITICAL: Speech is too fragmented. Reduce unnecessary pauses between thoughts.")
    elif f.get("speech_continuity", 1.0) < 0.8:
        tips.append("⚠️  Speech flow could be smoother. Connect related ideas more seamlessly.")
    
    # Speaking efficiency
    if f["speaking_efficiency"] < 0.7:
        tips.append("⚠️  Low speaking efficiency. Reduce unnecessary pauses and filler words.")
    
    if f["pause_efficiency"] > 3.0:
        tips.append("⚠️  Pauses are too long. Target 0.5-2.0 second pauses for natural flow.")
    
    # Advanced feature suggestions
    if f.get("jitter_local", 0) > 0.02:
        tips.append("💡 Voice stability: Reduce vocal jitter for clearer speech.")
    
    if f.get("shimmer_local", 0) > 0.1:
        tips.append("💡 Voice clarity: Work on reducing vocal shimmer for better articulation.")
    
    if f.get("pitch_std", 0) > 50:
        tips.append("💡 Pitch control: Maintain more consistent pitch throughout presentation.")
    
    if not tips:
        tips.append("✅ Excellent pace control! Maintain consistent phrasing and intentional 0.5-1.0s pauses.")
    
    return tips

# ---------- ENHANCED SUGGESTION GENERATION SYSTEM ----------
def generate_actionable_suggestions(features: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive, actionable suggestions for speech improvement"""
    
    suggestions = {
        "critical_issues": [],
        "major_improvements": [],
        "minor_refinements": [],
        "practice_exercises": [],
        "immediate_actions": [],
        "long_term_goals": [],
        "toastmasters_tips": [],
        "confidence_builders": []
    }
    
    # 🚨 CRITICAL ISSUES (Immediate attention required)
    if features.get("pause_ratio", 0) > 0.15:
        suggestions["critical_issues"].append({
            "issue": "Excessive pause time",
            "current": f"{features['pause_ratio']:.1%}",
            "target": "8-12%",
            "action": "Reduce unnecessary pauses by 30-40%",
            "impact": "High - affects audience engagement and professional image"
        })
    
    if features.get("wpm_cv", 0) > 0.4:
        suggestions["critical_issues"].append({
            "issue": "Very inconsistent pace",
            "current": f"{features.get('wpm_cv', 0):.2f}",
            "target": "<0.25",
            "action": "Practice with metronome at steady WPM for 15 minutes daily",
            "impact": "High - makes speech hard to follow"
        })
    
    if features.get("excessive_count", 0) > 2:
        suggestions["critical_issues"].append({
            "issue": "Too many excessive pauses",
            "current": f"{features.get('excessive_count', 0)}",
            "target": "0",
            "action": "Eliminate all pauses longer than 5 seconds",
            "impact": "High - creates awkward silences and loses audience attention"
        })
    
    if features.get("long_count", 0) > 6:
        suggestions["critical_issues"].append({
            "issue": "Too many long pauses",
            "current": f"{features.get('long_count', 0)}",
            "target": "<4",
            "action": "Use transition phrases and signposting to bridge gaps",
            "impact": "Medium - affects speech flow"
        })
    
    # ⚠️ MAJOR IMPROVEMENTS (Significant impact)
    if features.get("contextual_pause_score", 0) < 0.0:
        suggestions["major_improvements"].append({
            "issue": "Poor pause context",
            "current": f"{features.get('contextual_pause_score', 0):.2f}",
            "target": ">0.5",
            "action": "Study pause timing in professional speeches (TED Talks, Toastmasters)",
            "impact": "Medium - affects speech effectiveness"
        })
    
    if features.get("pause_rhythm_consistency", 1.0) < 0.5:
        suggestions["major_improvements"].append({
            "issue": "Inconsistent pause rhythm",
            "current": f"{features.get('pause_rhythm_consistency', 1.0):.2f}",
            "target": ">0.7",
            "action": "Practice with rhythmic patterns: short-short-long, short-long-short",
            "impact": "Medium - affects speech flow"
        })
    
    # 💡 MINOR REFINEMENTS (Polish and enhancement)
    if features.get("golden_ratio_pauses", 0.0) < 0.3:
        suggestions["minor_refinements"].append({
            "issue": "Missing natural speech rhythms",
            "current": f"{features.get('golden_ratio_pauses', 0.0):.1%}",
            "target": ">30%",
            "action": "Incorporate 1.618x longer pauses at key moments",
            "impact": "Low - enhances naturalness"
        })
    
    if features.get("pause_entropy", 0.0) > 0.8:
        suggestions["minor_refinements"].append({
            "issue": "Too random pause patterns",
            "current": f"{features.get('pause_entropy', 0.0):.2f}",
            "target": "<0.6",
            "action": "Create intentional pause patterns: emphasis, transition, breathing",
            "impact": "Low - improves predictability"
        })
    
    # 🏋️ PRACTICE EXERCISES (Specific training)
    if features.get("wpm_jerk", 0) > 5:
        suggestions["practice_exercises"].append({
            "exercise": "Smooth Pace Transitions",
            "duration": "10 minutes daily",
            "method": "Read text starting at 120 WPM, gradually increase to 150 WPM over 30 seconds, then decrease back",
            "goal": "Eliminate sudden speed changes",
            "frequency": "Daily for 2 weeks"
        })
    
    if features.get("rhythm_outliers", 0) >= 3:
        suggestions["practice_exercises"].append({
            "exercise": "Consistent Segment Timing",
            "duration": "15 minutes daily",
            "method": "Practice speaking in 10-second segments with consistent timing",
            "goal": "Reduce timing variations",
            "frequency": "Daily for 3 weeks"
        })
    
    # ⚡ IMMEDIATE ACTIONS (Can do today)
    suggestions["immediate_actions"].extend([
        {
            "action": "Record yourself speaking for 2 minutes",
            "focus": "Pause timing and rhythm",
            "time_required": "5 minutes",
            "expected_outcome": "Baseline measurement"
        },
        {
            "action": "Practice 3 transition phrases",
            "focus": "Smooth topic changes",
            "time_required": "10 minutes",
            "expected_outcome": "Better flow between ideas"
        }
    ])
    
    # 🎯 LONG-TERM GOALS (3-6 months)
    suggestions["long_term_goals"].extend([
        {
            "goal": "Achieve Toastmasters compliance",
            "target_score": ">0.8",
            "timeline": "3 months",
            "milestones": ["Week 4: 0.6", "Week 8: 0.7", "Week 12: 0.8"]
        },
        {
            "goal": "Master pause patterns",
            "target_consistency": ">0.8",
            "timeline": "6 months",
            "milestones": ["Month 2: Basic patterns", "Month 4: Advanced rhythms", "Month 6: Mastery"]
        }
    ])
    
    # 🏆 TOASTMASTERS SPECIFIC TIPS
    toastmasters_score = features.get("toastmasters_compliance_score", 0.0)
    if toastmasters_score < 0.6:
        suggestions["toastmasters_tips"].extend([
            "Join a local Toastmasters club for structured practice",
            "Use the Competent Communicator manual for systematic improvement",
            "Practice Table Topics to improve spontaneous speaking",
            "Record and analyze your speeches monthly"
        ])
    elif toastmasters_score < 0.8:
        suggestions["toastmasters_tips"].extend([
            "Work on Advanced Communication Series projects",
            "Focus on vocal variety and body language",
            "Practice with different speech types (informative, persuasive, entertaining)",
            "Seek feedback from experienced Toastmasters"
        ])
    
    # 💪 CONFIDENCE BUILDERS
    if features.get("confidence_score", 1.0) < 0.8:
        suggestions["confidence_builders"].extend([
            "Practice in front of a mirror to build comfort",
            "Start with familiar topics to reduce anxiety",
            "Use power poses before speaking",
            "Focus on breathing exercises to calm nerves"
        ])
    
    return suggestions

def generate_personalized_feedback(features: Dict[str, Any]) -> str:
    """Generate personalized, actionable feedback based on analysis results"""
    
    suggestions = generate_actionable_suggestions(features)
    
    feedback = "🎯 PERSONALIZED SPEECH IMPROVEMENT PLAN\n"
    feedback += "=" * 50 + "\n\n"
    
    # Overall Assessment
    toastmasters_score = features.get("toastmasters_compliance_score", 0.0)
    if toastmasters_score >= 0.8:
        feedback += "🏆 EXCELLENT! You're meeting Toastmasters standards.\n"
    elif toastmasters_score >= 0.6:
        feedback += "👍 GOOD! You're on the right track with room for improvement.\n"
    else:
        feedback += "📈 NEEDS WORK! Focus on fundamental improvements first.\n"
    
    feedback += f"Current Toastmasters Score: {toastmasters_score:.1%}\n\n"
    
    # Critical Issues
    if suggestions["critical_issues"]:
        feedback += "🚨 CRITICAL ISSUES (Address Immediately):\n"
        for issue in suggestions["critical_issues"]:
            feedback += f"• {issue['issue']}: {issue['current']} → {issue['target']}\n"
            feedback += f"  Action: {issue['action']}\n"
            feedback += f"  Impact: {issue['impact']}\n\n"
    
    # Major Improvements
    if suggestions["major_improvements"]:
        feedback += "⚠️ MAJOR IMPROVEMENTS (Focus This Week):\n"
        for improvement in suggestions["major_improvements"]:
            feedback += f"• {improvement['issue']}: {improvement['current']} → {improvement['target']}\n"
            feedback += f"  Action: {improvement['action']}\n"
            feedback += f"  Impact: {improvement['impact']}\n\n"
    
    # Practice Exercises
    if suggestions["practice_exercises"]:
        feedback += "🏋️ PRACTICE EXERCISES:\n"
        for exercise in suggestions["practice_exercises"]:
            feedback += f"• {exercise['exercise']}\n"
            feedback += f"  Duration: {exercise['duration']}\n"
            feedback += f"  Method: {exercise['method']}\n"
            feedback += f"  Goal: {exercise['goal']}\n"
            feedback += f"  Frequency: {exercise['frequency']}\n\n"
    
    # Immediate Actions
    if suggestions["immediate_actions"]:
        feedback += "⚡ IMMEDIATE ACTIONS (Do Today):\n"
        for action in suggestions["immediate_actions"]:
            feedback += f"• {action['action']}\n"
            feedback += f"  Focus: {action['focus']}\n"
            feedback += f"  Time: {action['time_required']}\n"
            feedback += f"  Outcome: {action['expected_outcome']}\n\n"
    
    # Long-term Goals
    if suggestions["long_term_goals"]:
        feedback += "🎯 LONG-TERM GOALS (3-6 Months):\n"
        for goal in suggestions["long_term_goals"]:
            feedback += f"• {goal['goal']}\n"
            feedback += f"  Target: {goal['target_score']}\n"
            feedback += f"  Timeline: {goal['timeline']}\n"
            feedback += f"  Milestones: {' → '.join(goal['milestones'])}\n\n"
    
    # Toastmasters Tips
    if suggestions["toastmasters_tips"]:
        feedback += "🏆 TOASTMASTERS TIPS:\n"
        for tip in suggestions["toastmasters_tips"]:
            feedback += f"• {tip}\n"
        feedback += "\n"
    
    # Confidence Builders
    if suggestions["confidence_builders"]:
        feedback += "💪 CONFIDENCE BUILDERS:\n"
        for builder in suggestions["confidence_builders"]:
            feedback += f"• {builder}\n"
        feedback += "\n"
    
    # Progress Tracking
    feedback += "📊 PROGRESS TRACKING:\n"
    feedback += "• Record your speech weekly and compare metrics\n"
    feedback += "• Focus on 1-2 improvements at a time\n"
    feedback += "• Celebrate small wins to maintain motivation\n"
    feedback += "• Reassess every 2 weeks to track improvement\n\n"
    
    feedback += "🎉 Remember: Every great speaker started somewhere. Focus on progress, not perfection!"
    
    return feedback

def identify_improvement_areas(f: Dict[str, Any]) -> List[str]:
    """Identify specific areas that need improvement"""
    areas = []
    
    if f["pause_ratio"] > 0.10:
        areas.append("Pause Management")
    if f["wpm_cv"] > 0.25:
        areas.append("Pace Consistency")
    if f.get("wpm_jerk", 0) > 5:
        areas.append("Pace Stability")
    if f["rhythm_outliers"] >= 3:
        areas.append("Rhythm Control")
    if f.get("speech_continuity", 1.0) < 0.8:
        areas.append("Speech Flow")
    if f.get("pause_pattern_regularity", 1.0) < 0.6:
        areas.append("Pause Patterns")
    if f["speaking_efficiency"] < 0.8:
        areas.append("Speaking Efficiency")
    if f.get("jitter_local", 0) > 0.02:
        areas.append("Voice Stability")
    if f.get("pitch_std", 0) > 50:
        areas.append("Pitch Control")
    
    return areas if areas else ["All areas performing well"]

def predict_file(path: str, model_path="enhanced_pause_model.joblib", cfg_path="enhanced_pause_features.json") -> EnhancedPrediction:
    """Enhanced prediction with confidence and improvement areas"""
    try:
        pipe = joblib.load(model_path)
    except:
        print("Model not found. Please train the model first.")
        return None
        
    with open(cfg_path, "r") as f:
        meta = json.load(f)
    X_cols = meta["feature_order"]

    feats = pause_features_for_file(path)
    # add dist columns
    n_bins = len(feats["long_pause_distribution"])
    for i in range(n_bins):
        feats[f"long_dist_{i}"] = feats["long_pause_distribution"][i]
    # ensure every expected feature exists
    for c in X_cols:
        if c not in feats:
            feats[c] = 0.0

    x = np.array([feats[c] for c in X_cols]).reshape(1, -1)
    proba = pipe.predict_proba(x)[0]
    label = pipe.classes_[np.argmax(proba)]
    confidence = float(np.max(proba))
    probs = {cls: float(p) for cls, p in zip(pipe.classes_, proba)}
    
    tips = suggest_from_features(feats)
    improvement_areas = identify_improvement_areas(feats)
    
    return EnhancedPrediction(
        label=label, 
        probs=probs, 
        suggestions=tips, 
        features=feats,
        confidence=confidence,
        improvement_areas=improvement_areas
    )

# ---------- Real-time feedback system ----------
def generate_real_time_feedback(prediction: EnhancedPrediction) -> str:
    """Generate real-time feedback for presenters using enhanced suggestion system"""
    
    # Generate comprehensive suggestions
    suggestions = generate_actionable_suggestions(prediction.features)
    
    feedback = f"🎯 SPEECH PACE ANALYSIS - REAL-TIME FEEDBACK\n"
    feedback += f"Overall Assessment: {prediction.label.replace('_', ' ').title()}\n"
    feedback += f"Confidence: {prediction.confidence:.1%}\n\n"
    
    # Toastmasters Compliance
    toastmasters_score = prediction.features.get("toastmasters_compliance_score", 0.0)
    feedback += f"🏆 TOASTMASTERS COMPLIANCE: {toastmasters_score:.1%}\n"
    if toastmasters_score >= 0.8:
        feedback += "Status: EXCELLENT - Meeting industry standards! 🎉\n"
    elif toastmasters_score >= 0.6:
        feedback += "Status: GOOD - On track with room for improvement 👍\n"
    else:
        feedback += "Status: NEEDS WORK - Focus on fundamentals 📈\n"
    feedback += "\n"
    
    # Key Metrics Summary
    feedback += "📊 KEY METRICS:\n"
    f = prediction.features
    
    # Core metrics with targets
    feedback += f"• Pause Ratio: {f['pause_ratio']:.1%} (Target: 8-12%) "
    if 0.08 <= f['pause_ratio'] <= 0.12:
        feedback += "✅"
    elif f['pause_ratio'] > 0.15:
        feedback += "🚨"
    else:
        feedback += "⚠️"
    feedback += "\n"
    
    feedback += f"• Excessive Pauses: {f.get('excessive_count', 0)} (Target: 0) "
    if f.get('excessive_count', 0) == 0:
        feedback += "✅"
    elif f.get('excessive_count', 0) > 2:
        feedback += "🚨"
    else:
        feedback += "⚠️"
    feedback += "\n"
    
    feedback += f"• Long Pauses: {f['long_count']} (Target: <4) "
    if f['long_count'] < 4:
        feedback += "✅"
    elif f['long_count'] > 6:
        feedback += "🚨"
    else:
        feedback += "⚠️"
    feedback += "\n"
    
    feedback += f"• Pace Consistency: {1-f['wpm_cv']:.1%} (Target: >75%) "
    if 1-f['wpm_cv'] > 0.75:
        feedback += "✅"
    elif 1-f['wpm_cv'] < 0.6:
        feedback += "🚨"
    else:
        feedback += "⚠️"
    feedback += "\n"
    
    feedback += f"• Speech Flow: {f.get('speech_continuity', 1.0):.1%} (Target: >80%) "
    if f.get('speech_continuity', 1.0) > 0.8:
        feedback += "✅"
    elif f.get('speech_continuity', 1.0) < 0.6:
        feedback += "🚨"
    else:
        feedback += "⚠️"
    feedback += "\n"
    
    # Novel features
    if 'contextual_pause_score' in f:
        feedback += f"• Contextual Pause Score: {f['contextual_pause_score']:.2f} (Target: >0.5) "
        if f['contextual_pause_score'] > 0.5:
            feedback += "✅"
        elif f['contextual_pause_score'] < 0.0:
            feedback += "🚨"
        else:
            feedback += "⚠️"
        feedback += "\n"
    
    if 'pause_rhythm_consistency' in f:
        feedback += f"• Rhythm Consistency: {f['pause_rhythm_consistency']:.1%} (Target: >70%) "
        if f['pause_rhythm_consistency'] > 0.7:
            feedback += "✅"
        elif f['pause_rhythm_consistency'] < 0.4:
            feedback += "🚨"
        else:
            feedback += "⚠️"
        feedback += "\n"
    
    feedback += "\n"
    
    # Priority-based suggestions
    if suggestions["critical_issues"]:
        feedback += "🚨 IMMEDIATE ACTIONS NEEDED:\n"
        for issue in suggestions["critical_issues"][:2]:  # Show top 2 critical issues
            feedback += f"• {issue['issue']}: {issue['action']}\n"
        feedback += "\n"
    
    if suggestions["major_improvements"]:
        feedback += "⚠️ THIS WEEK'S FOCUS:\n"
        for improvement in suggestions["major_improvements"][:2]:  # Show top 2 improvements
            feedback += f"• {improvement['issue']}: {improvement['action']}\n"
        feedback += "\n"
    
    if suggestions["immediate_actions"]:
        feedback += "⚡ DO TODAY:\n"
        for action in suggestions["immediate_actions"][:2]:  # Show top 2 immediate actions
            feedback += f"• {action['action']} ({action['time_required']})\n"
        feedback += "\n"
    
    # Quick wins
    feedback += "💡 QUICK WINS:\n"
    if f.get('contextual_pause_score', 0) < 0.0:
        feedback += "• Practice 1.5-2.0s pauses between topics\n"
    if f.get('emphasis_pause_count', 0) < 3:
        feedback += "• Add 0.5-1.0s pauses for key points\n"
    if f.get('pause_rhythm_consistency', 1.0) < 0.6:
        feedback += "• Use consistent timing patterns\n"
    
    feedback += "\n"
    feedback += "📱 For detailed improvement plan, check the full analysis report.\n"
    feedback += "🔄 Re-record in 1 week to track progress!"
    
    return feedback

def generate_comprehensive_report(features: Dict[str, Any]) -> str:
    """Generate a comprehensive report with all suggestions and analysis"""
    
    # Get personalized feedback
    personalized_feedback = generate_personalized_feedback(features)
    
    # Add technical analysis
    report = personalized_feedback + "\n\n"
    report += "🔬 TECHNICAL ANALYSIS DETAILS\n"
    report += "=" * 50 + "\n\n"
    
    # Advanced metrics
    report += "📊 ADVANCED METRICS:\n"
    report += f"• WPM Coefficient of Variation: {features.get('wpm_cv', 0):.3f}\n"
    report += f"• WPM Delta Standard Deviation: {features.get('wpm_delta_std', 0):.2f}\n"
    report += f"• WPM Jerk (Acceleration Changes): {features.get('wpm_jerk', 0):.2f}\n"
    report += f"• Rhythm Outliers: {features.get('rhythm_outliers', 0)}\n"
    report += f"• Speaking Efficiency: {features.get('speaking_efficiency', 0):.1%}\n"
    report += f"• Pause Efficiency: {features.get('pause_efficiency', 0):.2f}s\n\n"
    
    # Novel features analysis
    if 'contextual_pause_score' in features:
        report += "🎭 CONTEXTUAL ANALYSIS:\n"
        report += f"• Contextual Pause Score: {features['contextual_pause_score']:.3f}\n"
        report += f"• Transition Pauses: {features.get('transition_pause_count', 0)}\n"
        report += f"• Emphasis Pauses: {features.get('emphasis_pause_count', 0)}\n"
        report += f"• Optimal Transition Ratio: {features.get('optimal_transition_ratio', 0):.1%}\n"
        report += f"• Optimal Emphasis Ratio: {features.get('optimal_emphasis_ratio', 0):.1%}\n\n"
    
    if 'pause_rhythm_consistency' in features:
        report += "🎵 RHYTHM ANALYSIS:\n"
        report += f"• Rhythm Consistency: {features['pause_rhythm_consistency']:.3f}\n"
        report += f"• Golden Ratio Pauses: {features.get('golden_ratio_pauses', 0):.1%}\n"
        report += f"• Pause Entropy: {features.get('pause_entropy', 0):.3f}\n"
        report += f"• Pause Autocorrelation: {features.get('pause_autocorrelation', 0):.3f}\n\n"
    
    if 'cognitive_pause_score' in features:
        report += "🧠 COGNITIVE ANALYSIS:\n"
        report += f"• Cognitive Pause Score: {features['cognitive_pause_score']:.3f}\n"
        report += f"• Memory Retrieval Pauses: {features.get('memory_retrieval_pauses', 0)}\n"
        report += f"• Confidence Score: {features.get('confidence_score', 0):.3f}\n"
        report += f"• Optimal Cognitive Pause Ratio: {features.get('optimal_cognitive_pause_ratio', 0):.1%}\n\n"
    
    if 'pause_fractal_dimension' in features:
        report += "📊 STATISTICAL ANALYSIS:\n"
        report += f"• Fractal Dimension: {features.get('pause_fractal_dimension', 0):.3f}\n"
        report += f"• Spectral Density: {features.get('pause_spectral_density', 0):.3f}\n"
        report += f"• Trend Analysis: {features.get('pause_trend_analysis', 0):.3f}\n"
        report += f"• Pause Volatility: {features.get('pause_volatility', 0):.3f}\n\n"
    
    # Voice quality metrics
    report += "🎤 VOICE QUALITY METRICS:\n"
    report += f"• Pitch Standard Deviation: {features.get('pitch_std', 0):.1f} Hz\n"
    report += f"• Jitter (Local): {features.get('jitter_local', 0):.4f}\n"
    report += f"• Shimmer (Local): {features.get('shimmer_local', 0):.3f}\n"
    report += f"• Harmonic-to-Noise Ratio: {features.get('hnr_mean', 0):.2f}\n\n"
    
    # Recommendations summary
    report += "🎯 RECOMMENDATIONS SUMMARY:\n"
    report += "=" * 50 + "\n"
    
    suggestions = generate_actionable_suggestions(features)
    
    if suggestions["critical_issues"]:
        report += f"🚨 Critical Issues: {len(suggestions['critical_issues'])} (Address immediately)\n"
    if suggestions["major_improvements"]:
        report += f"⚠️ Major Improvements: {len(suggestions['major_improvements'])} (Focus this week)\n"
    if suggestions["minor_refinements"]:
        report += f"💡 Minor Refinements: {len(suggestions['minor_refinements'])} (Polish and enhance)\n"
    if suggestions["practice_exercises"]:
        report += f"🏋️ Practice Exercises: {len(suggestions['practice_exercises'])} (Daily training)\n"
    
    report += "\n"
    report += "📈 EXPECTED IMPROVEMENT TIMELINE:\n"
    report += "• Week 1-2: Address critical issues\n"
    report += "• Week 3-4: Implement major improvements\n"
    report += "• Month 2-3: Focus on refinements\n"
    report += "• Month 4-6: Master advanced techniques\n\n"
    
    report += "🎉 Keep practicing and recording to track your progress!"
    
    return report
