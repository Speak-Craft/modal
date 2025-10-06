from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
from joblib import load
import uvicorn
import tempfile
import os
import noisereduce as nr
import soundfile as sf 
from rate_activity_endpoints import router as rate_activity_router


app = FastAPI()

# CORS for the React frontend pornt no 8000 is connected 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount activity router for rate-related activities
app.include_router(rate_activity_router)

# Load models (leave for /rate-analysis/ path)
model = None
clf = load("./mlp_classifier.pkl")
scaler = load("./scaler.pkl")

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def generate_enhanced_feedback(wpm, consistency_score, pacing_curve, duration, word_count, model_prediction=None):
    """Generate comprehensive feedback based on speech analysis"""
    
    # Use model prediction if available, otherwise fall back to WPM-based categorization
    if model_prediction:
        pace_category = model_prediction
    else:
        # Base pace feedback - Using same 3 categories as the model
        if wpm < 100:
            pace_category = "Slow"
        elif wpm <= 150:
            pace_category = "Ideal"
        else:  # wpm > 150
            pace_category = "Fast"
    
    # Generate feedback based on the determined category
    if pace_category == "Slow":
        pace_emoji = "🟡"
        pace_feedback = "Your pace is below the ideal range but manageable with practice."
        pace_suggestions = [
            "Aim to increase your pace by 15-20 WPM",
            "Practice with timing exercises using a stopwatch",
            "Focus on reducing unnecessary pauses between words"
        ]
    elif pace_category == "Ideal":
        pace_emoji = "🟢"
        pace_feedback = "Excellent! You're speaking at an optimal pace for audience engagement."
        pace_suggestions = [
            "Maintain this pace consistently throughout your presentation",
            "Use strategic pauses for emphasis rather than slowing down",
            "Practice varying your pace slightly for dynamic delivery"
        ]
    else:  # Fast
        pace_emoji = "🟠"
        pace_feedback = "Your pace is above optimal and may affect audience comprehension."
        pace_suggestions = [
            "Practice breathing exercises to control your pace",
            "Use punctuation marks as natural pause indicators",
            "Record yourself and identify sections that need slowing down"
        ]
    
    # Consistency analysis
    if len(pacing_curve) >= 3:
        wpm_values = [point['wpm'] for point in pacing_curve]
        wpm_std = np.std(wpm_values)
        wpm_range = max(wpm_values) - min(wpm_values)
        
        if wpm_std <= 15:
            consistency_category = "Excellent"
            consistency_emoji = "🏆"
            consistency_feedback = "Your pacing consistency is outstanding!"
            consistency_suggestions = [
                "Maintain this level of consistency",
                "Use your natural rhythm to enhance delivery"
            ]
        elif wpm_std <= 25:
            consistency_category = "Good"
            consistency_emoji = "✅"
            consistency_feedback = "Good pacing consistency with room for improvement."
            consistency_suggestions = [
                "Practice maintaining steady rhythm",
                "Use breathing techniques for consistent pacing"
            ]
        elif wpm_std <= 40:
            consistency_category = "Fair"
            consistency_emoji = "⚠️"
            consistency_feedback = "Moderate pacing consistency that can be improved."
            consistency_suggestions = [
                "Practice with a metronome to develop steady rhythm",
                "Identify sections where pace varies significantly"
            ]
        else:
            consistency_category = "Needs Improvement"
            consistency_emoji = "🔄"
            consistency_feedback = "Significant pacing variations detected."
            consistency_suggestions = [
                "Focus on maintaining consistent breathing patterns",
                "Practice reading with consistent timing",
                "Use pacing markers throughout your speech"
            ]
    else:
        consistency_category = "Insufficient Data"
        consistency_emoji = "📊"
        consistency_feedback = "Not enough speech data for consistency analysis."
        consistency_suggestions = [
            "Record longer speech samples for better analysis",
            "Aim for at least 30 seconds of continuous speech"
        ]
    
    # Speech flow analysis
    if duration > 0:
        words_per_second = word_count / duration
        if words_per_second < 1.5:
            flow_category = "Paused"
            flow_emoji = "⏸️"
            flow_feedback = "Your speech has many pauses, which can be strategic or distracting."
        elif words_per_second < 2.5:
            flow_category = "Natural"
            flow_emoji = "🌊"
            flow_feedback = "Good natural flow with appropriate pauses."
        else:
            flow_category = "Rapid"
            flow_emoji = "⚡"
            flow_feedback = "Very rapid speech flow that may need pacing control."
    else:
        flow_category = "Unknown"
        flow_emoji = "❓"
        flow_feedback = "Unable to analyze speech flow."
    
    # Generate actionable recommendations
    priority_recommendations = []
    
    if wpm < 100:
        priority_recommendations.append({
            "priority": "High",
            "area": "Pace Increase",
            "action": f"Gradually increase from {wpm:.2f} WPM to 120-150 WPM",
            "exercise": "Practice with metronome at 120 BPM"
        })
    elif wpm > 150:
        priority_recommendations.append({
            "priority": "High",
            "area": "Pace Control",
            "action": f"Reduce from {wpm:.2f} WPM to 120-150 WPM",
            "exercise": "Practice breathing exercises and slow reading"
        })
    
    if consistency_score < 70:
        priority_recommendations.append({
            "priority": "Medium",
            "area": "Consistency",
            "action": "Improve pacing consistency",
            "exercise": "Use timing markers and practice steady rhythm"
        })
    
    if not priority_recommendations:
        priority_recommendations.append({
            "priority": "Maintenance",
            "area": "Current Performance",
            "action": "Maintain excellent speech patterns",
            "exercise": "Continue current practice routine"
        })
    
    return {
        "pace_analysis": {
            "category": pace_category,
            "emoji": pace_emoji,
            "feedback": pace_feedback,
            "suggestions": pace_suggestions
        },
        "consistency_analysis": {
            "category": consistency_category,
            "emoji": consistency_emoji,
            "feedback": consistency_feedback,
            "suggestions": consistency_suggestions,
            "score": consistency_score,
            "std_dev": wpm_std if len(pacing_curve) >= 3 else None,
            "range": wpm_range if len(pacing_curve) >= 3 else None
        },
        "flow_analysis": {
            "category": flow_category,
            "emoji": flow_emoji,
            "feedback": flow_feedback,
            "words_per_second": words_per_second if duration > 0 else None
        },
        "priority_recommendations": priority_recommendations,
        "overall_assessment": f"{pace_emoji} {pace_category} Pace • {consistency_emoji} {consistency_category} Consistency • {flow_emoji} {flow_category} Flow"
    }

@app.post("/generate-rate-feedback/")
async def generate_rate_feedback(payload: dict = Body(...)):
    """Generate rate feedback from frontend-provided metrics.

    Expects JSON body with keys: wpm, label, consistencyScore, pacingCurve, duration, wordCount.
    Returns an object under key `enhancedFeedback` that mirrors the structure used by /rate-analysis/.
    """
    try:
        wpm = float(payload.get("wpm", 0) or 0)
        label = payload.get("label") or None
        consistency_score = float(payload.get("consistencyScore", 0) or 0)
        pacing_curve = payload.get("pacingCurve") or []
        duration = float(payload.get("duration", 0) or 0)
        word_count = int(payload.get("wordCount", 0) or 0)

        feedback = generate_enhanced_feedback(
            wpm=wpm,
            consistency_score=consistency_score,
            pacing_curve=pacing_curve,
            duration=duration,
            word_count=word_count,
            model_prediction=label,
        )

        return {"enhancedFeedback": feedback}
    except Exception as exc:
        return {"error": str(exc)}

@app.post("/rate-analysis/")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        global model
        if model is None:
            import whisper
            model = whisper.load_model("base")
        # Save uploaded audio to a temporary .wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load and denoise audio
        y, sr = librosa.load(tmp_path, sr=16000)
        noise_sample = y[0:int(0.5 * sr)]
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

        # Save denoised audio to a new path using soundfile
        denoised_path = tmp_path.replace(".wav", "_denoised.wav")
        sf.write(denoised_path, y_denoised, sr)

        # Transcribe with Whisper
        result = model.transcribe(denoised_path, word_timestamps=True)
        transcription = result["text"].strip()
        segments = result["segments"]

        # Build pacing curve
        time_bins = []
        wpm_bins = []
        for seg in segments:
            start, end = seg["start"], seg["end"]
            duration = end - start
            words = seg["text"].strip().split()
            if duration > 0:
                wpm = len(words) / (duration / 60)
                time_bins.append((start + end) / 2)
                wpm_bins.append(wpm)

        pacing_curve = [{"time": round(t, 2), "wpm": round(w, 2)} for t, w in zip(time_bins, wpm_bins)]

        # Stats
        duration = librosa.get_duration(y=y_denoised, sr=sr)
        word_count = len(transcription.split())
        wpm = word_count / (duration / 60) if duration > 0 else 0

        # MFCC + prediction
        features = extract_mfcc(denoised_path).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred = clf.predict(features_scaled)[0]
        pred_label = ["Fast", "Ideal", "Slow"][pred]

        # Confidence (optional use)
        probs = clf.predict_proba(features_scaled)[0]
        pred_prob = round(probs[pred] * 100, 2)

        # Enhanced feedback generation using model prediction for consistency
        enhanced_feedback = generate_enhanced_feedback(wpm, 0, pacing_curve, duration, word_count, pred_label)
        
        # Calculate consistency score for enhanced feedback
        if len(wpm_bins) >= 3:
            wpm_std = np.std(wpm_bins)
            enhanced_feedback["consistency_analysis"]["score"] = max(0, round(100 - wpm_std, 2))
            enhanced_feedback["consistency_analysis"]["std_dev"] = round(wpm_std, 2)
            enhanced_feedback["consistency_analysis"]["range"] = round(max(wpm_bins) - min(wpm_bins), 2)

        # Ideal bounds
        ideal_lines = [{"time": round(t, 2), "upper": 150, "lower": 100} for t in time_bins]

        os.remove(tmp_path)
        os.remove(denoised_path)

        return {
            "wordCount": word_count,
            "duration": round(duration, 2),
            "wpm": round(wpm, 2),
            "prediction": pred_label,
            "consistencyScore": enhanced_feedback["consistency_analysis"]["score"],
            "feedback": enhanced_feedback["overall_assessment"],
            "pacingCurve": pacing_curve,
            "idealLines": ideal_lines,
            "enhancedFeedback": enhanced_feedback
        }

    except Exception as e:
        return {"error": str(e)}
    
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
