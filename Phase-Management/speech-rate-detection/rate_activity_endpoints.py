from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import io
import os
import tempfile
import librosa
import soundfile as sf
import numpy as np
from typing import Dict, Optional
import time
import base64

from rate_core import denoise_wav_to_path, transcribe_and_pacing, classify_rate, extract_mfcc, classify_rate

router = APIRouter()


@router.post("/real-time-analysis/")
async def real_time_analysis_rate(request: Dict):
    """Real-time analysis for rate activities with audio chunk processing.
    
    Accepts either:
    1. Duration-based mock (legacy): {"duration": 30, "activityType": "rate_match"}
    2. Audio chunk analysis: {"audioChunk": "base64_data", "activityType": "rate_match", "chunkIndex": 0}
    """
    try:
        # Check if this is a real audio chunk or just duration-based request
        if "audioChunk" in request:
            return await process_realtime_audio_chunk(request)
        else:
            # Legacy duration-based mock
            duration = float(request.get("duration", 0))
            activity_type = request.get("activityType", "rate_match")
            
            # Simple heuristic: progress by 2-min target
            target_seconds = 120.0
            progress = max(0.0, min(100.0, (duration / target_seconds) * 100.0))
            
            # Generate activity-specific mock feedback
            mock_feedback = generate_mock_feedback(activity_type, duration)
            
            return JSONResponse({
                "success": True,
                "score": mock_feedback["score"],
                "targetScore": 100,
                "progress": progress,
                "feedback": mock_feedback["feedback"],
                "metrics": mock_feedback["metrics"],
                "is_mock": True
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rate real-time analysis error: {str(e)}")


async def process_realtime_audio_chunk(request: Dict):
    """Process a 3-second audio chunk for real-time rate analysis."""
    try:
        import base64
        
        # Extract audio data
        audio_data = request.get("audioChunk")
        activity_type = request.get("activityType", "rate_match")
        chunk_index = request.get("chunkIndex", 0)
        session_id = request.get("sessionId", "default")
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Save the audio chunk temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            chunk_path = tmp.name
        
        try:
            # Process the 3-second chunk
            y, sr = librosa.load(chunk_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Skip if chunk is too short (less than 1 second)
            if duration < 1.0:
                return JSONResponse({
                    "success": False,
                    "message": "Audio chunk too short",
                    "chunk_index": chunk_index,
                    "session_id": session_id,
                    "is_mock": False
                })
            
            # Transcribe and analyze using rate_core functions
            denoised_path = denoise_wav_to_path(chunk_path)
            transcription, pacing_curve, wpm_bins = transcribe_and_pacing(denoised_path)
            
            # Calculate current WPM
            word_count = len(transcription.split())
            current_wpm = word_count / (duration / 60.0) if duration > 0 else 0
            
            # Classify rate using ML model
            rate_label, confidence = classify_rate(denoised_path)
            
            # Calculate WPM consistency for this chunk
            wpm_std = float(np.std(wpm_bins)) if len(wpm_bins) > 1 else 0.0
            consistency_score = max(0.0, min(100.0, 100.0 - (wpm_std * 3.0)))
            
            # Generate real-time feedback based on activity type and current WPM
            feedback_data = generate_realtime_rate_feedback(
                current_wpm, rate_label, consistency_score, activity_type, chunk_index
            )
            
            # Calculate chunk score (0-100)
            chunk_score = calculate_chunk_score(current_wpm, consistency_score, activity_type)
            
            # Calculate progress towards ideal WPM
            progress = feedback_data["progress_towards_ideal"]
            
            # Clean up temp files
            try:
                os.remove(chunk_path)
                os.remove(denoised_path)
            except Exception:
                pass
            
            return JSONResponse({
                "success": True,
                "chunk_index": chunk_index,
                "session_id": session_id,
                "current_wpm": round(current_wpm, 1),
                "rate_classification": rate_label,
                "confidence": round(confidence, 2),
                "consistency_score": round(consistency_score, 1),
                "chunk_score": round(chunk_score, 1),
                "feedback": feedback_data["feedback"],
                "suggestion": feedback_data["suggestion"],
                "target_wpm": feedback_data["target_wpm"],
                "progress_towards_ideal": feedback_data["progress_towards_ideal"],
                "score": round(chunk_score, 1),
                "targetScore": 100,
                "progress": progress,
                "metrics": {
                    "wpm": round(current_wpm, 1),
                    "consistency": round(consistency_score, 1),
                    "classification": rate_label,
                    "confidence": round(confidence, 2)
                },
                "timestamp": time.time(),
                "is_mock": False
            })
            
        except Exception as e:
            # Clean up on error
            try:
                os.remove(chunk_path)
            except Exception:
                pass
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Real-time chunk processing error: {str(e)}")


def generate_mock_feedback(activity_type: str, duration: float) -> Dict:
    """Generate mock feedback for duration-based requests."""
    mock_feedbacks = {
        "pacing_curve": {
            "score": 78,
            "feedback": "Good pacing detected! Keep maintaining steady rhythm.",
            "metrics": {"wpm": 128, "consistency": 82}
        },
        "rate_match": {
            "score": 85,
            "feedback": "You're close to the ideal pace! Target 125 WPM.",
            "metrics": {"wpm": 130, "consistency": 88}
        },
        "speed_shift": {
            "score": 72,
            "feedback": "Nice pace variation! Work on smoother transitions.",
            "metrics": {"wpm": 135, "consistency": 75}
        },
        "consistency_tracker": {
            "score": 90,
            "feedback": "Excellent consistency! You're maintaining steady pace.",
            "metrics": {"wpm": 122, "consistency": 95}
        }
    }
    
    return mock_feedbacks.get(activity_type, mock_feedbacks["rate_match"])


@router.post("/streaming-analysis/")
async def streaming_analysis_rate(
    audio_chunk: UploadFile = File(...),
    activity_type: str = Form(...),
    chunk_index: int = Form(0),
    session_id: str = Form("default")
):
    """Real-time analysis of 3-second audio chunks for rate activities.
    
    This endpoint processes streaming audio chunks and provides immediate feedback
    to help users achieve ideal WPM through real-time coaching.
    """
    try:
        # Save the audio chunk temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio_chunk.read())
            chunk_path = tmp.name
        
        try:
            # Process the 3-second chunk
            y, sr = librosa.load(chunk_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Skip if chunk is too short (less than 1 second)
            if duration < 1.0:
                return JSONResponse({
                    "success": False,
                    "message": "Audio chunk too short",
                    "chunk_index": chunk_index,
                    "session_id": session_id
                })
            
            # Transcribe and analyze
            denoised_path = denoise_wav_to_path(chunk_path)
            transcription, pacing_curve, wpm_bins = transcribe_and_pacing(denoised_path)
            
            # Calculate current WPM
            word_count = len(transcription.split())
            current_wpm = word_count / (duration / 60.0) if duration > 0 else 0
            
            # Classify rate using ML model
            rate_label, confidence = classify_rate(denoised_path)
            
            # Calculate WPM consistency for this chunk
            wpm_std = float(np.std(wpm_bins)) if len(wpm_bins) > 1 else 0.0
            consistency_score = max(0.0, min(100.0, 100.0 - (wpm_std * 3.0)))
            
            # Generate real-time feedback based on activity type and current WPM
            feedback_data = generate_realtime_rate_feedback(
                current_wpm, rate_label, consistency_score, activity_type, chunk_index
            )
            
            # Calculate chunk score (0-100)
            chunk_score = calculate_chunk_score(current_wpm, consistency_score, activity_type)
            
            # Clean up temp files
            try:
                os.remove(chunk_path)
                os.remove(denoised_path)
            except Exception:
                pass
            
            return JSONResponse({
                "success": True,
                "chunk_index": chunk_index,
                "session_id": session_id,
                "current_wpm": round(current_wpm, 1),
                "rate_classification": rate_label,
                "confidence": round(confidence, 2),
                "consistency_score": round(consistency_score, 1),
                "chunk_score": round(chunk_score, 1),
                "feedback": feedback_data["feedback"],
                "suggestion": feedback_data["suggestion"],
                "target_wpm": feedback_data["target_wpm"],
                "progress_towards_ideal": feedback_data["progress_towards_ideal"],
                "timestamp": time.time()
            })
            
        except Exception as e:
            # Clean up on error
            try:
                os.remove(chunk_path)
            except Exception:
                pass
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming analysis error: {str(e)}")


def generate_realtime_rate_feedback(wpm: float, rate_label: str, consistency: float, activity_type: str, chunk_index: int) -> Dict:
    """Generate real-time feedback for rate activities based on current performance."""
    
    # Activity-specific targets and feedback
    targets = {
        "pacing_curve": {"min_wpm": 100, "max_wpm": 150, "ideal": 125},
        "rate_match": {"min_wpm": 110, "max_wpm": 140, "ideal": 125},
        "speed_shift": {"min_wpm": 90, "max_wpm": 160, "ideal": 125},
        "consistency_tracker": {"min_wpm": 100, "max_wpm": 150, "ideal": 125}
    }
    
    target = targets.get(activity_type, targets["rate_match"])
    ideal_wpm = target["ideal"]
    min_wpm = target["min_wpm"]
    max_wpm = target["max_wpm"]
    
    # Calculate progress towards ideal
    if wpm == 0:
        progress = 0
    elif wpm < min_wpm:
        progress = (wpm / min_wpm) * 50  # 0-50% range for too slow
    elif wpm > max_wpm:
        progress = 50 + ((max_wpm - wpm) / (max_wpm - ideal_wpm)) * 50  # 50-100% range for too fast
    else:
        # In ideal range
        progress = 50 + ((ideal_wpm - abs(wpm - ideal_wpm)) / ideal_wpm) * 50
    
    progress = max(0, min(100, progress))
    
    # Generate feedback messages
    feedback_messages = {
        "too_slow": [
            "🚀 Speed up! You're below the ideal range.",
            "💨 Pick up the pace for better engagement.",
            "⚡ Increase your speaking speed gradually.",
            "🎯 Aim for 120-140 WPM for clarity."
        ],
        "too_fast": [
            "🛑 Slow down! You're speaking too quickly.",
            "🐌 Take your time for better comprehension.",
            "⏰ Pause more between sentences.",
            "🎯 Aim for 120-140 WPM for clarity."
        ],
        "ideal": [
            "✅ Perfect pace! Keep it up!",
            "🎉 Excellent rhythm and speed.",
            "🏆 You're in the ideal range!",
            "⭐ Great pacing consistency!"
        ],
        "inconsistent": [
            "📊 Work on maintaining steady pace.",
            "🎵 Keep consistent rhythm throughout.",
            "⚖️ Balance your speaking speed.",
            "🎯 Practice with a metronome."
        ]
    }
    
    # Determine feedback category
    if wpm < min_wpm:
        category = "too_slow"
        suggestion = f"Try to increase your pace to {min_wpm}-{ideal_wpm} WPM"
    elif wpm > max_wpm:
        category = "too_fast" 
        suggestion = f"Slow down to {ideal_wpm}-{max_wpm} WPM"
    elif consistency < 70:
        category = "inconsistent"
        suggestion = "Focus on maintaining consistent speed"
    else:
        category = "ideal"
        suggestion = "Continue with your excellent pacing!"
    
    # Select random feedback message
    import random
    feedback = random.choice(feedback_messages[category])
    
    return {
        "feedback": feedback,
        "suggestion": suggestion,
        "target_wpm": ideal_wpm,
        "progress_towards_ideal": round(progress, 1)
    }


def calculate_chunk_score(wpm: float, consistency: float, activity_type: str) -> float:
    """Calculate a score (0-100) for the current audio chunk."""
    
    targets = {
        "pacing_curve": {"ideal": 125, "tolerance": 20},
        "rate_match": {"ideal": 125, "tolerance": 15},
        "speed_shift": {"ideal": 125, "tolerance": 25},
        "consistency_tracker": {"ideal": 125, "tolerance": 20}
    }
    
    target = targets.get(activity_type, targets["rate_match"])
    ideal_wpm = target["ideal"]
    tolerance = target["tolerance"]
    
    # WPM score (0-70 points)
    if wpm == 0:
        wpm_score = 0
    else:
        wpm_diff = abs(wpm - ideal_wpm)
        wpm_score = max(0, 70 - (wpm_diff / tolerance) * 70)
    
    # Consistency score (0-30 points)
    consistency_score = (consistency / 100.0) * 30
    
    total_score = wpm_score + consistency_score
    return min(100, max(0, total_score))


@router.post("/ideal-pace-challenge/")
async def ideal_pace_challenge(request: Dict):
    """Special challenge activity: Talk until you achieve ideal WPM consistently.
    
    This is a gamified activity where users try to maintain ideal pace for extended periods.
    Can accept either audio chunks or duration-based requests.
    """
    try:
        # Check if this is a real audio chunk or just duration-based request
        if "audioChunk" in request:
            return await process_ideal_pace_chunk(request)
        else:
            # Duration-based mock for ideal pace challenge
            duration = float(request.get("duration", 0))
            session_id = request.get("sessionId", "default")
            
            # Generate mock challenge data
            mock_wpm = 125 + (np.random.random() - 0.5) * 20  # Random WPM around ideal
            mock_consistency = 70 + np.random.random() * 25   # Random consistency 70-95
            
            ideal_wpm = 125
            wpm_tolerance = 15
            is_ideal = abs(mock_wpm - ideal_wpm) <= wpm_tolerance
            is_consistent = mock_consistency >= 70
            
            challenge_score = 80 if is_ideal and is_consistent else (60 if is_ideal else 40)
            
            return JSONResponse({
                "success": True,
                "session_id": session_id,
                "current_wpm": round(mock_wpm, 1),
                "is_ideal_pace": is_ideal,
                "is_consistent": is_consistent,
                "challenge_score": round(challenge_score, 1),
                "consistency_score": round(mock_consistency, 1),
                "feedback": "🎯 Keep working towards ideal pace!",
                "suggestion": f"Target {ideal_wpm} WPM (±{wpm_tolerance})",
                "achievement": "Pace Challenger",
                "target_wpm": ideal_wpm,
                "wpm_tolerance": wpm_tolerance,
                "timestamp": time.time(),
                "is_mock": True
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ideal pace challenge error: {str(e)}")


async def process_ideal_pace_chunk(request: Dict):
    """Process a 3-second audio chunk for the ideal pace challenge."""
    try:
        import base64
        
        # Extract audio data
        audio_data = request.get("audioChunk")
        session_id = request.get("sessionId", "default")
        chunk_index = request.get("chunkIndex", 0)
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Save the audio chunk temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            chunk_path = tmp.name
        
        try:
            # Process the chunk
            denoised_path = denoise_wav_to_path(chunk_path)
            transcription, pacing_curve, wpm_bins = transcribe_and_pacing(denoised_path)
            
            y, sr = librosa.load(chunk_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            word_count = len(transcription.split())
            current_wpm = word_count / (duration / 60.0) if duration > 0 else 0
            
            rate_label, confidence = classify_rate(denoised_path)
            wpm_std = float(np.std(wpm_bins)) if len(wpm_bins) > 1 else 0.0
            consistency_score = max(0.0, min(100.0, 100.0 - (wpm_std * 3.0)))
            
            # Challenge scoring - stricter criteria for ideal pace
            ideal_wpm = 125
            wpm_tolerance = 12  # ±12 WPM from ideal (stricter than regular activities)
            
            # Check if current chunk is in ideal range
            is_ideal = (ideal_wpm - wpm_tolerance) <= current_wpm <= (ideal_wpm + wpm_tolerance)
            is_consistent = consistency_score >= 75  # Higher consistency requirement
            
            # Calculate challenge score with bonus for perfect performance
            challenge_score = 0
            if is_ideal and is_consistent:
                challenge_score = 100
            elif is_ideal:
                challenge_score = 80
            elif is_consistent:
                challenge_score = 65
            else:
                # Score based on distance from ideal
                wpm_diff = abs(current_wpm - ideal_wpm)
                challenge_score = max(0, 50 - (wpm_diff / wpm_tolerance) * 25)
            
            # Generate challenge-specific feedback with encouragement
            if is_ideal and is_consistent:
                feedback = "🎯 PERFECT! You're hitting the ideal pace consistently!"
                suggestion = "Keep this rhythm going! You're mastering ideal pace!"
                achievement = "Ideal Pace Master"
                emoji = "🏆"
            elif is_ideal:
                feedback = "✅ Great pace, but work on consistency!"
                suggestion = "Try to maintain this speed more steadily"
                achievement = "Pace Achiever"
                emoji = "⭐"
            elif is_consistent:
                feedback = "📊 Good consistency, but adjust your speed!"
                suggestion = f"Target {ideal_wpm} WPM (±{wpm_tolerance}) for ideal pace"
                achievement = "Consistency Champion"
                emoji = "🎯"
            else:
                feedback = "🎯 Focus on both pace and consistency!"
                suggestion = f"Target {ideal_wpm} WPM and maintain steady rhythm"
                achievement = "Keep Practicing"
                emoji = "💪"
            
            # Calculate progress towards ideal (0-100%)
            if current_wpm == 0:
                progress = 0
            else:
                wpm_diff = abs(current_wpm - ideal_wpm)
                progress = max(0, min(100, 100 - (wpm_diff / wpm_tolerance) * 100))
            
            # Clean up
            try:
                os.remove(chunk_path)
                os.remove(denoised_path)
            except Exception:
                pass
            
            return JSONResponse({
                "success": True,
                "session_id": session_id,
                "chunk_index": chunk_index,
                "current_wpm": round(current_wpm, 1),
                "is_ideal_pace": is_ideal,
                "is_consistent": is_consistent,
                "challenge_score": round(challenge_score, 1),
                "consistency_score": round(consistency_score, 1),
                "feedback": feedback,
                "suggestion": suggestion,
                "achievement": achievement,
                "emoji": emoji,
                "target_wpm": ideal_wpm,
                "wpm_tolerance": wpm_tolerance,
                "progress_towards_ideal": round(progress, 1),
                "rate_classification": rate_label,
                "confidence": round(confidence, 2),
                "timestamp": time.time(),
                "is_mock": False
            })
            
        except Exception as e:
            try:
                os.remove(chunk_path)
            except Exception:
                pass
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ideal pace chunk processing error: {str(e)}")


@router.get("/challenge-status/{session_id}")
async def get_challenge_status(session_id: str):
    """Get the current status of an ideal pace challenge session."""
    # In a real implementation, this would track session data in a database
    # For now, return a mock status
    return JSONResponse({
        "session_id": session_id,
        "challenge_active": True,
        "total_chunks_analyzed": 0,
        "ideal_chunks_count": 0,
        "average_score": 0,
        "current_streak": 0,
        "best_streak": 0,
        "challenge_duration": 60,
        "time_remaining": 60
    })


@router.post("/analyze-activity/")
async def analyze_rate_activity(file: UploadFile = File(...)):
    """Analyze a finished recording for rate-focused activities (pacing_curve, rate_match, speed_shift, consistency_tracker)."""
    try:
        # Persist upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            denoised = denoise_wav_to_path(tmp_path)
            transcription, pacing_curve, wpm_bins = transcribe_and_pacing(denoised)
            label, conf = classify_rate(denoised)

            # Stats
            y, sr = librosa.load(denoised, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            words = len(transcription.split())
            wpm = words / (duration / 60.0) if duration > 0 else 0.0
            wpm_std = float(np.std(wpm_bins)) if len(wpm_bins) else 0.0
            wpm_consistency = max(0.0, min(100.0, 100.0 - (wpm_std * 2.0)))

            # Scoring per activity flavor (pacing consistency + closeness to 125)
            target_wpm = 125.0
            wpm_score = max(0.0, 100.0 - abs(wpm - target_wpm) * 2.0)
            final_score = round((wpm_score + wpm_consistency) / 2.0, 1)

            # Ideal lines for chart
            ideal_lines = [{"time": pc["time"], "upper": 150, "lower": 100} for pc in pacing_curve]

            return JSONResponse({
                "success": True,
                "finalScore": final_score,
                "averageWPM": round(wpm, 1),
                "consistencyScore": round(wpm_consistency, 1),
                "wpmStd": round(wpm_std, 2),
                "prediction": label,
                "confidence": conf,
                "pacingCurve": pacing_curve,
                "idealLines": ideal_lines,
                "duration": round(duration, 2),
                "recommendations": [
                    "Maintain 120–150 WPM for clarity.",
                    "Use metronome practice for steadier rhythm.",
                ]
            })
        finally:
            # Clean temp files
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            try:
                os.remove(denoised)
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rate activity analysis error: {str(e)}")



