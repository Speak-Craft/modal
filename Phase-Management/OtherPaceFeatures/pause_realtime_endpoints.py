from fastapi import APIRouter, HTTPException, Form, File, UploadFile, Query, Header
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import tempfile
import os
import base64
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

# Import the core pause analysis functions
from feature_extraction2 import pause_features_for_file, suggest_from_features
import joblib

router = APIRouter()

# Real-time pause activity types
PAUSE_REALTIME_ACTIVITIES = {
    "pause_monitoring": "Real-time pause detection and alerts",
    "pause_improvement": "Interactive pause improvement coaching",
    "pause_rhythm_training": "Rhythm-based pause training",
    "confidence_pause_practice": "Confidence pause practice",
    "impact_pause_training": "Impact pause timing training"
}

# Scoring thresholds for real-time pause activities
PAUSE_SCORING_THRESHOLDS = {
    "optimal_pause_ratio": (0.08, 0.12),  # 8-12% is optimal
    "excessive_pause_threshold": 5.0,     # 5+ seconds is excessive
    "long_pause_threshold": 2.5,          # 2.5+ seconds is long
    "short_pause_threshold": 0.5,         # 0.5+ seconds is meaningful
    "confidence_pause_range": (1.5, 2.0), # 1.5-2.0s for confidence
    "impact_pause_range": (1.0, 1.5),     # 1.0-1.5s for impact
}

# Badge system for pause activities
PAUSE_BADGES = {
    "PAUSE_MASTER": {
        "name": "Pause Master",
        "description": "Achieved optimal pause timing",
        "requirement": "80%+ optimal pause ratio"
    },
    "FLOW_GUARDIAN": {
        "name": "Flow Guardian", 
        "description": "Maintained speech flow without excessive pauses",
        "requirement": "Zero excessive pauses for 5+ sessions"
    },
    "RHYTHM_KEEPER": {
        "name": "Rhythm Keeper",
        "description": "Consistent pause rhythm",
        "requirement": "85%+ rhythm consistency"
    },
    "CONFIDENCE_BUILDER": {
        "name": "Confidence Builder",
        "description": "Perfect confidence pause timing",
        "requirement": "90%+ confidence pause accuracy"
    },
    "IMPACT_SPECIALIST": {
        "name": "Impact Specialist",
        "description": "Mastered impact pause timing",
        "requirement": "95%+ impact pause accuracy"
    }
}

@dataclass
class RealtimePauseMetrics:
    """Real-time pause metrics for monitoring"""
    total_duration: float
    speech_duration: float
    pause_duration: float
    pause_ratio: float
    pause_count: int
    short_pauses: int
    medium_pauses: int
    long_pauses: int
    excessive_pauses: int
    current_pause_duration: float
    average_pause_length: float
    pause_rhythm_score: float
    confidence_score: float
    impact_score: float

def detect_pauses_realtime(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """Detect pauses in real-time audio chunk"""
    try:
        # Calculate RMS energy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Dynamic threshold based on audio level
        energy_threshold = np.percentile(rms, 20) * 1.5
        
        # Find silence periods
        silence_frames = rms < energy_threshold
        silence_duration = np.sum(silence_frames) * hop_length / sr
        
        # Calculate speech vs pause ratio
        total_duration = len(audio_data) / sr
        speech_duration = total_duration - silence_duration
        pause_ratio = silence_duration / total_duration if total_duration > 0 else 0
        
        # Classify pauses by duration
        pause_segments = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            time = i * hop_length / sr
            
            if is_silent and not in_pause:
                pause_start = time
                in_pause = True
            elif not is_silent and in_pause:
                pause_duration = time - pause_start
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pause_segments.append(pause_duration)
                in_pause = False
        
        # Count pauses by category
        short_pauses = sum(1 for p in pause_segments if 0.5 <= p < 1.5)
        medium_pauses = sum(1 for p in pause_segments if 1.5 <= p < 2.5)
        long_pauses = sum(1 for p in pause_segments if 2.5 <= p < 5.0)
        excessive_pauses = sum(1 for p in pause_segments if p >= 5.0)
        
        return {
            "total_duration": total_duration,
            "speech_duration": speech_duration,
            "pause_duration": silence_duration,
            "pause_ratio": pause_ratio,
            "pause_count": len(pause_segments),
            "short_pauses": short_pauses,
            "medium_pauses": medium_pauses,
            "long_pauses": long_pauses,
            "excessive_pauses": excessive_pauses,
            "average_pause_length": np.mean(pause_segments) if pause_segments else 0,
            "pause_segments": pause_segments,
            "current_pause_duration": pause_segments[-1] if pause_segments else 0
        }
        
    except Exception as e:
        print(f"Error in real-time pause detection: {e}")
        return {
            "total_duration": 0,
            "speech_duration": 0,
            "pause_duration": 0,
            "pause_ratio": 0,
            "pause_count": 0,
            "short_pauses": 0,
            "medium_pauses": 0,
            "long_pauses": 0,
            "excessive_pauses": 0,
            "average_pause_length": 0,
            "pause_segments": [],
            "current_pause_duration": 0
        }

def calculate_pause_scores(metrics: Dict[str, Any], activity_type: str) -> Dict[str, Any]:
    """Calculate scores for real-time pause activities"""
    scores = {}
    
    # Base pause ratio score
    pause_ratio = metrics.get("pause_ratio", 0)
    optimal_min, optimal_max = PAUSE_SCORING_THRESHOLDS["optimal_pause_ratio"]
    
    if optimal_min <= pause_ratio <= optimal_max:
        scores["pause_ratio_score"] = 100
    elif pause_ratio < optimal_min:
        scores["pause_ratio_score"] = max(0, 100 - (optimal_min - pause_ratio) * 500)
    else:
        scores["pause_ratio_score"] = max(0, 100 - (pause_ratio - optimal_max) * 300)
    
    # Excessive pause penalty
    excessive_count = metrics.get("excessive_pauses", 0)
    scores["excessive_penalty"] = max(0, 100 - excessive_count * 20)
    
    # Flow score (combination of ratio and excessive pauses)
    scores["flow_score"] = (scores["pause_ratio_score"] + scores["excessive_penalty"]) / 2
    
    # Activity-specific scores
    if activity_type == "pause_monitoring":
        # Focus on avoiding excessive pauses and maintaining flow
        scores["activity_score"] = scores["flow_score"]
        
    elif activity_type == "pause_improvement":
        # Focus on optimal pause timing
        avg_pause = metrics.get("average_pause_length", 0)
        if 0.5 <= avg_pause <= 2.0:
            scores["timing_score"] = 100
        else:
            scores["timing_score"] = max(0, 100 - abs(avg_pause - 1.25) * 40)
        
        scores["activity_score"] = (scores["pause_ratio_score"] + scores["timing_score"]) / 2
        
    elif activity_type == "pause_rhythm_training":
        # Focus on consistent pause patterns
        pause_segments = metrics.get("pause_segments", [])
        if len(pause_segments) > 1:
            pause_std = np.std(pause_segments)
            scores["rhythm_score"] = max(0, 100 - pause_std * 50)
        else:
            scores["rhythm_score"] = 50  # Neutral score for insufficient data
            
        scores["activity_score"] = scores["rhythm_score"]
        
    elif activity_type == "confidence_pause_practice":
        # Focus on 1.5-2.0 second pauses
        confidence_pauses = [p for p in metrics.get("pause_segments", []) 
                           if PAUSE_SCORING_THRESHOLDS["confidence_pause_range"][0] <= p <= PAUSE_SCORING_THRESHOLDS["confidence_pause_range"][1]]
        total_pauses = metrics.get("pause_count", 1)
        confidence_ratio = len(confidence_pauses) / total_pauses
        scores["confidence_score"] = confidence_ratio * 100
        scores["activity_score"] = scores["confidence_score"]
        
    elif activity_type == "impact_pause_training":
        # Focus on 1.0-1.5 second pauses
        impact_pauses = [p for p in metrics.get("pause_segments", []) 
                        if PAUSE_SCORING_THRESHOLDS["impact_pause_range"][0] <= p <= PAUSE_SCORING_THRESHOLDS["impact_pause_range"][1]]
        total_pauses = metrics.get("pause_count", 1)
        impact_ratio = len(impact_pauses) / total_pauses
        scores["impact_score"] = impact_ratio * 100
        scores["activity_score"] = scores["impact_score"]
    
    return scores

def generate_realtime_pause_feedback(metrics: Dict[str, Any], scores: Dict[str, Any], activity_type: str) -> Dict[str, Any]:
    """Generate real-time feedback for pause activities"""
    feedback = {
        "alerts": [],
        "suggestions": [],
        "encouragement": [],
        "current_status": "Good"
    }
    
    # Check for immediate alerts
    excessive_pauses = metrics.get("excessive_pauses", 0)
    if excessive_pauses > 0:
        feedback["alerts"].append({
            "type": "warning",
            "message": f"🚨 EXCESSIVE PAUSE ALERT! {excessive_pauses} pause(s) longer than 5 seconds",
            "action": "Immediately shorten your pauses to maintain audience engagement!"
        })
        feedback["current_status"] = "Critical Alert"
    
    long_pauses = metrics.get("long_pauses", 0)
    if long_pauses > 4:
        feedback["alerts"].append({
            "type": "warning",
            "message": f"⚠️ Too many long pauses: {long_pauses} pauses over 2.5 seconds",
            "action": "Break up long pauses with shorter, strategic ones"
        })
        feedback["current_status"] = "Needs Attention"
    elif long_pauses > 2:
        feedback["alerts"].append({
            "type": "caution",
            "message": f"⏱️ Moderate long pauses detected: {long_pauses} pauses",
            "action": "Consider using shorter pauses for better rhythm"
        })
        feedback["current_status"] = "Good with Room for Improvement"
    
    # Check pause ratio
    pause_ratio = metrics.get("pause_ratio", 0)
    if pause_ratio > 0.15:
        feedback["alerts"].append({
            "type": "warning",
            "message": f"⏱️ High pause ratio ({pause_ratio*100:.1f}%)",
            "action": "Reduce pause time to 8-12% of speech duration"
        })
    elif pause_ratio < 0.05:
        feedback["alerts"].append({
            "type": "info",
            "message": f"💨 Low pause ratio ({pause_ratio*100:.1f}%)",
            "action": "Add strategic pauses for emphasis and clarity"
        })
    
    # Activity-specific feedback
    if activity_type == "pause_monitoring":
        if scores.get("flow_score", 0) > 80:
            feedback["encouragement"].append("🌟 Excellent flow! Your pause timing is natural.")
        elif scores.get("flow_score", 0) > 60:
            feedback["suggestions"].append("💡 Good flow, try to optimize pause timing further.")
        else:
            feedback["suggestions"].append("🎯 Focus on reducing excessive pauses and maintaining steady rhythm.")
            
    elif activity_type == "pause_improvement":
        if scores.get("activity_score", 0) > 85:
            feedback["encouragement"].append("🎉 Outstanding pause improvement! Your timing is excellent.")
        elif scores.get("activity_score", 0) > 70:
            feedback["suggestions"].append("📈 Good progress! Fine-tune your pause timing for perfection.")
        else:
            feedback["suggestions"].append("🎯 Practice pausing for 0.5-2 seconds for optimal timing.")
            
    elif activity_type == "pause_rhythm_training":
        if scores.get("rhythm_score", 0) > 80:
            feedback["encouragement"].append("🎵 Perfect rhythm! Your pause patterns are consistent.")
        elif scores.get("rhythm_score", 0) > 60:
            feedback["suggestions"].append("🎶 Good rhythm, work on making pause lengths more consistent.")
        else:
            feedback["suggestions"].append("🎼 Focus on maintaining consistent pause timing throughout.")
            
    elif activity_type == "confidence_pause_practice":
        if scores.get("confidence_score", 0) > 90:
            feedback["encouragement"].append("💪 Excellent confidence pauses! Perfect 1.5-2s timing.")
        elif scores.get("confidence_score", 0) > 70:
            feedback["suggestions"].append("💪 Good confidence pauses, aim for 1.5-2 second timing.")
        else:
            feedback["suggestions"].append("💪 Practice pausing for 1.5-2 seconds to show confidence.")
            
    elif activity_type == "impact_pause_training":
        if scores.get("impact_score", 0) > 90:
            feedback["encouragement"].append("⚡ Perfect impact pauses! Great 1.0-1.5s timing.")
        elif scores.get("impact_score", 0) > 70:
            feedback["suggestions"].append("⚡ Good impact pauses, aim for 1.0-1.5 second timing.")
        else:
            feedback["suggestions"].append("⚡ Practice pausing for 1.0-1.5 seconds for maximum impact.")
    
    # General encouragement
    if not feedback["alerts"] and not feedback["suggestions"]:
        feedback["encouragement"].append("🎯 Keep up the great work! Your pause management is on track.")
    
    return feedback

@router.post("/pause/realtime-monitoring/")
async def realtime_pause_monitoring(
    activityType: str = Form("pause_monitoring"),
    audioChunk: Optional[str] = Form(None),
    duration: Optional[float] = Form(None),
    sessionId: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None)
):
    """Real-time pause monitoring endpoint"""
    try:
        # Handle mock data for testing
        if audioChunk is None and duration is not None:
            # Generate more realistic mock pause data based on duration and activity type
            import random
            
            # Simulate different pause patterns based on activity type
            if activityType == "pause_monitoring":
                # Simulate monitoring with some issues
                pause_ratio = random.uniform(0.10, 0.18)  # Sometimes too high
                excessive_count = random.randint(0, 1) if duration > 15 else 0
                long_count = random.randint(1, 4)
            elif activityType == "pause_improvement":
                # Simulate improvement challenge
                pause_ratio = random.uniform(0.08, 0.15)
                excessive_count = 0
                long_count = random.randint(0, 2)
            elif activityType == "confidence_pause_practice":
                # Simulate confidence pause practice
                pause_ratio = random.uniform(0.12, 0.20)
                excessive_count = 0
                long_count = random.randint(0, 1)
            else:
                # Default pattern
                pause_ratio = random.uniform(0.10, 0.14)
                excessive_count = random.randint(0, 1) if duration > 20 else 0
                long_count = random.randint(0, 3)
            
            # Generate realistic pause segments
            pause_segments = []
            for i in range(int(duration / 4)):  # Roughly one pause every 4 seconds
                if random.random() < 0.7:  # 70% chance of a pause
                    pause_length = random.uniform(0.3, 3.0)
                    if pause_length > 2.5:
                        pause_length = random.uniform(2.5, 6.0)  # Some longer pauses
                    pause_segments.append(pause_length)
            
            # Categorize pauses
            short_pauses = sum(1 for p in pause_segments if 0.5 <= p < 1.5)
            medium_pauses = sum(1 for p in pause_segments if 1.5 <= p < 2.5)
            long_pauses = sum(1 for p in pause_segments if 2.5 <= p < 5.0)
            excessive_pauses = sum(1 for p in pause_segments if p >= 5.0)
            
            mock_metrics = {
                "total_duration": duration,
                "speech_duration": duration * (1 - pause_ratio),
                "pause_duration": duration * pause_ratio,
                "pause_ratio": pause_ratio,
                "pause_count": len(pause_segments),
                "short_pauses": short_pauses,
                "medium_pauses": medium_pauses,
                "long_pauses": long_pauses,
                "excessive_pauses": excessive_pauses,
                "average_pause_length": np.mean(pause_segments) if pause_segments else 0,
                "pause_segments": pause_segments,
                "current_pause_duration": pause_segments[-1] if pause_segments else 0
            }
            
            scores = calculate_pause_scores(mock_metrics, activityType)
            feedback = generate_realtime_pause_feedback(mock_metrics, scores, activityType)
            
            return {
                "success": True,
                "activity_type": activityType,
                "metrics": mock_metrics,
                "scores": scores,
                "feedback": feedback,
                "is_mock": True,
                "timestamp": datetime.now().isoformat()
            }
        
        # Process actual audio chunk
        if audioChunk is None:
            raise HTTPException(status_code=400, detail="Audio chunk or duration required")
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audioChunk)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Load audio with librosa
            audio_data, sr = librosa.load(tmp_file_path, sr=16000)
            
            # Detect pauses
            pause_metrics = detect_pauses_realtime(audio_data, sr)
            
            # Calculate scores
            scores = calculate_pause_scores(pause_metrics, activityType)
            
            # Generate feedback
            feedback = generate_realtime_pause_feedback(pause_metrics, scores, activityType)
            
            return {
                "success": True,
                "activity_type": activityType,
                "metrics": pause_metrics,
                "scores": scores,
                "feedback": feedback,
                "is_mock": False,
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        print(f"Error in real-time pause monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Real-time pause monitoring error: {str(e)}")

@router.post("/pause/pause-improvement-challenge/")
async def pause_improvement_challenge(
    activityType: str = Form("pause_improvement"),
    audioChunk: Optional[str] = Form(None),
    duration: Optional[float] = Form(None),
    sessionId: Optional[str] = Form(None),
    targetPauseRatio: Optional[float] = Form(0.10),
    timestamp: Optional[str] = Form(None)
):
    """Interactive pause improvement challenge"""
    try:
        # Use the same logic as realtime_pause_monitoring but with improvement focus
        result = await realtime_pause_monitoring(
            activityType=activityType,
            audioChunk=audioChunk,
            duration=duration,
            sessionId=sessionId,
            timestamp=timestamp
        )
        
        # Add improvement-specific enhancements
        metrics = result["metrics"]
        scores = result["scores"]
        
        # Calculate improvement progress
        target_ratio = targetPauseRatio or 0.10
        current_ratio = metrics.get("pause_ratio", 0)
        improvement_progress = max(0, 100 - abs(current_ratio - target_ratio) * 1000)
        
        # Enhanced feedback for improvement
        improvement_feedback = {
            "progress": improvement_progress,
            "target_achieved": abs(current_ratio - target_ratio) < 0.02,
            "next_goal": f"Aim for {target_ratio*100:.1f}% pause ratio",
            "current_performance": f"Current: {current_ratio*100:.1f}%",
            "improvement_tip": "Focus on natural breathing pauses at sentence boundaries"
        }
        
        result["improvement_feedback"] = improvement_feedback
        result["target_pause_ratio"] = target_ratio
        
        return result
        
    except Exception as e:
        print(f"Error in pause improvement challenge: {e}")
        raise HTTPException(status_code=500, detail=f"Pause improvement challenge error: {str(e)}")

@router.post("/pause/analyze-pause-session/")
async def analyze_pause_session(
    activityType: str = Form("pause_monitoring"),
    file: UploadFile = File(...),
    sessionId: Optional[str] = Form(None),
    totalDuration: Optional[float] = Form(None)
):
    """Analyze complete pause session for final scoring and badge eligibility"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Extract comprehensive pause features
            features = pause_features_for_file(tmp_file_path)
            
            # Calculate session scores
            session_metrics = {
                "total_duration": totalDuration or features.get("total_duration", 0),
                "pause_ratio": features.get("pause_ratio", 0),
                "excessive_pauses": features.get("excessive_count", 0),
                "long_pauses": features.get("long_count", 0),
                "medium_pauses": features.get("med_count", 0),
                "short_pauses": features.get("short_count", 0),
                "average_pause_length": features.get("pause_p50", 0),
                "pause_rhythm_consistency": features.get("pause_rhythm_consistency", 0),
                "confidence_score": features.get("confidence_score", 0)
            }
            
            # Calculate comprehensive scores
            scores = calculate_pause_scores(session_metrics, activityType)
            
            # Check badge eligibility
            badges_earned = []
            if session_metrics["pause_ratio"] >= 0.08 and session_metrics["pause_ratio"] <= 0.12:
                if session_metrics["excessive_pauses"] == 0:
                    badges_earned.append(PAUSE_BADGES["PAUSE_MASTER"])
            
            if session_metrics["excessive_pauses"] == 0:
                badges_earned.append(PAUSE_BADGES["FLOW_GUARDIAN"])
            
            if session_metrics["pause_rhythm_consistency"] > 0.85:
                badges_earned.append(PAUSE_BADGES["RHYTHM_KEEPER"])
            
            if session_metrics["confidence_score"] > 0.90:
                badges_earned.append(PAUSE_BADGES["CONFIDENCE_BUILDER"])
            
            # Generate final recommendations
            recommendations = []
            if session_metrics["excessive_pauses"] > 0:
                recommendations.append("Eliminate excessive pauses (>5s) to improve flow")
            
            if session_metrics["pause_ratio"] > 0.15:
                recommendations.append("Reduce overall pause time to 8-12% of speech duration")
            elif session_metrics["pause_ratio"] < 0.08:
                recommendations.append("Add strategic pauses for emphasis and clarity")
            
            if session_metrics["pause_rhythm_consistency"] < 0.70:
                recommendations.append("Practice consistent pause timing patterns")
            
            # Calculate final score
            final_score = scores.get("activity_score", 0)
            
            return {
                "success": True,
                "activity_type": activityType,
                "session_metrics": session_metrics,
                "scores": scores,
                "final_score": final_score,
                "badges_earned": badges_earned,
                "recommendations": recommendations,
                "raw_features": features,
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        print(f"Error in pause session analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Pause session analysis error: {str(e)}")

@router.post("/pause/process-audio-chunk/")
async def process_audio_chunk_realtime(
    audioChunk: Optional[str] = Form(None),
    activityType: str = Form("pause_monitoring"),
    duration: Optional[float] = Form(None),
    sessionId: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None)
):
    """Process real audio chunk through ML model for real-time pause analysis"""
    try:
        print(f"Processing audio chunk for activity: {activityType}")
        
        # Handle audio chunk processing
        if audioChunk:
            try:
                # Decode base64 audio
                audio_data = base64.b64decode(audioChunk)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file_path = tmp_file.name
                
                print(f"Saved audio chunk to: {tmp_file_path}")
                
                # Process through ML model
                features = pause_features_for_file(tmp_file_path)
                print(f"Extracted features: {len(features)} features")
                
                # Load the trained model
                model_path = os.path.join(os.path.dirname(__file__), "enhanced_pause_model.joblib")
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    # Prepare features for prediction
                    feature_vector = []
                    feature_names = [
                        'pause_ratio', 'wpm_cv', 'wpm_delta_std', 'wpm_jerk', 
                        'rhythm_outliers', 'speech_continuity', 'confidence_score',
                        'short_rate_pm', 'med_rate_pm', 'long_rate_pm', 'excessive_rate_pm'
                    ]
                    
                    for name in feature_names:
                        feature_vector.append(features.get(name, 0.0))
                    
                    # Make prediction
                    prediction = model.predict([feature_vector])[0]
                    probabilities = model.predict_proba([feature_vector])[0]
                    
                    print(f"ML Prediction: {prediction}")
                    print(f"Probabilities: {probabilities}")
                    
                    # Generate real-time metrics from features
                    realtime_metrics = {
                        "pause_ratio": features.get('pause_ratio', 0.0),
                        "excessive_pauses": features.get('excessive_pause_count', 0),
                        "long_pauses": features.get('long_pause_count', 0),
                        "short_pauses": features.get('short_pause_count', 0),
                        "medium_pauses": features.get('medium_pause_count', 0),
                        "current_pause_duration": features.get('average_pause_length', 0.0),
                        "flow_score": calculate_flow_score_from_features(features),
                        "ml_prediction": prediction,
                        "ml_confidence": float(max(probabilities))
                    }
                    
                    # Generate alerts based on real ML analysis
                    alerts = generate_realtime_alerts_from_ml(features, prediction, probabilities)
                    suggestions = generate_realtime_suggestions_from_ml(features, prediction)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    return {
                        "status": "success",
                        "metrics": realtime_metrics,
                        "feedback": {
                            "alerts": alerts,
                            "suggestions": suggestions,
                            "encouragement": [f"ML Confidence: {(max(probabilities) * 100):.1f}%"]
                        },
                        "scores": {
                            "flow_score": realtime_metrics["flow_score"],
                            "activity_score": realtime_metrics["flow_score"],
                            "ml_score": float(max(probabilities)) * 100
                        },
                        "is_real_analysis": True
                    }
                    
                else:
                    print(f"Model not found at: {model_path}")
                    # Fall back to feature-based analysis
                    return generate_feature_based_analysis(features, activityType)
                    
            except Exception as e:
                print(f"Audio processing error: {e}")
                # Fall back to duration-based mock data
                return generate_mock_analysis(activityType, duration or 3.0)
        else:
            # No audio chunk, use duration for mock analysis
            return generate_mock_analysis(activityType, duration or 3.0)
            
    except Exception as e:
        print(f"Error in real-time audio processing: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

def calculate_flow_score_from_features(features: Dict[str, Any]) -> float:
    """Calculate flow score from extracted features"""
    base_score = 100.0
    
    # Penalize high pause ratio
    pause_ratio = features.get('pause_ratio', 0.0)
    if pause_ratio > 0.15:
        base_score -= (pause_ratio - 0.15) * 200
    elif pause_ratio < 0.05:
        base_score -= (0.05 - pause_ratio) * 100
    
    # Penalize excessive pauses
    excessive = features.get('excessive_pause_count', 0)
    base_score -= excessive * 10
    
    # Reward good rhythm
    rhythm_outliers = features.get('rhythm_outliers', 0)
    base_score -= rhythm_outliers * 5
    
    # Reward speech continuity
    continuity = features.get('speech_continuity', 1.0)
    base_score += (continuity - 1.0) * 20
    
    return max(0, min(100, base_score))

def generate_realtime_alerts_from_ml(features: Dict[str, Any], prediction: str, probabilities: np.ndarray) -> List[Dict]:
    """Generate alerts based on ML analysis"""
    alerts = []
    
    # Check pause ratio
    pause_ratio = features.get('pause_ratio', 0.0)
    if pause_ratio > 0.18:
        alerts.append({
            "type": "warning",
            "message": f"🚨 HIGH PAUSE RATIO: {(pause_ratio * 100):.1f}% of speech time",
            "action": "Reduce pause frequency - speak more continuously"
        })
    elif pause_ratio < 0.05:
        alerts.append({
            "type": "caution", 
            "message": f"⚡ LOW PAUSE RATIO: {(pause_ratio * 100):.1f}% - very fast speech",
            "action": "Add strategic pauses for emphasis and clarity"
        })
    
    # Check excessive pauses
    excessive = features.get('excessive_pause_count', 0)
    if excessive > 0:
        alerts.append({
            "type": "warning",
            "message": f"🚨 EXCESSIVE PAUSES: {excessive} pause(s) longer than 5 seconds",
            "action": "Immediately shorten your pauses to maintain audience engagement!"
        })
    
    # Check long pauses
    long_pauses = features.get('long_pause_count', 0)
    if long_pauses > 3:
        alerts.append({
            "type": "warning",
            "message": f"⚠️ TOO MANY LONG PAUSES: {long_pauses} pauses over 2.5 seconds",
            "action": "Break up long pauses with shorter, strategic ones"
        })
    
    # Check ML prediction confidence
    confidence = max(probabilities)
    if confidence < 0.6:
        alerts.append({
            "type": "info",
            "message": f"🤖 LOW ML CONFIDENCE: {(confidence * 100):.1f}%",
            "action": "Continue speaking for better analysis"
        })
    
    return alerts

def generate_realtime_suggestions_from_ml(features: Dict[str, Any], prediction: str) -> List[str]:
    """Generate suggestions based on ML analysis"""
    suggestions = []
    
    pause_ratio = features.get('pause_ratio', 0.0)
    if pause_ratio > 0.12:
        suggestions.append("💡 Reduce pause frequency - aim for 8-12% pause ratio")
        suggestions.append("🎯 Use shorter, more strategic pauses")
    elif pause_ratio < 0.08:
        suggestions.append("⚡ Add strategic pauses after key points")
        suggestions.append("🎭 Use pauses to create emphasis and drama")
    
    rhythm_outliers = features.get('rhythm_outliers', 0)
    if rhythm_outliers > 2:
        suggestions.append("🎵 Improve rhythm consistency")
        suggestions.append("📐 Practice more regular pause patterns")
    
    speech_continuity = features.get('speech_continuity', 1.0)
    if speech_continuity < 0.8:
        suggestions.append("🔄 Improve speech flow and continuity")
        suggestions.append("🌊 Reduce interruptions and hesitations")
    
    return suggestions

def generate_feature_based_analysis(features: Dict[str, Any], activity_type: str) -> Dict[str, Any]:
    """Generate analysis based on features when ML model is not available"""
    metrics = {
        "pause_ratio": features.get('pause_ratio', 0.0),
        "excessive_pauses": features.get('excessive_pause_count', 0),
        "long_pauses": features.get('long_pause_count', 0),
        "short_pauses": features.get('short_pause_count', 0),
        "medium_pauses": features.get('medium_pause_count', 0),
        "current_pause_duration": features.get('average_pause_length', 0.0),
        "flow_score": calculate_flow_score_from_features(features)
    }
    
    alerts = generate_realtime_alerts_from_ml(features, "unknown", np.array([0.5, 0.5]))
    suggestions = generate_realtime_suggestions_from_ml(features, "unknown")
    
    return {
        "status": "success",
        "metrics": metrics,
        "feedback": {
            "alerts": alerts,
            "suggestions": suggestions,
            "encouragement": ["Feature-based analysis (ML model not available)"]
        },
        "scores": {
            "flow_score": metrics["flow_score"],
            "activity_score": metrics["flow_score"]
        },
        "is_real_analysis": True
    }

@router.get("/pause/activity-types/")
async def get_pause_activity_types():
    """Get available real-time pause activity types"""
    return {
        "activity_types": PAUSE_REALTIME_ACTIVITIES,
        "badges": PAUSE_BADGES,
        "scoring_thresholds": PAUSE_SCORING_THRESHOLDS
    }
