# app.py - FastAPI Backend for Speech Pace Management
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import librosa
import tempfile
import os
import joblib
import json
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

# Enhanced Prediction dataclass
@dataclass
class EnhancedPrediction:
    label: str
    probs: Dict[str, float]
    suggestions: List[str]
    features: Dict[str, float]
    confidence: float
    improvement_areas: List[str]

# Import your feature extraction functions
from feature_extraction2 import (
    pause_features_for_file,
    suggest_from_features,
    generate_actionable_suggestions,
    generate_real_time_feedback,
    generate_personalized_feedback,
    generate_comprehensive_report,
    identify_improvement_areas,
    predict_file,
    EnhancedPrediction
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SpeakCraft API", description="AI-Powered Speech Pace Management", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
pause_model = None
pause_model_config = None
PAUSE_MODEL_PATH = "enhanced_pause_model.joblib"
PAUSE_CONFIG_PATH = "enhanced_pause_features.json"

def load_pause_model():
    """Load the trained pause analysis model and configuration"""
    global pause_model, pause_model_config
    
    try:
        if os.path.exists(PAUSE_MODEL_PATH) and os.path.exists(PAUSE_CONFIG_PATH):
            pause_model = joblib.load(PAUSE_MODEL_PATH)
            with open(PAUSE_CONFIG_PATH, "r") as f:
                pause_model_config = json.load(f)
            logger.info("✅ Pause analysis model loaded successfully")
            return True
        else:
            logger.warning("❌ Pause model files not found. Please train the model first.")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading pause model: {e}")
        return False

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_pause_model()

@app.get("/")
async def root():
    """API information and available endpoints"""
    pause_model_status = "loaded" if pause_model is not None else "not loaded"
    return {
        "message": "SpeakCraft API is running",
        "pause_model_status": pause_model_status,
        "version": "1.0.0",
        "available_endpoints": {
            "POST /analyze/": "Complete speech analysis with visualizations",
            "POST /pause-analysis/": "Focus on pause pattern analysis only",
            "POST /suggestions/": "Get detailed improvement suggestions",
            "POST /real-time-feedback/": "Real-time feedback for practice sessions",
            "POST /comprehensive-report/": "Detailed technical analysis report",
            "POST /improvement-areas/": "Identify specific areas needing improvement",
            "GET /model/status": "Check model training status",
            "GET /health": "API health check"
        },
        "features": [
            "Advanced pause analysis with Toastmasters standards",
            "Real-time feedback generation",
            "Comprehensive technical reports",
            "Personalized improvement suggestions",
            "Voice quality analysis",
            "Rhythm and flow assessment",
            "Cognitive load indicators"
        ]
    }

@app.get("/model/status")
async def model_status():
    """Check model status"""
    if pause_model is None:
        return {"status": "not_loaded", "message": "Pause model not loaded. Please train the model first."}
    
    return {
        "status": "loaded",
        "model_type": pause_model_config.get("model_type", "Unknown"),
        "performance": pause_model_config.get("performance", {}),
        "features_count": len(pause_model_config.get("feature_order", [])),
        "target_accuracy_achieved": pause_model_config.get("target_accuracy_achieved", False)
    }

def create_pacing_curve(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create mock pacing curve data for visualization"""
    # This is a simplified version - you can enhance this based on your actual data
    duration = features.get("total_duration", 60)
    points = min(int(duration), 60)  # Max 60 points for performance
    
    curve_data = []
    base_wpm = features.get("wpm_mean", 120)
    wpm_std = features.get("wpm_std", 20)
    
    for i in range(points):
        time = i * (duration / points)
        # Add some variation based on actual features
        variation = np.sin(i * 0.3) * wpm_std * 0.5
        wpm = max(50, base_wpm + variation + np.random.normal(0, 5))
        
        curve_data.append({
            "time": round(time, 1),
            "wpm": round(wpm, 1),
            "upper": 150,  # Ideal upper bound
            "lower": 100   # Ideal lower bound
        })
    
    return curve_data

def create_pause_timeline(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create pause timeline data"""
    timeline_data = []
    
    # Extract pause information including excessive pauses
    short_count = features.get("short_count", 0)
    med_count = features.get("med_count", 0)
    long_count = features.get("long_count", 0)
    excessive_count = features.get("excessive_count", 0)
    total_duration = features.get("total_duration", 60)
    
    # Create timeline points including excessive pauses
    pause_types = (
        ["short"] * short_count + 
        ["medium"] * med_count + 
        ["long"] * long_count +
        ["excessive"] * excessive_count
    )
    
    for i, pause_type in enumerate(pause_types[:20]):  # Limit to 20 points
        timeline_data.append({
            "time": round((i + 1) * (total_duration / len(pause_types)), 1),
            "type": pause_type,
            "duration": {
                "short": 0.5,
                "medium": 1.2,
                "long": 2.5,
                "excessive": 6.0  # Representative duration for excessive pauses
            }.get(pause_type, 1.0),
            "reason": {
                "short": "Natural flow",
                "medium": "Transition",
                "long": "Emphasis",
                "excessive": "Problematic"
            }.get(pause_type, "Unknown")
        })
    
    return timeline_data

def create_pause_distribution(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create pause type distribution data with EXCESSIVE_PAUSE detection"""
    # Get actual excessive pauses count from features
    excessive_count = features.get("excessive_count", 0)
    
    total_pauses = features.get("short_count", 0) + features.get("med_count", 0) + features.get("long_count", 0) + excessive_count
    
    return [
        {
            "type": "Excessive Pauses",
            "count": excessive_count,
            "percentage": round((excessive_count / max(1, total_pauses)) * 100, 1),
            "color": "#ff0000",
            "threshold": ">5.0s"
        },
        {
            "type": "Long Pauses",
            "count": features.get("long_count", 0) - excessive_count,
            "percentage": round(((features.get("long_count", 0) - excessive_count) / max(1, total_pauses)) * 100, 1),
            "color": "#ff6600",
            "threshold": "2.5-5.0s"
        },
        {
            "type": "Medium Pauses",
            "count": features.get("med_count", 0),
            "percentage": round((features.get("med_count", 0) / max(1, total_pauses)) * 100, 1),
            "color": "#ffaa00",
            "threshold": "1.0-2.5s"
        },
        {
            "type": "Short Pauses",
            "count": features.get("short_count", 0),
            "percentage": round((features.get("short_count", 0) / max(1, total_pauses)) * 100, 1),
            "color": "#00ff00",
            "threshold": "0.3-1.0s"
        }
    ]

def predict_pause_analysis(features: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction using the trained pause analysis model"""
    if pause_model is None:
        return {
            "prediction": "Model not available",
            "confidence": 0.0,
            "probabilities": {},
            "suggestions": ["Model not loaded. Please train the pause analysis model first."]
        }
    
    try:
        # Prepare features for prediction
        feature_cols = pause_model_config.get("feature_order", [])
        X = np.array([features.get(col, 0.0) for col in feature_cols]).reshape(1, -1)
        
        # Make prediction
        prediction = pause_model.predict(X)[0]
        probabilities = pause_model.predict_proba(X)[0]
        
        # Calculate confidence with better handling
        max_prob = float(np.max(probabilities))
        # Ensure confidence is reasonable (not too low due to model uncertainty)
        confidence = max(0.3, min(0.95, max_prob))  # Clamp between 30% and 95%
        
        # Get class probabilities
        class_names = pause_model.classes_
        prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        
        # Generate suggestions based on actual features
        suggestions = generate_improved_suggestions(features)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "suggestions": suggestions[:5]  # Top 5 suggestions
        }
        
    except Exception as e:
        logger.error(f"Error in pause prediction: {e}")
        return {
            "prediction": "Error in analysis",
            "confidence": 0.0,
            "probabilities": {},
            "suggestions": [f"Analysis error: {str(e)}"]
        }

def generate_improved_suggestions(features: Dict[str, Any]) -> List[str]:
    """Generate improved suggestions based on actual feature values"""
    suggestions = []
    
    # Pause ratio analysis
    pause_ratio = features.get("pause_ratio", 0)
    if pause_ratio > 0.15:
        suggestions.append("🚨 CRITICAL: Reduce total pause time significantly. Target pause ratio 8-12% (Toastmasters standard).")
    elif pause_ratio > 0.12:
        suggestions.append("⚠️ Reduce pause time. Aim for 8-12% pause ratio for natural flow.")
    elif pause_ratio < 0.08:
        suggestions.append("💡 Consider adding more strategic pauses. Optimal range is 8-12% for audience engagement.")
    
    # Excessive pauses analysis
    excessive_count = features.get("excessive_count", 0)
    if excessive_count > 2:
        suggestions.append("🚨 CRITICAL: Too many excessive pauses (>5s). Eliminate all pauses longer than 5 seconds.")
    elif excessive_count > 0:
        suggestions.append("⚠️ Reduce excessive pauses. No pause should exceed 5 seconds.")
    
    # Long pauses analysis
    long_count = features.get("long_count", 0)
    if long_count > 6:
        suggestions.append("⚠️ Too many long pauses (2.5-5s). Limit to <4 long pauses per presentation.")
    elif long_count > 4:
        suggestions.append("💡 Consider reducing long pauses. Use transitions or signposting to avoid gaps.")
    
    # WPM consistency analysis
    wpm_cv = features.get("wpm_cv", 0)
    if wpm_cv > 0.4:
        suggestions.append("🎯 CRITICAL: Very inconsistent pace. Practice with metronome at steady WPM for 15 minutes daily.")
    elif wpm_cv > 0.25:
        suggestions.append("📈 Improve pace consistency. Practice speaking at steady rate to reduce variation.")
    
    # Pause pattern regularity
    pattern_regularity = features.get("pause_pattern_regularity", 0)
    if pattern_regularity < 0.4:
        suggestions.append("🎵 RHYTHM: Work on consistent pause timing. Practice with metronome for rhythm consistency.")
    
    # Speech continuity
    speech_continuity = features.get("speech_continuity", 0)
    if speech_continuity < 0.6:
        suggestions.append("🔄 FLOW: Improve speech continuity. Reduce unnecessary pauses and use transition phrases.")
    
    # If no specific issues found, provide general positive feedback
    if not suggestions:
        suggestions.append("✅ Good pace management! Continue practicing to maintain consistency.")
    
    return suggestions

def create_advanced_metrics(features: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive advanced metrics for display"""
    
    # Calculate Toastmasters compliance score based on actual features
    pause_ratio = features.get("pause_ratio", 0)
    excessive_count = features.get("excessive_count", 0)
    pause_max = features.get("pause_max", 0)
    
    # Toastmasters score calculation
    toastmasters_score = 0
    if 0.08 <= pause_ratio <= 0.12:
        toastmasters_score += 40  # Good pause ratio
    elif pause_ratio < 0.08:
        toastmasters_score += pause_ratio * 500  # Partial credit
    else:
        toastmasters_score += max(0, 40 - (pause_ratio - 0.12) * 200)  # Penalty for too much pausing
    
    if pause_max <= 5.0:
        toastmasters_score += 30  # No excessive pauses
    else:
        toastmasters_score += max(0, 30 - (pause_max - 5) * 5)  # Penalty for long pauses
    
    if excessive_count == 0:
        toastmasters_score += 30  # No excessive pauses
    else:
        toastmasters_score += max(0, 30 - excessive_count * 10)  # Penalty for excessive pauses
    
    # Calculate contextual score
    contextual_score = features.get("contextual_pause_score", 0)
    if contextual_score == 0:
        # Calculate based on pause distribution
        short_count = features.get("short_count", 0)
        med_count = features.get("med_count", 0)
        long_count = features.get("long_count", 0)
        total_pauses = short_count + med_count + long_count
        if total_pauses > 0:
            contextual_score = (med_count / total_pauses) * 100  # Percentage of medium pauses
        else:
            contextual_score = 50  # Default neutral score
    
    return {
        # Toastmasters & Industry Standards
        "toastmasters_score": min(100, max(0, toastmasters_score)),
        "contextual_score": min(100, max(0, contextual_score)),
        
        # Rhythm & Flow Analysis
        "rhythm_consistency": features.get("pause_rhythm_consistency", 0) * 100,
        "rhythm_regularity": features.get("rhythm_regularity", 0) * 100,
        "rhythm_outliers": features.get("rhythm_outliers", 0),
        "speech_continuity": features.get("speech_continuity", 0) * 100,
        "golden_ratio_pauses": features.get("golden_ratio_pauses", 0) * 100,
        
        # Cognitive & Confidence Analysis
        "confidence_score": max(0, min(100, 100 - (features.get("pause_std", 0) * 10))),  # Based on pause consistency
        "cognitive_load": min(100, excessive_count * 10 + features.get("max_long_streak", 0) * 5),
        "memory_retrieval_pauses": max(0, int(excessive_count * 0.3)),
        "optimal_cognitive_pause_ratio": max(0, min(100, 100 - (excessive_count * 10 + features.get("max_long_streak", 0) * 5))),
        
        # Speaking Efficiency & Performance
        "speaking_efficiency": max(0, min(100, (features.get("wpm_mean", 120) / 150) * 100 - (pause_ratio - 0.10) * 200)),
        "pause_efficiency": features.get("pause_efficiency", 0),
        "pause_pattern_regularity": features.get("pause_pattern_regularity", 0) * 100,
        "pause_spacing_consistency": features.get("pause_spacing_consistency", 0) * 100,
        
        # Advanced Statistical Analysis
        "pause_entropy": features.get("pause_entropy", 0) * 100,
        "pause_autocorrelation": features.get("pause_autocorrelation", 0) * 100,
        "pause_fractal_dimension": features.get("pause_fractal_dimension", 0),
        "pause_spectral_density": features.get("pause_spectral_density", 0),
        "pause_trend_analysis": max(0, min(100, 100 - abs(pause_ratio - 0.10) * 500)),
        "pause_volatility": features.get("pause_volatility", 0),
        
        # Pace Management
        "wpm_consistency": max(0, min(100, (1 - features.get("wpm_cv", 0.5)) * 100)),
        "wpm_stability": max(0, min(100, 100 - features.get("wpm_jerk", 0) * 10)),
        "wpm_acceleration": features.get("wpm_acceleration", 0),
        "gap_clustering": features.get("gap_clustering", 0),
        
        # Transition & Emphasis Analysis
        "transition_pause_count": features.get("short_count", 0),  # Use short pauses as transitions
        "emphasis_pause_count": features.get("long_count", 0),    # Use long pauses as emphasis
        "optimal_transition_ratio": (features.get("short_count", 0) / max(1, features.get("short_count", 0) + features.get("med_count", 0) + features.get("long_count", 0))) * 100,
        "optimal_emphasis_ratio": (features.get("long_count", 0) / max(1, features.get("short_count", 0) + features.get("med_count", 0) + features.get("long_count", 0))) * 100
    }

@app.post("/analyze/")
async def analyze_speech(file: UploadFile = File(...)):
    """Analyze uploaded speech file with integrated pause analysis"""
    
    if pause_model is None:
        raise HTTPException(status_code=503, detail="Pause analysis model not loaded. Please train the model first.")
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV, MP3, M4A, or WebM files.")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing file: {file.filename}")
        
        # Extract features
        features = pause_features_for_file(temp_file_path)
        logger.info("✅ Features extracted successfully")
        
        # Get pause analysis prediction
        pause_analysis = predict_pause_analysis(features)
        logger.info("✅ Pause analysis completed")
        
        # Generate suggestions using improved function
        suggestions = generate_improved_suggestions(features)
        structured_suggestions = generate_actionable_suggestions(features)
        
        # Create visualizations data
        pacing_curve = create_pacing_curve(features)
        pause_timeline = create_pause_timeline(features)
        pause_distribution = create_pause_distribution(features)
        advanced_metrics = create_advanced_metrics(features)
        
        # Calculate basic metrics
        word_count = max(1, int(features.get("total_duration", 60) * features.get("wpm_mean", 120) / 60))
        duration = features.get("total_duration", 0)
        wpm = features.get("wpm_mean", 0)
        consistency_score = max(0, min(100, (1 - features.get("wpm_cv", 0.5)) * 100))
        
        # Determine prediction and feedback from pause analysis
        prediction = pause_analysis.get("prediction", "Analyzing...")
        feedback = pause_analysis.get("suggestions", ["Analysis in progress..."])[0] if pause_analysis.get("suggestions") else "Analysis in progress..."
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        response_data = {
            # Basic metrics
            "wordCount": word_count,
            "duration": duration,
            "wpm": wpm,
            "prediction": prediction,
            "consistencyScore": consistency_score,
            "feedback": feedback,
            
            # Visualization data
            "pacingCurve": pacing_curve,
            "pauseTimeline": pause_timeline,
            "pauseDistribution": pause_distribution,
            
            # Advanced metrics
            "advancedMetrics": advanced_metrics,
            
            # Comprehensive Pause Analysis
            "pauseAnalysis": {
                # AI Model Predictions
                "prediction": pause_analysis.get("prediction", "Unknown"),
                "confidence": pause_analysis.get("confidence", 0.0),
                "probabilities": pause_analysis.get("probabilities", {}),
                "suggestions": pause_analysis.get("suggestions", []),
                
                # Raw Pause Metrics
                "shortPauses": features.get("short_count", 0),
                "mediumPauses": features.get("med_count", 0),
                "longPauses": features.get("long_count", 0),
                "excessivePauses": features.get("excessive_count", 0),
                "totalPauseTime": features.get("total_pause_time", 0),
                "pauseRatio": features.get("pause_ratio", 0) * 100,
                "averagePauseLength": features.get("pause_p50", 0),
                "pauseStd": features.get("pause_std", 0),
                "pauseMax": features.get("pause_max", 0),
                "pauseMin": features.get("pause_min", 0),
                "pauseP90": features.get("pause_p90", 0),
                "pauseP95": features.get("pause_p95", 0),
                "maxLongStreak": features.get("max_long_streak", 0),
                "pauseEfficiency": features.get("pause_efficiency", 0),
                "pausePatternRegularity": features.get("pause_pattern_regularity", 0),
                "pauseSpacingConsistency": features.get("pause_spacing_consistency", 0)
            },
            
            # Comprehensive Voice Quality Analysis
            "voiceQuality": {
                "pitch": {
                    "mean": features.get("pitch_mean", 0),
                    "std": features.get("pitch_std", 0),
                    "min": features.get("pitch_min", 0),
                    "max": features.get("pitch_max", 0),
                    "range": features.get("pitch_max", 0) - features.get("pitch_min", 0)
                },

                "jitter": {
                    "local": features.get("jitter_local", 0),
                    "rap": features.get("jitter_rap", 0),
                    "ppq5": features.get("jitter_ppq5", 0)
                },
                "shimmer": {
                    "local": features.get("shimmer_local", 0),
                    "apq3": features.get("shimmer_apq3", 0),
                    "apq5": features.get("shimmer_apq5", 0)
                },
                "hnr": features.get("hnr_mean", 0),
                "formants": {
                    "f1": features.get("f1_mean", 0),
                    "f2": features.get("f2_mean", 0),
                    "f3": features.get("f3_mean", 0)
                },
                "spectral": {
                    "centroid": features.get("spectral_centroid", 0),
                    "rolloff": features.get("spectral_rolloff", 0),
                    "bandwidth": features.get("spectral_bandwidth", 0),
                    "zeroCrossingRate": features.get("zero_crossing_rate", 0)
                },
                "tempo": features.get("tempo", 0)
            },
            
            # Suggestions
            "suggestions": suggestions[:5],  # Top 5 suggestions
            "structuredSuggestions": structured_suggestions,
            
            # Raw features for debugging
            "rawFeatures": {k: v for k, v in features.items() if isinstance(v, (int, float))}
        }
        
        logger.info("✅ Analysis completed successfully")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/pause-analysis/")
async def analyze_pause_only(file: UploadFile = File(...)):
    """Analyze only pause patterns using the trained model"""
    
    if pause_model is None:
        raise HTTPException(status_code=503, detail="Pause analysis model not loaded. Please train the model first.")
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV, MP3, M4A, or WebM files.")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing pause analysis for: {file.filename}")
        
        # Extract features
        features = pause_features_for_file(temp_file_path)
        logger.info("✅ Features extracted successfully")
        
        # Get pause analysis prediction
        pause_analysis = predict_pause_analysis(features)
        logger.info("✅ Pause analysis completed")
        
        # Generate suggestions using improved function
        suggestions = generate_improved_suggestions(features)
        structured_suggestions = generate_actionable_suggestions(features)
        
        # Create visualizations data
        pause_timeline = create_pause_timeline(features)
        pause_distribution = create_pause_distribution(features)
        advanced_metrics = create_advanced_metrics(features)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return {
            "pauseAnalysis": {
                "prediction": pause_analysis.get("prediction", "Unknown"),
                "confidence": pause_analysis.get("confidence", 0.0),
                "probabilities": pause_analysis.get("probabilities", {}),
                "suggestions": pause_analysis.get("suggestions", []),
                "shortPauses": features.get("short_count", 0),
                "mediumPauses": features.get("med_count", 0),
                "longPauses": features.get("long_count", 0),
                "excessivePauses": features.get("excessive_count", 0),
                "totalPauseTime": features.get("total_pause_time", 0),
                "pauseRatio": features.get("pause_ratio", 0) * 100,
                "averagePauseLength": features.get("pause_p50", 0),
                "pauseStd": features.get("pause_std", 0),
                "pauseMax": features.get("pause_max", 0),
                "pauseMin": features.get("pause_min", 0),
                "pauseP90": features.get("pause_p90", 0),
                "pauseP95": features.get("pause_p95", 0),
                "maxLongStreak": features.get("max_long_streak", 0),
                "pauseEfficiency": features.get("pause_efficiency", 0),
                "pausePatternRegularity": features.get("pause_pattern_regularity", 0),
                "pauseSpacingConsistency": features.get("pause_spacing_consistency", 0)
            },
            "pauseTimeline": pause_timeline,
            "pauseDistribution": pause_distribution,
            "advancedMetrics": advanced_metrics,
            "suggestions": suggestions,
            "structuredSuggestions": structured_suggestions
        }
        
    except Exception as e:
        logger.error(f"❌ Error in pause analysis: {e}")
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Pause analysis failed: {str(e)}")

@app.post("/suggestions/")
async def get_suggestions(file: UploadFile = File(...)):
    """Get detailed suggestions for uploaded speech file"""
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract features
        features = pause_features_for_file(temp_file_path)
        
        # Generate comprehensive suggestions
        suggestions = suggest_from_features(features)
        structured_suggestions = generate_actionable_suggestions(features)
        personalized_feedback = generate_personalized_feedback(features)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "basicSuggestions": suggestions,
            "structuredSuggestions": structured_suggestions,
            "personalizedFeedback": personalized_feedback,
            "improvementAreas": [
                area for area in [
                    "Pause Management" if features.get("pause_ratio", 0) > 0.12 else None,
                    "Pace Consistency" if features.get("wpm_cv", 0) > 0.25 else None,
                    "Rhythm Control" if features.get("rhythm_outliers", 0) >= 3 else None,
                    "Voice Quality" if features.get("jitter_local", 0) > 0.02 else None
                ] if area is not None
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating suggestions: {e}")
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Suggestion generation failed: {str(e)}")

@app.post("/real-time-feedback/")
async def get_real_time_feedback(file: UploadFile = File(...)):
    """Get real-time feedback for presentation practice"""
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract features and get prediction
        features = pause_features_for_file(temp_file_path)
        prediction = predict_file(temp_file_path)
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="Prediction failed. Please ensure model is trained.")
        
        # Generate real-time feedback
        real_time_feedback = generate_real_time_feedback(prediction)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "realTimeFeedback": real_time_feedback,
            "prediction": {
                "label": prediction.label,
                "confidence": prediction.confidence,
                "probabilities": prediction.probs
            },
            "improvementAreas": prediction.improvement_areas,
            "keyMetrics": {
                "pauseRatio": features.get("pause_ratio", 0),
                "wpmConsistency": 1 - features.get("wpm_cv", 0.5),
                "toastmastersScore": features.get("toastmasters_compliance_score", 0),
                "speechContinuity": features.get("speech_continuity", 1.0)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating real-time feedback: {e}")
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Real-time feedback generation failed: {str(e)}")

@app.post("/comprehensive-report/")
async def get_comprehensive_report(file: UploadFile = File(...)):
    """Get comprehensive analysis report with detailed technical analysis"""
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract features
        features = pause_features_for_file(temp_file_path)
        
        # Generate comprehensive report
        comprehensive_report = generate_comprehensive_report(features)
        
        # Get improvement areas
        improvement_areas = identify_improvement_areas(features)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "comprehensiveReport": comprehensive_report,
            "improvementAreas": improvement_areas,
            "technicalMetrics": {
                "advancedFeatures": {
                    "contextualPauseScore": features.get("contextual_pause_score", 0),
                    "pauseRhythmConsistency": features.get("pause_rhythm_consistency", 0),
                    "cognitivePauseScore": features.get("cognitive_pause_score", 0),
                    "pauseEntropy": features.get("pause_entropy", 0),
                    "pauseVolatility": features.get("pause_volatility", 0)
                },
                "voiceQuality": {
                    "jitterLocal": features.get("jitter_local", 0),
                    "shimmerLocal": features.get("shimmer_local", 0),
                    "pitchStd": features.get("pitch_std", 0),
                    "hnrMean": features.get("hnr_mean", 0)
                },
                "paceMetrics": {
                    "wpmCv": features.get("wpm_cv", 0),
                    "wpmJerk": features.get("wpm_jerk", 0),
                    "rhythmOutliers": features.get("rhythm_outliers", 0),
                    "speakingEfficiency": features.get("speaking_efficiency", 0)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating comprehensive report: {e}")
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Comprehensive report generation failed: {str(e)}")

@app.post("/improvement-areas/")
async def get_improvement_areas(file: UploadFile = File(...)):
    """Get specific improvement areas for the speech"""
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract features
        features = pause_features_for_file(temp_file_path)
        
        # Get improvement areas
        improvement_areas = identify_improvement_areas(features)
        
        # Generate targeted suggestions for each area
        area_suggestions = {}
        for area in improvement_areas:
            if area == "Pause Management":
                area_suggestions[area] = [
                    "Practice 1.5-2.0s pauses between topics",
                    "Use 0.5-1.0s pauses for emphasis",
                    "Reduce unnecessary long pauses"
                ]
            elif area == "Pace Consistency":
                area_suggestions[area] = [
                    "Practice with metronome at steady WPM",
                    "Record and analyze pace variations",
                    "Focus on smooth transitions"
                ]
            elif area == "Rhythm Control":
                area_suggestions[area] = [
                    "Practice consistent segment timing",
                    "Use rhythmic patterns: short-short-long",
                    "Work on speech flow continuity"
                ]
            elif area == "Voice Quality":
                area_suggestions[area] = [
                    "Practice vocal exercises for stability",
                    "Work on consistent pitch control",
                    "Focus on breathing techniques"
                ]
            else:
                area_suggestions[area] = [
                    "Focus on overall speech improvement",
                    "Practice regularly with recordings",
                    "Seek professional feedback"
                ]
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "improvementAreas": improvement_areas,
            "areaSuggestions": area_suggestions,
            "priorityLevel": "High" if len(improvement_areas) > 3 else "Medium" if len(improvement_areas) > 1 else "Low",
            "totalAreas": len(improvement_areas)
        }
        
    except Exception as e:
        logger.error(f"❌ Error identifying improvement areas: {e}")
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Improvement areas analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pause_model_loaded": pause_model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

if __name__ == "__main__":
    # Check if pause model exists before starting
    if not os.path.exists(PAUSE_MODEL_PATH):
        print("❌ Pause analysis model file not found!")
        print("Please run the model training first:")
        print("1. Run feature extraction to create enhanced_pause_features.csv")
        print("2. Run model training to create enhanced_pause_model.joblib")
        print("3. Then start this API server")
    else:
        print("🚀 Starting SpeakCraft API server with pause analysis...")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
