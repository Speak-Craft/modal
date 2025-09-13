from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
from joblib import load
import os
import noisereduce as nr
import soundfile as sf 
import uvicorn
import numpy as np
import pandas as pd
import librosa
import tempfile
import joblib
import json
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

# Import activity endpoints
from activity_endpoints import router as activity_router

app = FastAPI()

# CORS for the React frontend pornt no 8000 is connected 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include activity router
app.include_router(activity_router)

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

# Load the trained model and configuration
try:
    model = joblib.load("enhanced_pause_model.joblib")
    with open("enhanced_pause_features.json", "r") as f:
        model_config = json.load(f)
    feature_order = model_config["feature_order"]
except Exception as e:
    print(f"Warning: Could not load model files: {e}")
    model = None
    feature_order = []

def safe_generate_feedback(func, *args):
    """Safely generate feedback with error handling"""
    try:
        result = func(*args)
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return f"Feedback data: {json.dumps(result, indent=2)}"
        else:
            return f"Generated feedback: {str(result)}"
    except Exception as e:
        print(f"Feedback generation error: {e}")
        import traceback
        traceback.print_exc()
        return f"Feedback generation failed: {str(e)}"

def generate_pause_specific_recommendations(features):
    """Generate pause-specific recommendations for Pause Analysis tab"""
    recommendations = {
        "High": [],
        "Medium": [],
        "Low": []
    }
    
    # High Priority Pause Issues
    if features.get('pause_ratio', 0) > 0.15:
        recommendations["High"].append({
            "issue": "Excessive Pause Time",
            "action": "Reduce total pause time to 8-12% of speech duration",
            "current": f"{features.get('pause_ratio', 0)*100:.1f}%",
            "target": "8-12%",
            "impact": "Critical - affects audience engagement"
        })
    
    if features.get('excessive_count', 0) > 0:
        recommendations["High"].append({
            "issue": "Excessive Pauses Detected",
            "action": "Eliminate all pauses longer than 5 seconds",
            "current": f"{features.get('excessive_count', 0)}",
            "target": "0",
            "impact": "Critical - creates awkward silences"
        })
    
    if features.get('long_count', 0) > 6:
        recommendations["High"].append({
            "issue": "Too Many Long Pauses",
            "action": "Use transition phrases and signposting to bridge gaps",
            "current": f"{features.get('long_count', 0)}",
            "target": "<4",
            "impact": "High - affects speech flow"
        })
    
    if features.get('pause_rhythm_consistency', 0) < 0.4:
        recommendations["High"].append({
            "issue": "Inconsistent Pause Rhythm",
            "action": "Practice with metronome for consistent pause timing",
            "current": f"{features.get('pause_rhythm_consistency', 0)*100:.1f}%",
            "target": ">70%",
            "impact": "High - affects speech flow"
        })
    
    # Medium Priority Pause Issues
    if 0.12 < features.get('pause_ratio', 0) <= 0.15:
        recommendations["Medium"].append({
            "issue": "High Pause Time",
            "action": "Aim for 8-12% pause ratio for natural flow",
            "current": f"{features.get('pause_ratio', 0)*100:.1f}%",
            "target": "8-12%",
            "impact": "Medium - affects pacing"
        })
    
    if 4 < features.get('long_count', 0) <= 6:
        recommendations["Medium"].append({
            "issue": "Moderate Long Pauses",
            "action": "Consider reducing long pauses or adding content",
            "current": f"{features.get('long_count', 0)}",
            "target": "<4",
            "impact": "Medium - affects pacing"
        })
    
    if 0.4 <= features.get('pause_rhythm_consistency', 0) < 0.7:
        recommendations["Medium"].append({
            "issue": "Moderate Rhythm Issues",
            "action": "Work on consistent pause patterns",
            "current": f"{features.get('pause_rhythm_consistency', 0)*100:.1f}%",
            "target": ">70%",
            "impact": "Medium - affects naturalness"
        })
    
    if features.get('pause_efficiency', 0) > 3.0:
        recommendations["Medium"].append({
            "issue": "Inefficient Pause Length",
            "action": "Aim for 0.5-2.0 second pauses for natural flow",
            "current": f"{features.get('pause_efficiency', 0):.1f}s avg",
            "target": "0.5-2.0s",
            "impact": "Medium - affects pacing"
        })
    
    # Low Priority Pause Issues
    if features.get('pause_ratio', 0) < 0.08:
        recommendations["Low"].append({
            "issue": "Insufficient Pause Time",
            "action": "Add strategic pauses for emphasis and audience processing",
            "current": f"{features.get('pause_ratio', 0)*100:.1f}%",
            "target": "8-12%",
            "impact": "Low - speech may feel rushed"
        })
    
    if features.get('pause_pattern_regularity', 0) < 0.6:
        recommendations["Low"].append({
            "issue": "Irregular Pause Patterns",
            "action": "Develop consistent pause patterns for better flow",
            "current": f"{features.get('pause_pattern_regularity', 0)*100:.1f}%",
            "target": ">80%",
            "impact": "Low - affects predictability"
        })
    
    if features.get('contextual_pause_score', 0) < 0.0:
        recommendations["Low"].append({
            "issue": "Poor Pause Context",
            "action": "Use 1.5-2.0s for transitions, 0.5-1.0s for emphasis",
            "current": f"{features.get('contextual_pause_score', 0):.2f}",
            "target": ">0.5",
            "impact": "Low - affects speech effectiveness"
        })
    
    return recommendations

def generate_comprehensive_improvements(features):
    """Generate comprehensive improvements for AI Insights tab using feature_extraction2.py logic"""
    improvements = {
        "High": [],
        "Medium": [],
        "Low": [],
        "Practice": [],
        "Immediate": []
    }
    
    # Use the same logic as suggest_from_features but convert to structured format
    wpm_cv = features.get('wpm_cv', 0)
    wpm_delta_std = features.get('wpm_delta_std', 0)
    wpm_jerk = features.get('wpm_jerk', 0)
    rhythm_outliers = features.get('rhythm_outliers', 0)
    speech_continuity = features.get('speech_continuity', 0)
    confidence_score = features.get('confidence_score', 0)
    speaking_efficiency = features.get('speaking_efficiency', 0)
    jitter_local = features.get('jitter_local', 0)
    shimmer_local = features.get('shimmer_local', 0)
    pitch_std = features.get('pitch_std', 0)
    cognitive_score = features.get('cognitive_pause_score', 0)
    golden_ratio = features.get('golden_ratio_pauses', 0)
    
    # High Priority Issues - Critical speech problems
    if wpm_cv > 0.4:
        improvements["High"].append({
            "issue": "Very Inconsistent Speech Rate",
            "action": "Practice with metronome at steady WPM for 15 minutes daily",
            "current": f"{wpm_cv:.2f} (CV)",
            "target": "<0.25",
            "impact": "Critical - makes speech hard to follow"
        })
    
    if wpm_delta_std > 8:
        improvements["High"].append({
            "issue": "Sudden Pace Changes",
            "action": "Practice smooth transitions between sections",
            "current": f"{wpm_delta_std:.1f}",
            "target": "<5",
            "impact": "High - creates jarring experience"
        })
    
    if wpm_jerk > 10:
        improvements["High"].append({
            "issue": "Very Jerky Pace Changes",
            "action": "Practice gradual speed adjustments",
            "current": f"{wpm_jerk:.1f}",
            "target": "<5",
            "impact": "High - creates uncomfortable listening experience"
        })
    
    if rhythm_outliers >= 6:
        improvements["High"].append({
            "issue": "Inconsistent Rhythm",
            "action": "Practice consistent segment lengths",
            "current": f"{rhythm_outliers}",
            "target": "<3",
            "impact": "High - affects speech structure"
        })
    
    if speech_continuity < 0.6:
        improvements["High"].append({
            "issue": "Fragmented Speech",
            "action": "Reduce unnecessary pauses between thoughts",
            "current": f"{speech_continuity*100:.1f}%",
            "target": ">80%",
            "impact": "High - affects message clarity"
        })
    
    if confidence_score < 0.6:
        improvements["High"].append({
            "issue": "Low Confidence Indicators",
            "action": "Practice content to minimize thinking pauses during delivery",
            "current": f"{confidence_score*100:.1f}%",
            "target": ">80%",
            "impact": "High - affects overall presentation"
        })
    
    if speaking_efficiency < 0.6:
        improvements["High"].append({
            "issue": "Very Low Speaking Efficiency",
            "action": "Reduce unnecessary pauses and filler words significantly",
            "current": f"{speaking_efficiency*100:.1f}%",
            "target": ">80%",
            "impact": "High - affects presentation quality"
        })
    
    # Medium Priority Issues - Significant improvements needed
    if 0.25 < wpm_cv <= 0.4:
        improvements["Medium"].append({
            "issue": "Moderate Pace Inconsistency",
            "action": "Aim for ±15% WPM variation",
            "current": f"{wpm_cv:.2f} (CV)",
            "target": "<0.25",
            "impact": "Medium - affects speech flow"
        })
    
    if 5 < wpm_delta_std <= 8:
        improvements["Medium"].append({
            "issue": "Abrupt Pace Changes",
            "action": "Smooth out transitions between ideas",
            "current": f"{wpm_delta_std:.1f}",
            "target": "<5",
            "impact": "Medium - affects naturalness"
        })
    
    if 5 < wpm_jerk <= 10:
        improvements["Medium"].append({
            "issue": "Jerky Pace Changes",
            "action": "Smooth out acceleration and deceleration",
            "current": f"{wpm_jerk:.1f}",
            "target": "<5",
            "impact": "Medium - affects speech quality"
        })
    
    if 3 <= rhythm_outliers < 6:
        improvements["Medium"].append({
            "issue": "Moderate Rhythm Issues",
            "action": "Aim for consistent speech segment durations",
            "current": f"{rhythm_outliers}",
            "target": "<3",
            "impact": "Medium - affects speech flow"
        })
    
    if 0.6 <= speech_continuity < 0.8:
        improvements["Medium"].append({
            "issue": "Moderate Speech Fragmentation",
            "action": "Connect related ideas more seamlessly",
            "current": f"{speech_continuity*100:.1f}%",
            "target": ">80%",
            "impact": "Medium - affects flow"
        })
    
    if 0.6 <= speaking_efficiency < 0.7:
        improvements["Medium"].append({
            "issue": "Low Speaking Efficiency",
            "action": "Reduce unnecessary pauses and filler words",
            "current": f"{speaking_efficiency*100:.1f}%",
            "target": ">80%",
            "impact": "Medium - affects presentation quality"
        })
    
    if 0.6 <= confidence_score < 0.7:
        improvements["Medium"].append({
            "issue": "Moderate Confidence Issues",
            "action": "Practice content familiarity and breathing techniques",
            "current": f"{confidence_score*100:.1f}%",
            "target": ">80%",
            "impact": "Medium - affects presentation confidence"
        })
    
    # Voice Quality Issues
    if jitter_local > 0.02:
        improvements["Medium"].append({
            "issue": "Voice Instability",
            "action": "Reduce vocal jitter for clearer speech",
            "current": f"{jitter_local:.3f}",
            "target": "<0.02",
            "impact": "Medium - affects voice clarity"
        })
    
    if shimmer_local > 0.1:
        improvements["Medium"].append({
            "issue": "Voice Clarity Issues",
            "action": "Work on reducing vocal shimmer for better articulation",
            "current": f"{shimmer_local:.3f}",
            "target": "<0.1",
            "impact": "Medium - affects articulation"
        })
    
    # Low Priority Issues - Refinements
    if pitch_std > 50:
        improvements["Low"].append({
            "issue": "Inconsistent Pitch",
            "action": "Maintain more consistent pitch throughout presentation",
            "current": f"{pitch_std:.1f} Hz",
            "target": "<50 Hz",
            "impact": "Low - affects vocal variety"
        })
    
    if cognitive_score < 0.3:
        improvements["Low"].append({
            "issue": "High Cognitive Load",
            "action": "Optimize thinking pauses (1.0-3.0s for complex concepts)",
            "current": f"{cognitive_score:.2f}",
            "target": ">0.5",
            "impact": "Low - affects confidence"
        })
    
    if golden_ratio < 0.3:
        improvements["Low"].append({
            "issue": "Missing Natural Speech Rhythms",
            "action": "Incorporate 1.618x longer pauses at key moments",
            "current": f"{golden_ratio*100:.1f}%",
            "target": ">30%",
            "impact": "Low - enhances naturalness"
        })
    
    # Always add practice exercises based on detected issues
    if wpm_cv > 0.25 or wpm_delta_std > 5 or wpm_jerk > 5:
        improvements["Practice"].append({
            "issue": "Pace Control Training",
            "action": "Practice speaking at consistent WPM for 15 minutes daily with metronome",
            "current": "Inconsistent pace detected",
            "target": "Smooth, consistent pace",
            "impact": "Improves overall speech stability and flow"
        })
    
    if rhythm_outliers >= 3 or speech_continuity < 0.8:
        improvements["Practice"].append({
            "issue": "Rhythm and Flow Training",
            "action": "Practice speaking in 10-second segments with consistent timing",
            "current": "Irregular rhythm detected",
            "target": "Regular, flowing rhythm",
            "impact": "Enhances speech flow and naturalness"
        })
    
    if confidence_score < 0.8:
        improvements["Practice"].append({
            "issue": "Confidence Building",
            "action": "Practice content familiarity and breathing exercises daily",
            "current": "Low confidence indicators",
            "target": "High confidence delivery",
            "impact": "Reduces hesitation and improves delivery"
        })
    
    # Voice quality practice
    if jitter_local > 0.015 or shimmer_local > 0.08:
        improvements["Practice"].append({
            "issue": "Voice Quality Training",
            "action": "Practice vocal exercises and breathing techniques",
            "current": "Voice quality issues detected",
            "target": "Clear, stable voice",
            "impact": "Improves voice clarity and stability"
        })
    
    # Immediate Actions - Always provide actionable steps
    improvements["Immediate"].append({
        "issue": "Record Baseline Analysis",
        "action": "Record yourself speaking for 2 minutes and analyze the results",
        "current": "Not done",
        "target": "Baseline measurement",
        "impact": "Identifies current speech patterns and issues"
    })
    
    if wpm_cv > 0.25:
        improvements["Immediate"].append({
            "issue": "Metronome Practice",
            "action": "Download a metronome app and practice speaking at 140 WPM for 5 minutes",
            "current": "Not started",
            "target": "Consistent pace",
            "impact": "Immediate pace improvement"
        })
    
    if confidence_score < 0.8:
        improvements["Immediate"].append({
            "issue": "Content Familiarity",
            "action": "Practice your speech content 3 times before recording",
            "current": "Not done",
            "target": "Familiar content",
            "impact": "Reduces hesitation and improves confidence"
        })
    
    improvements["Immediate"].append({
        "issue": "Breathing Exercises",
        "action": "Practice deep breathing exercises for 5 minutes before speaking",
        "current": "Not done",
        "target": "Better voice control",
        "impact": "Reduces vocal jitter and improves stability"
    })
    
    return improvements

@app.post("/pause-analysis/")
async def analyze_pause_management(file: UploadFile = File(...)):
    """
    Comprehensive pause analysis endpoint that provides all data needed by the frontend
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract features using the enhanced feature extraction
        try:
            features = pause_features_for_file(tmp_file_path)
            print(f"Feature extraction successful. Extracted {len(features)} features")
            print(f"Key features for debugging:")
            print(f"  pause_ratio: {features.get('pause_ratio', 0)}")
            print(f"  wpm_cv: {features.get('wpm_cv', 0)}")
            print(f"  wpm_delta_std: {features.get('wpm_delta_std', 0)}")
            print(f"  wpm_jerk: {features.get('wpm_jerk', 0)}")
            print(f"  rhythm_outliers: {features.get('rhythm_outliers', 0)}")
            print(f"  speech_continuity: {features.get('speech_continuity', 0)}")
            print(f"  confidence_score: {features.get('confidence_score', 0)}")
        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            # Return a basic error response
            return {
                "error": f"Feature extraction failed: {str(e)}",
                "pauseAnalysis": {
                    "prediction": "error",
                    "confidence": 0.0,
                    "probabilities": {},
                    "suggestions": ["Feature extraction failed. Please check your audio file."],
                    "shortPauses": 0,
                    "mediumPauses": 0,
                    "longPauses": 0,
                    "excessivePauses": 0,
                    "totalPauseTime": 0,
                    "pauseRatio": 0,
                    "averagePauseLength": 0,
                    "pauseStd": 0,
                    "pauseMax": 0,
                    "pauseMin": 0,
                    "pauseP90": 0,
                    "pauseP95": 0,
                    "maxLongStreak": 0,
                    "pauseEfficiency": 0,
                    "pausePatternRegularity": 0,
                    "pauseSpacingConsistency": 0
                },
                "pauseTimeline": [],
                "pauseDistribution": [],
                "advancedMetrics": {},
                "suggestions": [],
                "structuredSuggestions": {},
                "enhancedFeedback": "Feature extraction failed. Please try again.",
                "realTimeFeedback": "Feature extraction failed. Please try again.",
                "comprehensiveReport": "Feature extraction failed. Please try again."
            }
        
        # Get AI prediction if model is available
        if model is not None:
            try:
                # Prepare features for prediction
                prediction_data = {}
                for col in feature_order:
                    if col in features:
                        prediction_data[col] = features[col]
                    else:
                        prediction_data[col] = 0.0
                
                # Make prediction
                X = np.array([list(prediction_data.values())]).reshape(1, -1)
                probabilities = model.predict_proba(X)[0]
                predicted_class = model.classes_[np.argmax(probabilities)]
                confidence = float(np.max(probabilities))
                
                # Create probability dictionary
                prob_dict = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}
                
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_class = "analyzing"
                confidence = 0.0
                prob_dict = {}
        else:
            predicted_class = "analyzing"
            confidence = 0.0
            prob_dict = {}
        
        # Generate suggestions using the enhanced suggestion system
        try:
            suggestions = suggest_from_features(features)
            print(f"Generated {len(suggestions)} basic suggestions")
        except Exception as e:
            print(f"Basic suggestion generation error: {e}")
            suggestions = ["Analysis completed successfully"]
        
        try:
            structured_suggestions = generate_actionable_suggestions(features)
            print(f"Generated {len(structured_suggestions)} structured suggestion categories")
        except Exception as e:
            print(f"Structured suggestion generation error: {e}")
            import traceback
            traceback.print_exc()
            structured_suggestions = {
                "critical_issues": [],
                "major_improvements": [],
                "minor_refinements": [],
                "long_term_goals": [],
                "practice_exercises": [],
                "immediate_actions": []
            }
        
        # Generate pause timeline data
        pause_timeline = []
        if 'long_pause_distribution' in features:
            # Create timeline from distribution data
            chunk_size = 30.0  # 30 seconds per chunk
            for i, pause_time in enumerate(features['long_pause_distribution']):
                if pause_time > 0:
                    pause_timeline.append({
                        "time": i * chunk_size,
                        "duration": pause_time,
                        "type": "long" if pause_time > 2.5 else "medium"
                    })
        
        # Generate pause distribution data
        pause_distribution = [
            {
                "type": "Short Pauses",
                "count": features.get('short_count', 0),
                "percentage": (features.get('short_count', 0) / max(1, features.get('short_count', 0) + features.get('med_count', 0) + features.get('long_count', 0) + features.get('excessive_count', 0))) * 100,
                "color": "#00ff00"
            },
            {
                "type": "Medium Pauses", 
                "count": features.get('med_count', 0),
                "percentage": (features.get('med_count', 0) / max(1, features.get('short_count', 0) + features.get('med_count', 0) + features.get('long_count', 0) + features.get('excessive_count', 0))) * 100,
                "color": "#ffaa00"
            },
            {
                "type": "Long Pauses",
                "count": features.get('long_count', 0),
                "percentage": (features.get('long_count', 0) / max(1, features.get('short_count', 0) + features.get('med_count', 0) + features.get('long_count', 0) + features.get('excessive_count', 0))) * 100,
                "color": "#ff6600"
            },
            {
                "type": "Excessive Pauses",
                "count": features.get('excessive_count', 0),
                "percentage": (features.get('excessive_count', 0) / max(1, features.get('short_count', 0) + features.get('med_count', 0) + features.get('long_count', 0) + features.get('excessive_count', 0))) * 100,
                "color": "#ff0000"
            }
        ]
        
        # Calculate advanced metrics for frontend display
        advanced_metrics = {
            # Core pause metrics
            "short_pauses": features.get('short_count', 0),
            "medium_pauses": features.get('med_count', 0),
            "long_pauses": features.get('long_count', 0),
            "excessive_pauses": features.get('excessive_count', 0),
            "total_pause_time": features.get('total_pause_time', 0),
            "pause_ratio": features.get('pause_ratio', 0) * 100,  # Convert to percentage
            "average_pause_length": features.get('pause_p50', 0),
            "pause_std": features.get('pause_std', 0),
            "pause_max": features.get('pause_max', 0),
            "pause_min": features.get('pause_min', 0),
            "pause_p90": features.get('pause_p90', 0),
            "pause_p95": features.get('pause_p95', 0),
            "max_long_streak": features.get('max_long_streak', 0),
            "pause_efficiency": features.get('pause_efficiency', 0),
            "pause_pattern_regularity": features.get('pause_pattern_regularity', 0),
            "pause_spacing_consistency": features.get('pause_spacing_consistency', 0),
            
            # Novel pause features
            "contextual_pause_score": features.get('contextual_pause_score', 0),
            "transition_pause_count": features.get('transition_pause_count', 0),
            "emphasis_pause_count": features.get('emphasis_pause_count', 0),
            "optimal_transition_ratio": features.get('optimal_transition_ratio', 0) * 100,
            "optimal_emphasis_ratio": features.get('optimal_emphasis_ratio', 0) * 100,
            "pause_rhythm_consistency": features.get('pause_rhythm_consistency', 0) * 100,
            "golden_ratio_pauses": features.get('golden_ratio_pauses', 0) * 100,
            "pause_entropy": features.get('pause_entropy', 0) * 100,
            "pause_autocorrelation": features.get('pause_autocorrelation', 0) * 100,
            "cognitive_pause_score": features.get('cognitive_pause_score', 0) * 100,
            "memory_retrieval_pauses": features.get('memory_retrieval_pauses', 0),
            "confidence_score": features.get('confidence_score', 0) * 100,
            "optimal_cognitive_pause_ratio": features.get('optimal_cognitive_pause_ratio', 0) * 100,
            "pause_fractal_dimension": features.get('pause_fractal_dimension', 0),
            "pause_spectral_density": features.get('pause_spectral_density', 0),
            "pause_trend_analysis": features.get('pause_trend_analysis', 0) * 100,
            "pause_volatility": features.get('pause_volatility', 0),
            "toastmasters_compliance_score": features.get('toastmasters_compliance_score', 0) * 100,
            
            # Pace management metrics
            "wpm_mean": features.get('wpm_mean', 0),
            "wpm_std": features.get('wpm_std', 0),
            "wpm_cv": features.get('wpm_cv', 0) * 100,
            "wpm_delta_std": features.get('wpm_delta_std', 0),
            "wpm_jerk": features.get('wpm_jerk', 0),
            "wpm_acceleration": features.get('wpm_acceleration', 0),
            "wpm_consistency": (1 - features.get('wpm_cv', 0)) * 100,
            "wpm_stability": (1 - features.get('wpm_delta_std', 0) / 10) * 100,
            
            # Rhythm and flow metrics
            "rhythm_outliers": features.get('rhythm_outliers', 0),
            "rhythm_regularity": features.get('rhythm_regularity', 0) * 100,
            "speech_continuity": features.get('speech_continuity', 0) * 100,
            "speaking_efficiency": features.get('speaking_efficiency', 0) * 100,
            "gap_clustering": features.get('gap_clustering', 0),
            
            # Voice quality metrics
            "pitch_mean": features.get('pitch_mean', 0),
            "pitch_std": features.get('pitch_std', 0),
            "jitter_local": features.get('jitter_local', 0),
            "shimmer_local": features.get('shimmer_local', 0),
            "hnr_mean": features.get('hnr_mean', 0),
            "f1_mean": features.get('f1_mean', 0),
            "f2_mean": features.get('f2_mean', 0),
            "f3_mean": features.get('f3_mean', 0),
            
            # Additional metrics for radar chart
            "rhythm_consistency": features.get('pause_rhythm_consistency', 0) * 100,
            "cognitive_load": features.get('cognitive_pause_score', 0) * 100,
            "contextual_score": features.get('contextual_pause_score', 0) * 100,
        }
        
        # Generate separate recommendations for each tab
        pause_recommendations_dict = generate_pause_specific_recommendations(features)
        comprehensive_improvements_dict = generate_comprehensive_improvements(features)
        
        # Convert to flat lists for frontend compatibility
        pause_recommendations = []
        for priority, items in pause_recommendations_dict.items():
            for item in items:
                pause_recommendations.append({
                    "priority": priority,
                    "issue": item["issue"],
                    "action": item["action"],
                    "current": item["current"],
                    "target": item["target"],
                    "target_score": item["target"],
                    "impact": item["impact"]
                })
        
        priority_improvements = []
        for priority, items in comprehensive_improvements_dict.items():
            for item in items:
                priority_improvements.append({
                    "priority": priority,
                    "issue": item["issue"],
                    "action": item["action"],
                    "current": item["current"],
                    "target": item["target"],
                    "target_score": item["target"],
                    "impact": item["impact"]
                })
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Debug: Print the response structure
        response_data = {
            "pauseAnalysis": {
                "prediction": predicted_class,
                "confidence": features.get('confidence_score', 0.5),  # Use actual confidence score from features
                "probabilities": prob_dict,
                "suggestions": suggestions,
                "shortPauses": features.get('short_count', 0),
                "mediumPauses": features.get('med_count', 0),
                "longPauses": features.get('long_count', 0),
                "excessivePauses": features.get('excessive_count', 0),
                "totalPauseTime": features.get('total_pause_time', 0),
                "pauseRatio": features.get('pause_ratio', 0) * 100,
                "averagePauseLength": features.get('pause_p50', 0),
                "pauseStd": features.get('pause_std', 0),
                "pauseMax": features.get('pause_max', 0),
                "pauseMin": features.get('pause_min', 0),
                "pauseP90": features.get('pause_p90', 0),
                "pauseP95": features.get('pause_p95', 0),
                "maxLongStreak": features.get('max_long_streak', 0),
                "pauseEfficiency": features.get('pause_efficiency', 0),
                "pausePatternRegularity": features.get('pause_pattern_regularity', 0),
                "pauseSpacingConsistency": features.get('pause_spacing_consistency', 0)
            },
            "pauseTimeline": pause_timeline,
            "pauseDistribution": pause_distribution,
            "advancedMetrics": advanced_metrics,
            "suggestions": pause_recommendations,  # Pause-specific recommendations for Pause Analysis tab
            "priorityImprovements": priority_improvements,  # Comprehensive improvements for AI Insights tab
            "structuredSuggestions": structured_suggestions,
            "basicSuggestions": suggestions,  # From suggest_from_features
            "enhancedFeedback": safe_generate_feedback(generate_personalized_feedback, features),
            "realTimeFeedback": safe_generate_feedback(generate_real_time_feedback, EnhancedPrediction(
                label=predicted_class,
                probs=prob_dict,
                suggestions=suggestions,
                features=features,
                confidence=features.get('confidence_score', 0.5),
                improvement_areas=identify_improvement_areas(features)
            )),
            "comprehensiveReport": safe_generate_feedback(generate_comprehensive_report, features),
            # Raw features for frontend calculations
            "rawFeatures": features
        }
        
        # Ensure we have at least one suggestion to prevent frontend errors
        if not pause_recommendations:
            pause_recommendations = [{
                "priority": "Info",
                "issue": "Pause Analysis Complete",
                "action": "Review your pause metrics above",
                "current": "Analysis finished",
                "target": "Continue practicing pause timing",
                "target_score": "Continue practicing pause timing",
                "impact": "Keep improving your pause management"
            }]
        
        if not priority_improvements:
            priority_improvements = [{
                "priority": "Info",
                "issue": "Analysis Complete",
                "action": "Review your speech metrics above",
                "current": "Analysis finished",
                "target": "Continue practicing",
                "target_score": "Continue practicing",
                "impact": "Keep improving your speech"
            }]
        
        # Debug: Print first suggestion to check structure
        if pause_recommendations:
            print(f"First pause recommendation: {pause_recommendations[0]}")
        if priority_improvements:
            print(f"First priority improvement: {priority_improvements[0]}")
        else:
            print("No priority improvements generated!")
            print(f"Features that should trigger improvements:")
            print(f"  wpm_cv: {features.get('wpm_cv', 0)}")
            print(f"  wpm_delta_std: {features.get('wpm_delta_std', 0)}")
            print(f"  wpm_jerk: {features.get('wpm_jerk', 0)}")
            print(f"  rhythm_outliers: {features.get('rhythm_outliers', 0)}")
            print(f"  speech_continuity: {features.get('speech_continuity', 0)}")
            print(f"  confidence_score: {features.get('confidence_score', 0)}")
        
        # Debug: Check feedback types and content
        print(f"Enhanced feedback type: {type(response_data['enhancedFeedback'])}")
        print(f"Real-time feedback type: {type(response_data['realTimeFeedback'])}")
        print(f"Comprehensive report type: {type(response_data['comprehensiveReport'])}")
        print(f"Priority improvements count: {len(priority_improvements)}")
        print(f"Pause recommendations count: {len(pause_recommendations)}")
        
        # Debug: Show some key feature values
        print(f"Key features for comprehensive improvements:")
        print(f"  wpm_cv: {features.get('wpm_cv', 0)} (threshold: >0.4 for High)")
        print(f"  wpm_delta_std: {features.get('wpm_delta_std', 0)} (threshold: >8 for High)")
        print(f"  wpm_jerk: {features.get('wpm_jerk', 0)} (threshold: >10 for High)")
        print(f"  rhythm_outliers: {features.get('rhythm_outliers', 0)} (threshold: >=6 for High)")
        print(f"  speech_continuity: {features.get('speech_continuity', 0)} (threshold: <0.6 for High)")
        print(f"  confidence_score: {features.get('confidence_score', 0)} (threshold: <0.6 for High)")
        print(f"  speaking_efficiency: {features.get('speaking_efficiency', 0)} (threshold: <0.6 for High)")
        
        return response_data
        
    except Exception as e:
        print(f"Error in pause analysis: {e}")
        return {
            "error": str(e),
            "pauseAnalysis": {
                "prediction": "error",
                "confidence": 0.0,
                "probabilities": {},
                "suggestions": ["Analysis failed. Please try again."],
                "shortPauses": 0,
                "mediumPauses": 0,
                "longPauses": 0,
                "excessivePauses": 0,
                "totalPauseTime": 0,
                "pauseRatio": 0,
                "averagePauseLength": 0,
                "pauseStd": 0,
                "pauseMax": 0,
                "pauseMin": 0,
                "pauseP90": 0,
                "pauseP95": 0,
                "maxLongStreak": 0,
                "pauseEfficiency": 0,
                "pausePatternRegularity": 0,
                "pauseSpacingConsistency": 0
            },
            "pauseTimeline": [],
            "pauseDistribution": [],
            "advancedMetrics": {},
            "suggestions": [],
            "structuredSuggestions": {},
            "enhancedFeedback": "Analysis failed. Please try again.",
            "realTimeFeedback": "Analysis failed. Please try again.",
            "comprehensiveReport": "Analysis failed. Please try again."
        }

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the server is running"""
    return {
        "status": "Server is running",
        "model_loaded": model is not None,
        "feature_order_length": len(feature_order) if feature_order else 0
    }

@app.post("/generate-suggestions/")
async def generate_suggestions_from_frontend_calculations(frontend_features: dict):
    """
    Generate actionable suggestions from frontend-calculated features
    This endpoint takes the frontend-calculated metrics and returns backend-generated suggestions
    """
    try:
        # Convert frontend features to the format expected by backend functions
        features = frontend_features.get('features', {})
        
        # Generate suggestions using backend functions
        try:
            structured_suggestions = generate_actionable_suggestions(features)
            print(f"Generated {len(structured_suggestions)} structured suggestion categories from frontend features")
        except Exception as e:
            print(f"Structured suggestion generation error: {e}")
            import traceback
            traceback.print_exc()
            structured_suggestions = {
                "critical_issues": [],
                "major_improvements": [],
                "minor_refinements": [],
                "practice_exercises": [],
                "immediate_actions": [],
                "long_term_goals": [],
                "toastmasters_tips": [],
                "confidence_builders": []
            }
        
        # Generate pause-specific recommendations
        try:
            pause_recommendations_dict = generate_pause_specific_recommendations(features)
            print(f"Generated pause recommendations from frontend features")
        except Exception as e:
            print(f"Pause recommendation generation error: {e}")
            pause_recommendations_dict = {"High": [], "Medium": [], "Low": []}
        
        # Generate comprehensive improvements
        try:
            comprehensive_improvements_dict = generate_comprehensive_improvements(features)
            print(f"Generated comprehensive improvements from frontend features")
        except Exception as e:
            print(f"Comprehensive improvement generation error: {e}")
            comprehensive_improvements_dict = {"High": [], "Medium": [], "Low": [], "Practice": [], "Immediate": []}
        
        # Convert to flat lists for frontend compatibility
        pause_recommendations = []
        for priority, items in pause_recommendations_dict.items():
            for item in items:
                pause_recommendations.append({
                    "priority": priority,
                    "issue": item["issue"],
                    "action": item["action"],
                    "current": item["current"],
                    "target": item["target"],
                    "target_score": item["target"],
                    "impact": item["impact"]
                })
        
        priority_improvements = []
        for priority, items in comprehensive_improvements_dict.items():
            for item in items:
                priority_improvements.append({
                    "priority": priority,
                    "issue": item["issue"],
                    "action": item["action"],
                    "current": item["current"],
                    "target": item["target"],
                    "target_score": item["target"],
                    "impact": item["impact"]
                })
        
        return {
            "status": "success",
            "structuredSuggestions": structured_suggestions,
            "pauseRecommendations": pause_recommendations,
            "priorityImprovements": priority_improvements,
            "message": "Suggestions generated from frontend-calculated features"
        }
        
    except Exception as e:
        print(f"Error in suggestion generation: {e}")
        return {
            "status": "error",
            "error": str(e),
            "structuredSuggestions": {},
            "pauseRecommendations": [],
            "priorityImprovements": []
        }

@app.post("/test-features/")
async def test_feature_extraction(file: UploadFile = File(...)):
    """Test endpoint to verify feature extraction works"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract features
        features = pause_features_for_file(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return {
            "status": "success",
            "feature_count": len(features),
            "sample_features": dict(list(features.items())[:10])  # First 10 features
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

