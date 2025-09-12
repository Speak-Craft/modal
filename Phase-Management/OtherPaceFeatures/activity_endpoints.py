from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import io
import json
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
from feature_extraction2 import extract_advanced_features, pause_features_for_file, generate_actionable_suggestions, generate_personalized_feedback
import pickle
import os
import soundfile as sf
import tempfile

router = APIRouter()

# Activity Types
ACTIVITY_TYPES = {
    'pacing_curve': {
        'name': 'Pacing Curve Challenge',
        'description': 'Record a 2-3 minute speech and visualize your pacing stability',
        'target_metrics': ['wpm_consistency', 'wpm_std'],
        'badge_requirements': {'steady_flow': {'wpm_std': 20, 'threshold': 'less'}}
    },
    'rate_match': {
        'name': 'Rate Match Drill',
        'description': 'Practice with a metronome at target pace (100-150 WPM)',
        'target_metrics': ['wpm_mean', 'wpm_consistency'],
        'badge_requirements': {'tempo_master': {'consecutive_sessions': 3, 'threshold': 'greater_equal'}}
    },
    'speed_shift': {
        'name': 'Speed Shift Exercise',
        'description': 'Practice intentional pace variation for expressiveness',
        'target_metrics': ['wpm_delta_std', 'controlled_shifts'],
        'badge_requirements': {'dynamic_speaker': {'controlled_shifts': True, 'threshold': 'equal'}}
    },
    'consistency_tracker': {
        'name': 'Consistency Tracker',
        'description': 'Track WPM consistency over time',
        'target_metrics': ['wpm_consistency', 'consistency_streak'],
        'badge_requirements': {'consistency_streak': {'consistency_streak': 7, 'threshold': 'greater_equal'}}
    },
    'pause_timing': {
        'name': 'Pause Timing Drill',
        'description': 'Practice inserting pauses at punctuation',
        'target_metrics': ['optimal_pause_ratio', 'pause_precision'],
        'badge_requirements': {'pause_precision': {'optimal_pause_ratio': 0.7, 'threshold': 'greater_equal'}}
    },
    'excessive_pause_elimination': {
        'name': 'Excessive Pause Elimination',
        'description': 'Eliminate awkward >5s pauses',
        'target_metrics': ['excessive_pauses', 'pause_efficiency'],
        'badge_requirements': {'no_dead_air': {'excessive_pauses': 0, 'threshold': 'equal'}}
    },
    'pause_for_impact': {
        'name': 'Pause for Impact Challenge',
        'description': 'Insert deliberate 1.5-2s pauses after key messages',
        'target_metrics': ['impact_pause_score', 'comprehension_score'],
        'badge_requirements': {'impact_pause': {'comprehension_score': 80, 'threshold': 'greater_equal'}}
    },
    'pause_rhythm': {
        'name': 'Pause Rhythm Training',
        'description': 'Practice rhythmic reading patterns',
        'target_metrics': ['rhythm_consistency', 'pause_pattern_regularity'],
        'badge_requirements': {'rhythm_speaker': {'rhythm_consistency': 70, 'threshold': 'greater_equal'}}
    },
    'confidence_pause': {
        'name': 'Confidence Pause Coach',
        'description': 'Replace hesitation fillers with confident pauses',
        'target_metrics': ['filler_count', 'confidence_score'],
        'badge_requirements': {'confident_voice': {'filler_count': 2, 'threshold': 'less'}}
    },
    'golden_ratio': {
        'name': 'Golden Ratio Speech Challenge',
        'description': 'Achieve natural 1.618x longer pauses at emphasis points',
        'target_metrics': ['golden_ratio_pauses', 'golden_ratio_score'],
        'badge_requirements': {'golden_ratio_master': {'golden_ratio_score': 0.8, 'threshold': 'greater_equal'}}
    },
    'pause_entropy': {
        'name': 'Pause Entropy Game',
        'description': 'Reduce pause randomness with structured delivery',
        'target_metrics': ['pause_entropy', 'entropy_score'],
        'badge_requirements': {'entropy_controller': {'entropy_score': 0.3, 'threshold': 'less_equal'}}
    },
    'cognitive_pause': {
        'name': 'Cognitive Pause Mastery',
        'description': 'Master effective pauses in complex explanations',
        'target_metrics': ['cognitive_pause_score', 'cognitive_score'],
        'badge_requirements': {'cognitive_master': {'cognitive_score': 85, 'threshold': 'greater_equal'}}
    }
}

# Badge System
BADGES = {
    'steady_flow': {'name': 'Steady Flow', 'icon': 'FaEquals', 'color': 'green'},
    'tempo_master': {'name': 'Tempo Master', 'icon': 'FaMusic', 'color': 'blue'},
    'dynamic_speaker': {'name': 'Dynamic Speaker', 'icon': 'FaRocket', 'color': 'purple'},
    'consistency_streak': {'name': 'Consistency Streak', 'icon': 'FaFire', 'color': 'orange'},
    'pause_precision': {'name': 'Pause Precision', 'icon': 'FaCrosshairs', 'color': 'cyan'},
    'no_dead_air': {'name': 'No Dead Air', 'icon': 'FaTimesCircle', 'color': 'red'},
    'impact_pause': {'name': 'Impact Pause', 'icon': 'FaStar', 'color': 'yellow'},
    'rhythm_speaker': {'name': 'Rhythm Speaker', 'icon': 'FaMusic', 'color': 'pink'},
    'confident_voice': {'name': 'Confident Voice', 'icon': 'FaCrown', 'color': 'gold'},
    'golden_ratio_master': {'name': 'Golden Ratio Master', 'icon': 'FaGem', 'color': 'purple'},
    'entropy_controller': {'name': 'Entropy Controller', 'icon': 'FaBrain', 'color': 'indigo'},
    'cognitive_master': {'name': 'Cognitive Master', 'icon': 'FaBolt', 'color': 'teal'}
}

# Level System
LEVELS = {
    'bronze': {'name': 'Bronze', 'color': '#CD7F32', 'threshold': 0.3},
    'silver': {'name': 'Silver', 'color': '#C0C0C0', 'threshold': 0.2},
    'gold': {'name': 'Gold', 'color': '#FFD700', 'threshold': 0.1},
    'platinum': {'name': 'Platinum', 'color': '#E5E4E2', 'threshold': 0.05}
}

def calculate_activity_score(features: Dict, activity_type: str) -> Dict:
    """Calculate activity-specific score and metrics"""
    activity_config = ACTIVITY_TYPES.get(activity_type, {})
    target_metrics = activity_config.get('target_metrics', [])
    
    score = 0
    max_score = 100
    metrics = {}
    
    if activity_type == 'pacing_curve':
        wpm_std = features.get('wpm_std', 0)
        wpm_consistency = features.get('wpm_consistency', 0)
        
        # Score based on WPM consistency (lower std = higher score)
        std_score = max(0, 100 - (wpm_std * 2))  # Penalty for high variation
        consistency_score = wpm_consistency
        
        score = (std_score + consistency_score) / 2
        metrics = {
            'wpm_std': wpm_std,
            'wpm_consistency': wpm_consistency,
            'std_score': std_score,
            'consistency_score': consistency_score
        }
        
    elif activity_type == 'rate_match':
        wpm_mean = features.get('wpm_mean', 0)
        wpm_consistency = features.get('wpm_consistency', 0)
        
        # Target WPM range: 100-150
        target_wpm = 125
        wpm_score = max(0, 100 - abs(wpm_mean - target_wpm) * 2)
        consistency_score = wpm_consistency
        
        score = (wpm_score + consistency_score) / 2
        metrics = {
            'wpm_mean': wpm_mean,
            'wpm_consistency': wpm_consistency,
            'wpm_score': wpm_score,
            'consistency_score': consistency_score,
            'target_wpm': target_wpm
        }
        
    elif activity_type == 'speed_shift':
        wpm_delta_std = features.get('wpm_delta_std', 0)
        controlled_shifts = features.get('controlled_shifts', False)
        
        # Score based on controlled variation
        variation_score = max(0, 100 - wpm_delta_std * 10)
        control_bonus = 20 if controlled_shifts else 0
        
        score = min(100, variation_score + control_bonus)
        metrics = {
            'wpm_delta_std': wpm_delta_std,
            'controlled_shifts': controlled_shifts,
            'variation_score': variation_score,
            'control_bonus': control_bonus
        }
        
    elif activity_type == 'consistency_tracker':
        wpm_consistency = features.get('wpm_consistency', 0)
        consistency_streak = features.get('consistency_streak', 0)
        
        score = wpm_consistency
        metrics = {
            'wpm_consistency': wpm_consistency,
            'consistency_streak': consistency_streak
        }
        
    elif activity_type == 'pause_timing':
        optimal_pause_ratio = features.get('optimal_pause_ratio', 0)
        pause_precision = features.get('pause_precision', 0)
        
        ratio_score = optimal_pause_ratio * 100
        precision_score = pause_precision * 100
        
        score = (ratio_score + precision_score) / 2
        metrics = {
            'optimal_pause_ratio': optimal_pause_ratio,
            'pause_precision': pause_precision,
            'ratio_score': ratio_score,
            'precision_score': precision_score
        }
        
    elif activity_type == 'excessive_pause_elimination':
        excessive_pauses = features.get('excessive_pauses', 0)
        pause_efficiency = features.get('pause_efficiency', 0)
        
        # Penalty for excessive pauses
        excessive_penalty = excessive_pauses * 20
        efficiency_score = pause_efficiency * 100
        
        score = max(0, efficiency_score - excessive_penalty)
        metrics = {
            'excessive_pauses': excessive_pauses,
            'pause_efficiency': pause_efficiency,
            'excessive_penalty': excessive_penalty,
            'efficiency_score': efficiency_score
        }
        
    elif activity_type == 'pause_for_impact':
        impact_pause_score = features.get('impact_pause_score', 0)
        comprehension_score = features.get('comprehension_score', 0)
        
        score = (impact_pause_score + comprehension_score) / 2
        metrics = {
            'impact_pause_score': impact_pause_score,
            'comprehension_score': comprehension_score
        }
        
    elif activity_type == 'pause_rhythm':
        rhythm_consistency = features.get('rhythm_consistency', 0)
        pause_pattern_regularity = features.get('pause_pattern_regularity', 0)
        
        score = (rhythm_consistency + pause_pattern_regularity) / 2
        metrics = {
            'rhythm_consistency': rhythm_consistency,
            'pause_pattern_regularity': pause_pattern_regularity
        }
        
    elif activity_type == 'confidence_pause':
        filler_count = features.get('filler_count', 0)
        confidence_score = features.get('confidence_score', 0)
        
        # Penalty for fillers
        filler_penalty = filler_count * 10
        confidence_bonus = confidence_score
        
        score = max(0, confidence_bonus - filler_penalty)
        metrics = {
            'filler_count': filler_count,
            'confidence_score': confidence_score,
            'filler_penalty': filler_penalty,
            'confidence_bonus': confidence_bonus
        }
        
    elif activity_type == 'golden_ratio':
        golden_ratio_pauses = features.get('golden_ratio_pauses', 0)
        golden_ratio_score = features.get('golden_ratio_score', 0)
        
        score = golden_ratio_score * 100
        metrics = {
            'golden_ratio_pauses': golden_ratio_pauses,
            'golden_ratio_score': golden_ratio_score
        }
        
    elif activity_type == 'pause_entropy':
        pause_entropy = features.get('pause_entropy', 0)
        entropy_score = 1 - pause_entropy  # Lower entropy = higher score
        
        score = entropy_score * 100
        metrics = {
            'pause_entropy': pause_entropy,
            'entropy_score': entropy_score
        }
        
    elif activity_type == 'cognitive_pause':
        cognitive_pause_score = features.get('cognitive_pause_score', 0)
        cognitive_score = features.get('cognitive_score', 0)
        
        score = (cognitive_pause_score + cognitive_score) / 2
        metrics = {
            'cognitive_pause_score': cognitive_pause_score,
            'cognitive_score': cognitive_score
        }
    
    return {
        'score': min(100, max(0, score)),
        'max_score': max_score,
        'metrics': metrics
    }

def check_badge_eligibility(features: Dict, activity_type: str) -> List[Dict]:
    """Check if user is eligible for any badges"""
    activity_config = ACTIVITY_TYPES.get(activity_type, {})
    badge_requirements = activity_config.get('badge_requirements', {})
    
    earned_badges = []
    
    for badge_key, requirements in badge_requirements.items():
        if badge_key in BADGES:
            badge = BADGES[badge_key]
            metric_name = list(requirements.keys())[0]
            threshold = requirements[metric_name]
            threshold_type = requirements.get('threshold', 'greater_equal')
            
            metric_value = features.get(metric_name, 0)
            eligible = False
            
            if threshold_type == 'less':
                eligible = metric_value < threshold
            elif threshold_type == 'less_equal':
                eligible = metric_value <= threshold
            elif threshold_type == 'greater':
                eligible = metric_value > threshold
            elif threshold_type == 'greater_equal':
                eligible = metric_value >= threshold
            elif threshold_type == 'equal':
                eligible = metric_value == threshold
            
            if eligible:
                earned_badges.append(badge)
    
    return earned_badges

def generate_activity_feedback(score: float, activity_type: str, metrics: Dict) -> str:
    """Generate personalized feedback for the activity"""
    if score >= 90:
        return f"Excellent performance! You've mastered the {activity_type.replace('_', ' ')} challenge."
    elif score >= 70:
        return f"Good work! You're making solid progress in {activity_type.replace('_', ' ')}."
    elif score >= 50:
        return f"Fair performance. Keep practicing to improve your {activity_type.replace('_', ' ')} skills."
    else:
        return f"Needs more practice. Focus on the fundamentals of {activity_type.replace('_', ' ')}."

def generate_recommendations(score: float, activity_type: str, metrics: Dict) -> List[str]:
    """Generate specific recommendations based on performance"""
    recommendations = []
    
    if activity_type == 'pacing_curve':
        if metrics.get('wpm_std', 0) > 20:
            recommendations.append("Work on maintaining consistent pace throughout your speech")
        if metrics.get('wpm_consistency', 0) < 70:
            recommendations.append("Practice with a metronome to develop steady rhythm")
            
    elif activity_type == 'rate_match':
        if metrics.get('wpm_mean', 0) < 100:
            recommendations.append("Try to increase your speaking pace slightly")
        elif metrics.get('wpm_mean', 0) > 150:
            recommendations.append("Slow down your pace for better comprehension")
        if metrics.get('wpm_consistency', 0) < 70:
            recommendations.append("Practice maintaining steady pace for longer periods")
            
    elif activity_type == 'pause_timing':
        if metrics.get('optimal_pause_ratio', 0) < 0.7:
            recommendations.append("Practice inserting more strategic pauses at punctuation")
        if metrics.get('pause_precision', 0) < 0.7:
            recommendations.append("Focus on timing your pauses more precisely")
            
    elif activity_type == 'excessive_pause_elimination':
        if metrics.get('excessive_pauses', 0) > 0:
            recommendations.append("Work on reducing pauses longer than 5 seconds")
        if metrics.get('pause_efficiency', 0) < 0.5:
            recommendations.append("Practice more efficient pause usage")
            
    elif activity_type == 'confidence_pause':
        if metrics.get('filler_count', 0) > 2:
            recommendations.append("Replace 'um' and 'uh' with confident pauses")
        if metrics.get('confidence_score', 0) < 70:
            recommendations.append("Build confidence through regular practice")
    
    return recommendations

@router.post("/real-time-analysis/")
async def real_time_analysis(request: Dict):
    """Provide real-time analysis during activity recording"""
    try:
        activity_type = request.get('activityType')
        duration = request.get('duration', 0)
        timestamp = request.get('timestamp', 0)
        
        if not activity_type:
            raise HTTPException(status_code=400, detail="Activity type required")
        
        # Simulate real-time analysis (in production, this would analyze actual audio)
        # For now, return mock data based on activity type
        mock_scores = {
            'pacing_curve': {'score': 75, 'feedback': 'Good pace consistency detected'},
            'rate_match': {'score': 80, 'feedback': 'Target pace maintained well'},
            'speed_shift': {'score': 70, 'feedback': 'Controlled variations detected'},
            'pause_timing': {'score': 85, 'feedback': 'Strategic pauses well-timed'},
            'excessive_pause_elimination': {'score': 90, 'feedback': 'No excessive pauses detected'},
            'pause_for_impact': {'score': 78, 'feedback': 'Impact pauses effective'},
            'pause_rhythm': {'score': 82, 'feedback': 'Rhythmic pattern maintained'},
            'confidence_pause': {'score': 88, 'feedback': 'Confident delivery detected'},
            'golden_ratio': {'score': 76, 'feedback': 'Natural pause ratios detected'},
            'pause_entropy': {'score': 84, 'feedback': 'Structured delivery maintained'},
            'cognitive_pause': {'score': 79, 'feedback': 'Cognitive pauses effective'}
        }
        
        activity_data = mock_scores.get(activity_type, {'score': 70, 'feedback': 'Analysis in progress'})
        
        return JSONResponse({
            'success': True,
            'score': activity_data['score'],
            'feedback': activity_data['feedback'],
            'timestamp': timestamp,
            'duration': duration,
            'activity_type': activity_type
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Real-time analysis error: {str(e)}")

@router.post("/analyze-activity/")
async def analyze_activity(file: UploadFile = File(...), activityType: str = None):
    """Analyze completed activity and provide comprehensive results"""
    try:
        if not activityType:
            raise HTTPException(status_code=400, detail="Activity type required")
        
        if activityType not in ACTIVITY_TYPES:
            raise HTTPException(status_code=400, detail="Invalid activity type")
        
        # Read audio file
        audio_data = await file.read()
        audio_io = io.BytesIO(audio_data)
        
        # Load audio with librosa
        y, sr = librosa.load(audio_io, sr=16000)
        
        # Save audio to temporary file for feature extraction
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, y, sr)
            temp_path = temp_file.name
        
        try:
            # Extract comprehensive features
            features = extract_advanced_features(temp_path)
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
        
        # Calculate activity-specific score
        activity_result = calculate_activity_score(features, activityType)
        
        # Check badge eligibility
        earned_badges = check_badge_eligibility(features, activityType)
        
        # Generate feedback and recommendations
        feedback = generate_activity_feedback(
            activity_result['score'], 
            activityType, 
            activity_result['metrics']
        )
        
        recommendations = generate_recommendations(
            activity_result['score'], 
            activityType, 
            activity_result['metrics']
        )
        
        # Calculate additional metrics
        duration = len(y) / sr
        word_count = features.get('word_count', 0)
        average_wpm = features.get('wpm_mean', 0)
        consistency_score = features.get('wpm_consistency', 0)
        pause_ratio = features.get('pause_ratio', 0)
        
        return JSONResponse({
            'success': True,
            'activity_type': activityType,
            'final_score': activity_result['score'],
            'max_score': activity_result['max_score'],
            'metrics': activity_result['metrics'],
            'duration': duration,
            'word_count': word_count,
            'average_wpm': average_wpm,
            'consistency_score': consistency_score,
            'pause_ratio': pause_ratio,
            'feedback': feedback,
            'recommendations': recommendations,
            'new_badges': earned_badges,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activity analysis error: {str(e)}")

@router.get("/activity-types/")
async def get_activity_types():
    """Get all available activity types"""
    return JSONResponse({
        'success': True,
        'activity_types': ACTIVITY_TYPES,
        'badges': BADGES,
        'levels': LEVELS
    })

@router.get("/user-progress/")
async def get_user_progress():
    """Get user's overall progress and statistics"""
    # In production, this would fetch from database
    return JSONResponse({
        'success': True,
        'total_sessions': 0,
        'current_streak': 0,
        'best_streak': 0,
        'total_time': 0,
        'average_wpm': 0,
        'consistency_score': 0,
        'badges_earned': [],
        'current_level': 'bronze'
    })

@router.post("/update-progress/")
async def update_user_progress(progress_data: Dict):
    """Update user's progress after completing an activity"""
    try:
        # In production, this would update database
        return JSONResponse({
            'success': True,
            'message': 'Progress updated successfully',
            'updated_data': progress_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Progress update error: {str(e)}")
