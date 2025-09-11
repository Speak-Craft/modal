#!/usr/bin/env python3
"""
Enhanced Synthetic Data Generator for Speech Pace Management
Generates balanced synthetic data to improve model accuracy to 85-90%
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Industry-standard pause values (from feature_extraction2.txt)
MICRO_PAUSE = 0.1
SHORT_PAUSE = 0.3
MED_PAUSE = 1.0
LONG_PAUSE = 2.5
EXCESSIVE_PAUSE = 5.0

# Toastmasters Standards
OPTIMAL_PAUSE_RATIO_MIN = 0.08
OPTIMAL_PAUSE_RATIO_MAX = 0.12
TRANSITION_PAUSE_MIN = 1.5
TRANSITION_PAUSE_MAX = 2.0
EMPHASIS_PAUSE_MIN = 0.5
EMPHASIS_PAUSE_MAX = 1.0
AUDIENCE_PAUSE_MIN = 2.0
AUDIENCE_PAUSE_MAX = 3.0

@dataclass
class LabelProfile:
    """Profile defining characteristics for each label category"""
    name: str
    weight: float
    pause_ratio_range: Tuple[float, float]
    wpm_cv_range: Tuple[float, float]
    long_count_range: Tuple[int, int]
    rhythm_outliers_range: Tuple[int, int]
    contextual_score_range: Tuple[float, float]
    speech_continuity_range: Tuple[float, float]
    toastmasters_compliance_range: Tuple[float, float]

# Define label profiles based on the feature extraction logic
LABEL_PROFILES = {
    "toastmasters_excellent": LabelProfile(
        name="toastmasters_excellent",
        weight=0.15,  # 15% of data
        pause_ratio_range=(0.08, 0.12),
        wpm_cv_range=(0.05, 0.15),
        long_count_range=(0, 2),
        rhythm_outliers_range=(0, 1),
        contextual_score_range=(0.7, 1.0),
        speech_continuity_range=(0.85, 0.95),
        toastmasters_compliance_range=(0.8, 1.0)
    ),
    "good_pace_control": LabelProfile(
        name="good_pace_control",
        weight=0.20,  # 20% of data
        pause_ratio_range=(0.06, 0.15),
        wpm_cv_range=(0.10, 0.25),
        long_count_range=(0, 3),
        rhythm_outliers_range=(0, 2),
        contextual_score_range=(0.4, 0.8),
        speech_continuity_range=(0.75, 0.90),
        toastmasters_compliance_range=(0.6, 0.8)
    ),
    "moderate_pace_consistency": LabelProfile(
        name="moderate_pace_consistency",
        weight=0.15,  # 15% of data
        pause_ratio_range=(0.10, 0.18),
        wpm_cv_range=(0.20, 0.35),
        long_count_range=(1, 4),
        rhythm_outliers_range=(1, 4),
        contextual_score_range=(0.2, 0.6),
        speech_continuity_range=(0.65, 0.80),
        toastmasters_compliance_range=(0.4, 0.7)
    ),
    "needs_pause_improvement": LabelProfile(
        name="needs_pause_improvement",
        weight=0.15,  # 15% of data
        pause_ratio_range=(0.12, 0.20),
        wpm_cv_range=(0.15, 0.30),
        long_count_range=(2, 6),
        rhythm_outliers_range=(2, 5),
        contextual_score_range=(0.0, 0.4),
        speech_continuity_range=(0.60, 0.75),
        toastmasters_compliance_range=(0.3, 0.6)
    ),
    "inconsistent_pace": LabelProfile(
        name="inconsistent_pace",
        weight=0.15,  # 15% of data
        pause_ratio_range=(0.08, 0.18),
        wpm_cv_range=(0.25, 0.45),
        long_count_range=(1, 5),
        rhythm_outliers_range=(3, 8),
        contextual_score_range=(-0.2, 0.3),
        speech_continuity_range=(0.50, 0.75),
        toastmasters_compliance_range=(0.2, 0.5)
    ),
    "poor_pause_control": LabelProfile(
        name="poor_pause_control",
        weight=0.10,  # 10% of data
        pause_ratio_range=(0.15, 0.30),
        wpm_cv_range=(0.20, 0.50),
        long_count_range=(4, 12),
        rhythm_outliers_range=(4, 15),
        contextual_score_range=(-0.5, 0.2),
        speech_continuity_range=(0.40, 0.70),
        toastmasters_compliance_range=(0.0, 0.4)
    ),
    "poor_rhythm": LabelProfile(
        name="poor_rhythm",
        weight=0.05,  # 5% of data
        pause_ratio_range=(0.10, 0.25),
        wpm_cv_range=(0.30, 0.60),
        long_count_range=(3, 8),
        rhythm_outliers_range=(6, 20),
        contextual_score_range=(-0.3, 0.1),
        speech_continuity_range=(0.30, 0.60),
        toastmasters_compliance_range=(0.0, 0.3)
    ),
    "moderate_rhythm": LabelProfile(
        name="moderate_rhythm",
        weight=0.05,  # 5% of data
        pause_ratio_range=(0.08, 0.20),
        wpm_cv_range=(0.20, 0.40),
        long_count_range=(2, 6),
        rhythm_outliers_range=(3, 8),
        contextual_score_range=(0.0, 0.5),
        speech_continuity_range=(0.60, 0.80),
        toastmasters_compliance_range=(0.2, 0.6)
    ),
    "cognitive_load_issues": LabelProfile(
        name="cognitive_load_issues",
        weight=0.05,  # 5% of data
        pause_ratio_range=(0.12, 0.22),
        wpm_cv_range=(0.15, 0.35),
        long_count_range=(3, 7),
        rhythm_outliers_range=(2, 6),
        contextual_score_range=(-0.1, 0.3),
        speech_continuity_range=(0.55, 0.75),
        toastmasters_compliance_range=(0.1, 0.5)
    ),
    "needs_overall_improvement": LabelProfile(
        name="needs_overall_improvement",
        weight=0.05,  # 5% of data
        pause_ratio_range=(0.10, 0.25),
        wpm_cv_range=(0.25, 0.45),
        long_count_range=(2, 8),
        rhythm_outliers_range=(3, 10),
        contextual_score_range=(-0.2, 0.2),
        speech_continuity_range=(0.50, 0.70),
        toastmasters_compliance_range=(0.1, 0.4)
    )
}

class SyntheticDataGenerator:
    """Generate synthetic speech pace management data"""
    
    def __init__(self, target_samples: int = 2000):
        self.target_samples = target_samples
        self.samples_per_label = {}
        
        # Calculate samples per label based on weights
        for label, profile in LABEL_PROFILES.items():
            self.samples_per_label[label] = int(target_samples * profile.weight)
    
    def generate_duration(self, label_profile: LabelProfile) -> float:
        """Generate realistic speech duration based on label profile"""
        if "excellent" in label_profile.name or "good" in label_profile.name:
            # Better speakers tend to have longer, more structured speeches
            return np.random.uniform(120, 300)
        elif "poor" in label_profile.name:
            # Poor speakers might have very short or very long speeches
            return np.random.uniform(30, 400)
        else:
            # Moderate speakers
            return np.random.uniform(60, 250)
    
    def generate_pause_counts(self, duration: float, label_profile: LabelProfile) -> Tuple[int, int, int]:
        """Generate pause counts based on duration and label profile"""
        # Base pause frequency per minute
        if "excellent" in label_profile.name:
            base_freq = np.random.uniform(8, 15)  # More controlled pauses
        elif "poor" in label_profile.name:
            base_freq = np.random.uniform(15, 30)  # Many pauses
        else:
            base_freq = np.random.uniform(10, 20)
        
        total_pauses = int(duration * base_freq / 60)
        
        # Distribute pauses based on label characteristics
        if "excellent" in label_profile.name:
            # More short pauses, fewer long ones
            short_ratio = np.random.uniform(0.6, 0.8)
            med_ratio = np.random.uniform(0.2, 0.35)
            long_ratio = 1 - short_ratio - med_ratio
        elif "poor" in label_profile.name:
            # More long pauses
            short_ratio = np.random.uniform(0.3, 0.5)
            med_ratio = np.random.uniform(0.3, 0.5)
            long_ratio = 1 - short_ratio - med_ratio
        else:
            # Balanced distribution
            short_ratio = np.random.uniform(0.4, 0.6)
            med_ratio = np.random.uniform(0.3, 0.4)
            long_ratio = 1 - short_ratio - med_ratio
        
        short_count = int(total_pauses * short_ratio)
        med_count = int(total_pauses * med_ratio)
        long_count = max(0, total_pauses - short_count - med_count)
        
        # Ensure counts are within label profile ranges
        long_count = min(long_count, label_profile.long_count_range[1])
        long_count = max(long_count, label_profile.long_count_range[0])
        
        return short_count, med_count, long_count
    
    def generate_pause_times(self, short_count: int, med_count: int, long_count: int) -> Tuple[float, float, float]:
        """Generate pause times for each category"""
        short_time = sum(np.random.uniform(MICRO_PAUSE, SHORT_PAUSE) for _ in range(short_count))
        med_time = sum(np.random.uniform(SHORT_PAUSE, MED_PAUSE) for _ in range(med_count))
        long_time = sum(np.random.uniform(MED_PAUSE, EXCESSIVE_PAUSE) for _ in range(long_count))
        
        return short_time, med_time, long_time
    
    def generate_wpm_features(self, duration: float, label_profile: LabelProfile) -> Dict[str, float]:
        """Generate WPM-related features"""
        # Base WPM based on label
        if "excellent" in label_profile.name:
            base_wpm = np.random.uniform(140, 180)
            wpm_std = np.random.uniform(5, 15)
        elif "poor" in label_profile.name:
            base_wpm = np.random.uniform(100, 200)
            wpm_std = np.random.uniform(20, 50)
        else:
            base_wpm = np.random.uniform(120, 170)
            wpm_std = np.random.uniform(10, 30)
        
        # Generate WPM variations
        num_segments = max(5, int(duration / 10))  # ~10 second segments
        wpm_bins = np.random.normal(base_wpm, wpm_std, num_segments)
        wpm_bins = np.clip(wpm_bins, 50, 250)  # Realistic WPM range
        
        wpm_mean = float(np.mean(wpm_bins))
        wpm_std = float(np.std(wpm_bins))
        
        # WPM deltas
        wpm_delta = np.diff(wpm_bins) if len(wpm_bins) > 1 else np.array([0.0])
        wpm_delta_mean = float(np.mean(wpm_delta))
        wpm_delta_std = float(np.std(wpm_delta))
        
        # WPM coefficient of variation
        wpm_cv = wpm_std / wpm_mean if wpm_mean > 0 else 0.0
        
        # WPM range
        wpm_range = float(np.max(wpm_bins) - np.min(wpm_bins))
        
        # WPM acceleration and jerk
        wpm_acceleration = float(np.mean(np.diff(wpm_delta))) if len(wpm_delta) > 1 else 0.0
        wpm_jerk = float(np.mean(np.abs(np.diff(wpm_delta)))) if len(wpm_delta) > 1 else 0.0
        
        return {
            "wpm_mean": wpm_mean,
            "wpm_std": wpm_std,
            "wpm_delta_mean": wpm_delta_mean,
            "wpm_delta_std": wpm_delta_std,
            "wpm_cv": wpm_cv,
            "wpm_range": wpm_range,
            "wpm_acceleration": wpm_acceleration,
            "wpm_jerk": wpm_jerk
        }
    
    def generate_rhythm_features(self, duration: float, label_profile: LabelProfile) -> Dict[str, float]:
        """Generate rhythm and timing features"""
        # Segment durations
        num_segments = max(3, int(duration / 15))  # ~15 second segments
        if "excellent" in label_profile.name:
            seg_duration_std = np.random.uniform(0.5, 2.0)
            rhythm_outliers = np.random.randint(0, 2)
        elif "poor" in label_profile.name:
            seg_duration_std = np.random.uniform(2.0, 5.0)
            rhythm_outliers = np.random.randint(6, 15)
        else:
            seg_duration_std = np.random.uniform(1.0, 3.0)
            rhythm_outliers = np.random.randint(2, 8)
        
        # Ensure rhythm outliers are within profile range
        rhythm_outliers = min(rhythm_outliers, label_profile.rhythm_outliers_range[1])
        rhythm_outliers = max(rhythm_outliers, label_profile.rhythm_outliers_range[0])
        
        # Gap clustering
        gap_clustering = np.random.uniform(0.5, 3.0)
        
        # Rhythm regularity
        rhythm_regularity = 1.0 / (1.0 + seg_duration_std)
        
        return {
            "seg_duration_std": seg_duration_std,
            "rhythm_outliers": rhythm_outliers,
            "gap_clustering": gap_clustering,
            "rhythm_regularity": rhythm_regularity
        }
    
    def generate_contextual_features(self, label_profile: LabelProfile) -> Dict[str, float]:
        """Generate contextual pause analysis features"""
        contextual_score = np.random.uniform(
            label_profile.contextual_score_range[0],
            label_profile.contextual_score_range[1]
        )
        
        # Transition and emphasis pauses
        if contextual_score > 0.5:
            transition_count = np.random.randint(3, 8)
            emphasis_count = np.random.randint(5, 12)
        else:
            transition_count = np.random.randint(0, 5)
            emphasis_count = np.random.randint(0, 8)
        
        # Optimal ratios
        optimal_transition_ratio = np.random.uniform(0.6, 1.0) if transition_count > 0 else 0.0
        optimal_emphasis_ratio = np.random.uniform(0.7, 1.0) if emphasis_count > 0 else 0.0
        
        return {
            "contextual_pause_score": contextual_score,
            "transition_pause_count": transition_count,
            "emphasis_pause_count": emphasis_count,
            "optimal_transition_ratio": optimal_transition_ratio,
            "optimal_emphasis_ratio": optimal_emphasis_ratio
        }
    
    def generate_rhythm_analysis_features(self, label_profile: LabelProfile) -> Dict[str, float]:
        """Generate rhythm pattern analysis features"""
        if "excellent" in label_profile.name:
            rhythm_consistency = np.random.uniform(0.7, 0.95)
            golden_ratio_pauses = np.random.uniform(0.3, 0.6)
            pause_entropy = np.random.uniform(0.3, 0.7)
            pause_autocorrelation = np.random.uniform(0.4, 0.8)
        elif "poor" in label_profile.name:
            rhythm_consistency = np.random.uniform(0.2, 0.6)
            golden_ratio_pauses = np.random.uniform(0.0, 0.3)
            pause_entropy = np.random.uniform(0.6, 0.9)
            pause_autocorrelation = np.random.uniform(0.0, 0.4)
        else:
            rhythm_consistency = np.random.uniform(0.4, 0.8)
            golden_ratio_pauses = np.random.uniform(0.1, 0.5)
            pause_entropy = np.random.uniform(0.4, 0.8)
            pause_autocorrelation = np.random.uniform(0.2, 0.6)
        
        return {
            "pause_rhythm_consistency": rhythm_consistency,
            "golden_ratio_pauses": golden_ratio_pauses,
            "pause_entropy": pause_entropy,
            "pause_autocorrelation": pause_autocorrelation
        }
    
    def generate_cognitive_features(self, label_profile: LabelProfile) -> Dict[str, float]:
        """Generate cognitive load and confidence features"""
        if "excellent" in label_profile.name:
            cognitive_score = np.random.uniform(0.6, 1.0)
            confidence_score = np.random.uniform(0.8, 1.0)
            memory_retrieval_pauses = np.random.randint(0, 3)
        elif "poor" in label_profile.name:
            cognitive_score = np.random.uniform(0.0, 0.4)
            confidence_score = np.random.uniform(0.4, 0.7)
            memory_retrieval_pauses = np.random.randint(3, 8)
        else:
            cognitive_score = np.random.uniform(0.3, 0.7)
            confidence_score = np.random.uniform(0.6, 0.9)
            memory_retrieval_pauses = np.random.randint(1, 5)
        
        optimal_cognitive_pause_ratio = np.random.uniform(0.5, 1.0)
        
        return {
            "cognitive_pause_score": cognitive_score,
            "memory_retrieval_pauses": memory_retrieval_pauses,
            "confidence_score": confidence_score,
            "optimal_cognitive_pause_ratio": optimal_cognitive_pause_ratio
        }
    
    def generate_advanced_statistical_features(self) -> Dict[str, float]:
        """Generate advanced statistical features"""
        return {
            "pause_fractal_dimension": np.random.uniform(0.5, 2.0),
            "pause_spectral_density": np.random.uniform(0.0, 5.0),
            "pause_trend_analysis": np.random.uniform(0.0, 1.0),
            "pause_volatility": np.random.uniform(0.1, 3.0)
        }
    
    def generate_audio_features(self, label_profile: LabelProfile) -> Dict[str, float]:
        """Generate audio features (MFCC, spectral, pitch, etc.)"""
        features = {}
        
        # MFCC features (13 coefficients)
        for i in range(13):
            features[f"mfcc_{i}_mean"] = np.random.uniform(-20, 20)
            features[f"mfcc_{i}_std"] = np.random.uniform(0, 10)
        
        # Spectral features
        features.update({
            "spectral_centroid": np.random.uniform(1000, 4000),
            "spectral_rolloff": np.random.uniform(2000, 8000),
            "spectral_bandwidth": np.random.uniform(500, 2000),
            "zero_crossing_rate": np.random.uniform(0.01, 0.1),
            "tempo": np.random.uniform(60, 180)
        })
        
        # Pitch features
        features.update({
            "pitch_mean": np.random.uniform(80, 300),
            "pitch_std": np.random.uniform(10, 80),
            "pitch_min": np.random.uniform(60, 150),
            "pitch_max": np.random.uniform(200, 500)
        })
        
        # Voice quality features
        features.update({
            "jitter_local": np.random.uniform(0.001, 0.05),
            "jitter_rap": np.random.uniform(0.001, 0.05),
            "jitter_ppq5": np.random.uniform(0.001, 0.05),
            "shimmer_local": np.random.uniform(0.01, 0.2),
            "shimmer_apq3": np.random.uniform(0.01, 0.2),
            "shimmer_apq5": np.random.uniform(0.01, 0.2),
            "hnr_mean": np.random.uniform(5, 25)
        })
        
        # Formant features
        features.update({
            "f1_mean": np.random.uniform(200, 800),
            "f2_mean": np.random.uniform(800, 2500),
            "f3_mean": np.random.uniform(2000, 3500)
        })
        
        return features
    
    def generate_long_pause_distribution(self, duration: float, long_count: int) -> List[float]:
        """Generate long pause distribution across time chunks"""
        chunk_size = 30.0  # 30-second chunks
        num_chunks = int(np.ceil(duration / chunk_size))
        distribution = [0.0] * num_chunks
        
        if long_count > 0:
            # Distribute long pauses across chunks
            for _ in range(long_count):
                chunk_idx = np.random.randint(0, num_chunks)
                pause_duration = np.random.uniform(MED_PAUSE, EXCESSIVE_PAUSE)
                distribution[chunk_idx] += pause_duration
        
        return distribution
    
    def generate_single_sample(self, label: str) -> Dict[str, float]:
        """Generate a single synthetic sample for the given label"""
        profile = LABEL_PROFILES[label]
        
        # Generate basic features
        duration = self.generate_duration(profile)
        short_count, med_count, long_count = self.generate_pause_counts(duration, profile)
        short_time, med_time, long_time = self.generate_pause_times(short_count, med_count, long_count)
        
        # Calculate derived features
        total_pause_time = short_time + med_time + long_time
        pause_ratio = total_pause_time / duration if duration > 0 else 0.0
        
        # Rates per minute
        dur_min = duration / 60.0
        short_rate_pm = short_count / dur_min if dur_min > 0 else 0.0
        med_rate_pm = med_count / dur_min if dur_min > 0 else 0.0
        long_rate_pm = long_count / dur_min if dur_min > 0 else 0.0
        all_rate_pm = (short_count + med_count + long_count) / dur_min if dur_min > 0 else 0.0
        
        # Pause statistics
        all_pauses = []
        all_pauses.extend([np.random.uniform(MICRO_PAUSE, SHORT_PAUSE) for _ in range(short_count)])
        all_pauses.extend([np.random.uniform(SHORT_PAUSE, MED_PAUSE) for _ in range(med_count)])
        all_pauses.extend([np.random.uniform(MED_PAUSE, EXCESSIVE_PAUSE) for _ in range(long_count)])
        
        if all_pauses:
            pause_p50 = float(np.percentile(all_pauses, 50))
            pause_p90 = float(np.percentile(all_pauses, 90))
            pause_p95 = float(np.percentile(all_pauses, 95))
            pause_std = float(np.std(all_pauses))
            pause_min = float(np.min(all_pauses))
            pause_max = float(np.max(all_pauses))
        else:
            pause_p50 = pause_p90 = pause_p95 = pause_std = pause_min = pause_max = 0.0
        
        # Max long streak
        max_long_streak = np.random.randint(0, min(5, long_count + 1))
        
        # Generate additional features
        wpm_features = self.generate_wpm_features(duration, profile)
        rhythm_features = self.generate_rhythm_features(duration, profile)
        contextual_features = self.generate_contextual_features(profile)
        rhythm_analysis_features = self.generate_rhythm_analysis_features(profile)
        cognitive_features = self.generate_cognitive_features(profile)
        advanced_stats = self.generate_advanced_statistical_features()
        audio_features = self.generate_audio_features(profile)
        
        # Speech flow metrics
        total_segments = max(1, int(duration / 10))  # Approximate segments
        speech_continuity = np.random.uniform(
            profile.speech_continuity_range[0],
            profile.speech_continuity_range[1]
        )
        
        pause_spacing_consistency = np.random.uniform(0.3, 0.9)
        speaking_efficiency = 1.0 - pause_ratio
        pause_efficiency = total_pause_time / max(1, short_count + med_count + long_count)
        
        # Toastmasters compliance
        toastmasters_compliance = np.random.uniform(
            profile.toastmasters_compliance_range[0],
            profile.toastmasters_compliance_range[1]
        )
        
        # Long pause distribution
        long_dist = self.generate_long_pause_distribution(duration, long_count)
        
        # Create sample dictionary
        sample = {
            "total_duration": duration,
            "short_count": short_count,
            "med_count": med_count,
            "long_count": long_count,
            "short_time": short_time,
            "med_time": med_time,
            "long_time": long_time,
            "total_pause_time": total_pause_time,
            "pause_ratio": pause_ratio,
            "short_rate_pm": short_rate_pm,
            "med_rate_pm": med_rate_pm,
            "long_rate_pm": long_rate_pm,
            "all_rate_pm": all_rate_pm,
            "pause_p50": pause_p50,
            "pause_p90": pause_p90,
            "pause_p95": pause_p95,
            "pause_std": pause_std,
            "pause_min": pause_min,
            "pause_max": pause_max,
            "max_long_streak": max_long_streak,
            "speech_continuity": speech_continuity,
            "pause_spacing_consistency": pause_spacing_consistency,
            "speaking_efficiency": speaking_efficiency,
            "pause_efficiency": pause_efficiency,
            "toastmasters_compliance_score": toastmasters_compliance,
            "filename": f"synthetic_{label}_{np.random.randint(1000, 9999)}.wav",
            "label": label
        }
        
        # Add all generated features
        sample.update(wpm_features)
        sample.update(rhythm_features)
        sample.update(contextual_features)
        sample.update(rhythm_analysis_features)
        sample.update(cognitive_features)
        sample.update(advanced_stats)
        sample.update(audio_features)
        
        # Add long pause distribution columns
        for i, dist_val in enumerate(long_dist):
            sample[f"long_dist_{i}"] = dist_val
        
        # Fill remaining long_dist columns with 0
        max_dist_cols = 86  # Based on the original data
        for i in range(len(long_dist), max_dist_cols):
            sample[f"long_dist_{i}"] = 0.0
        
        return sample
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete synthetic dataset"""
        all_samples = []
        
        print("Generating synthetic data...")
        for label, count in self.samples_per_label.items():
            print(f"Generating {count} samples for {label}")
            for i in range(count):
                sample = self.generate_single_sample(label)
                all_samples.append(sample)
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{count} samples")
        
        df = pd.DataFrame(all_samples)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nGenerated {len(df)} synthetic samples")
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        print("\nLabel percentages:")
        print(df['label'].value_counts(normalize=True) * 100)
        
        return df

def main():
    """Main function to generate synthetic data"""
    print("🚀 Enhanced Synthetic Data Generator for Speech Pace Management")
    print("=" * 70)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(target_samples=2000)
    synthetic_df = generator.generate_dataset()
    
    # Save synthetic data
    synthetic_df.to_csv("synthetic_pause_features.csv", index=False)
    print(f"\n✅ Synthetic dataset saved to synthetic_pause_features.csv")
    
    # Load original data
    try:
        original_df = pd.read_csv("enhanced_pause_features.csv")
        print(f"\n📊 Original dataset: {len(original_df)} samples")
        print("Original label distribution:")
        print(original_df['label'].value_counts())
        
        # Combine datasets
        combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        combined_df.to_csv("enhanced_pause_features_with_synthetic.csv", index=False)
        print(f"\n✅ Combined dataset saved to enhanced_pause_features_with_synthetic.csv")
        print(f"📊 Combined dataset: {len(combined_df)} samples")
        print("\nCombined label distribution:")
        print(combined_df['label'].value_counts())
        print("\nCombined label percentages:")
        print(combined_df['label'].value_counts(normalize=True) * 100)
        
    except FileNotFoundError:
        print("\n⚠️  Original dataset not found. Only synthetic data generated.")
    
    print("\n🎉 Synthetic data generation complete!")
    print("\n📁 Next steps:")
    print("  1. Use enhanced_pause_features_with_synthetic.csv for training")
    print("  2. The balanced dataset should improve model accuracy to 85-90%")
    print("  3. Train your model with the enhanced dataset")

if __name__ == "__main__":
    main()