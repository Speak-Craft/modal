from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
import os
import tempfile
import librosa
import soundfile as sf
import numpy as np
from typing import Dict

from .rate_core import denoise_wav_to_path, transcribe_and_pacing, classify_rate

router = APIRouter()


@router.post("/real-time-analysis/")
async def real_time_analysis_rate(request: Dict):
    """Lightweight mock for real-time feedback specific to rate activities.
    Replace with streaming analysis when frontend sends audio chunks."""
    try:
        duration = float(request.get("duration", 0))
        # Simple heuristic: progress by 2-min target
        target_seconds = 120.0
        progress = max(0.0, min(100.0, (duration / target_seconds) * 100.0))
        # Keep a neutral helpful message
        return JSONResponse({
            "success": True,
            "score": 75,
            "targetScore": 100,
            "progress": progress,
            "feedback": "Keep a steady rhythm near 120–140 WPM.",
            "metrics": {}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rate real-time analysis error: {str(e)}")


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


