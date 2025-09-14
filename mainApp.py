from __future__ import annotations

import os
import sys
from pathlib import Path
import importlib.util
from types import ModuleType
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn


def _import_module_from_path(module_name: str, file_path: Path, add_sys_path: Optional[Path] = None, chdir_for_import: bool = False) -> ModuleType:
    """
    Import a python module from an arbitrary file path.

    - add_sys_path: if provided, temporarily append to sys.path to satisfy absolute imports inside the module
    - chdir_for_import: if True, temporarily chdir to the module's directory so relative file loads work during import
    """
    original_cwd = Path.cwd()
    added_sys_path = False

    try:
        if add_sys_path and str(add_sys_path) not in sys.path:
            sys.path.insert(0, str(add_sys_path))
            added_sys_path = True

        if chdir_for_import:
            os.chdir(str(file_path.parent))

        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to create import spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        # Restore working directory and sys.path
        if chdir_for_import:
            os.chdir(str(original_cwd))
        if added_sys_path and add_sys_path is not None:
            try:
                sys.path.remove(str(add_sys_path))
            except ValueError:
                pass


def create_unified_app() -> FastAPI:
    """Create a FastAPI app that mounts both sub-apps on a single server/port."""
    base_dir = Path(__file__).resolve().parent

    # Paths to the two services (moved under Phase-Management)
    other_pace_dir = base_dir / "Phase-Management" / "OtherPaceFeatures"
    other_pace_app_path = other_pace_dir / "app.py"

    rate_dir = base_dir / "Phase-Management" / "speech-rate-detection"
    rate_app_path = rate_dir / "test.py"

    # Filler-words service
    filler_dir = base_dir / "Filler-words"
    filler_app_path = filler_dir / "app.py"

    # Loudness service
    loudness_dir = base_dir / "loudness-model"
    loudness_app_path = loudness_dir / "predictLoudness.py"

    # Import sub-apps
    # app.py uses `from feature_extraction2 import ...` and reads local model files.
    # We add its directory to sys.path and chdir during import so absolute/relative lookups work.
    other_pace_module = _import_module_from_path(
        module_name="other_pace_app",
        file_path=other_pace_app_path,
        add_sys_path=other_pace_dir,
        chdir_for_import=True,
    )

    # test.py loads local .pkl files relative to its own folder during import.
    rate_module = _import_module_from_path(
        module_name="speech_rate_app",
        file_path=rate_app_path,
        add_sys_path=rate_dir,
        chdir_for_import=True,
    )

    # Import filler-words app
    filler_module = _import_module_from_path(
        module_name="filler_words_app",
        file_path=filler_app_path,
        add_sys_path=filler_dir,
        chdir_for_import=True,
    )

    # Import loudness app
    loudness_module = _import_module_from_path(
        module_name="loudness_app",
        file_path=loudness_app_path,
        add_sys_path=loudness_dir,
        chdir_for_import=True,
    )

    # Retrieve FastAPI instances from imported modules
    try:
        pause_app: FastAPI = getattr(other_pace_module, "app")
    except AttributeError as exc:
        raise RuntimeError("OtherPaceFeatures/app.py does not expose a FastAPI instance named 'app'") from exc

    try:
        rate_app: FastAPI = getattr(rate_module, "app")
    except AttributeError as exc:
        raise RuntimeError("speech-rate-detection/test.py does not expose a FastAPI instance named 'app'") from exc

    try:
        filler_app: FastAPI = getattr(filler_module, "app")
    except AttributeError as exc:
        raise RuntimeError("Filler-words/app.py does not expose a FastAPI instance named 'app'") from exc

    try:
        loudness_app: FastAPI = getattr(loudness_module, "app")
    except AttributeError as exc:
        raise RuntimeError("loudness-model/predictLoudness.py does not expose a FastAPI instance named 'app'") from exc

    # Create unified app and mount the sub-apps
    unified = FastAPI(title="SpeakCraft Unified API", version="1.0.0")

    # Root-level CORS so even 404s and redirects include CORS headers
    unified.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount at stable prefixes to avoid route collisions and to keep a single port
    unified.mount("/pause", pause_app)
    unified.mount("/rate", rate_app)
    unified.mount("/filler", filler_app)
    unified.mount("/loudness", loudness_app)

    # Backward/compatibility routes to support existing frontend paths
    @unified.post("/rate-analysis/")
    async def _compat_rate_analysis():  # type: ignore[misc]
        return RedirectResponse(url="/rate/rate-analysis/", status_code=307)

    @unified.post("/pause-analysis/")
    async def _compat_pause_analysis():  # type: ignore[misc]
        return RedirectResponse(url="/pause/pause-analysis/", status_code=307)

    # New compatibility routes for activity endpoints used by the Pace Management UI
    # Compatibility dispatchers that route based on activityType to keep FE paths stable
    from fastapi import Request

    RATE_ACTIVITY_TYPES = {"pacing_curve", "rate_match", "speed_shift", "consistency_tracker", "ideal_pace_challenge"}
    PAUSE_ACTIVITY_TYPES = {
        "pause_timing",
        "excessive_pause_elimination",
        "pause_for_impact",
        "pause_rhythm",
        "confidence_pause",
        "golden_ratio",
        "pause_entropy",
        "cognitive_pause",
    }

    @unified.post("/real-time-analysis/")
    async def _compat_real_time_analysis():  # type: ignore[misc]
        # For compatibility routing, we'll redirect to rate by default since the frontend 
        # is calling this for rate activities (ideal_pace_challenge)
        return RedirectResponse(url="/rate/real-time-analysis/", status_code=307)

    @unified.post("/analyze-activity/")
    async def _compat_analyze_activity(request: Request):  # type: ignore[misc]
        # Decide based on query param since we cannot read multipart body here
        qp = request.query_params
        activity_type = qp.get("activityType") or qp.get("activity_type") or qp.get("activity")
        scope = qp.get("scope")
        if scope == "rate" or activity_type in RATE_ACTIVITY_TYPES:
            return RedirectResponse(url="/rate/analyze-activity/", status_code=307)
        if scope == "pause" or activity_type in PAUSE_ACTIVITY_TYPES:
            return RedirectResponse(url="/pause/analyze-activity/", status_code=307)
        # Default to pause
        return RedirectResponse(url="/pause/analyze-activity/", status_code=307)

    @unified.post("/generate-suggestions/")
    async def _compat_generate_suggestions():  # type: ignore[misc]
        return RedirectResponse(url="/pause/generate-suggestions/", status_code=307)

    # Compatibility route for rate feedback generation used by the frontend
    @unified.post("/generate-rate-feedback/")
    async def _compat_generate_rate_feedback():  # type: ignore[misc]
        return RedirectResponse(url="/rate/generate-rate-feedback/", status_code=307)

    @unified.post("/ideal-pace-challenge/")
    async def _compat_ideal_pace_challenge():  # type: ignore[misc]
        return RedirectResponse(url="/rate/ideal-pace-challenge/", status_code=307)

    # New pause real-time activity compatibility routes
    @unified.post("/pause-realtime-monitoring/")
    async def _compat_pause_realtime_monitoring():  # type: ignore[misc]
        return RedirectResponse(url="/pause/process-audio-chunk/", status_code=307)

    @unified.post("/pause-improvement-challenge/")
    async def _compat_pause_improvement_challenge():  # type: ignore[misc]
        return RedirectResponse(url="/pause/pause-improvement-challenge/", status_code=307)

    @unified.post("/analyze-pause-session/")
    async def _compat_analyze_pause_session():  # type: ignore[misc]
        return RedirectResponse(url="/pause/analyze-pause-session/", status_code=307)

    @unified.post("/predict-filler-words/")
    async def _compat_predict_filler_words():  # type: ignore[misc]
        return RedirectResponse(url="/filler/predict-filler-words/", status_code=307)

    @unified.post("/predict-loudness/")
    async def _compat_predict_loudness():  # type: ignore[misc]
        return RedirectResponse(url="/loudness/predict-loudness/", status_code=307)

    @unified.get("/")
    def root():  # type: ignore[misc]
        return {
            "status": "ok",
            "services": {
                "pause_management": {
                    "base_path": "/pause",
                    "endpoints": [
                        "/pause/pause-analysis/",
                        "/pause/real-time-analysis/",
                        "/pause/analyze-activity/",
                        "/pause/realtime-monitoring/",
                        "/pause/pause-improvement-challenge/",
                        "/pause/analyze-pause-session/",
                        "/pause/activity-types/",
                        "/pause/test",
                        "/pause/test-features/",
                    ],
                },
                "speech_rate": {
                    "base_path": "/rate",
                    "endpoints": [
                        "/rate/rate-analysis/",
                        "/rate/real-time-analysis/",
                        "/rate/analyze-activity/",
                        "/rate/ideal-pace-challenge/",
                        "/rate/generate-rate-feedback/",
                    ],
                },
                "filler_words": {
                    "base_path": "/filler",
                    "endpoints": [
                        "/filler/predict-filler-words/",
                    ],
                },
                "loudness": {
                    "base_path": "/loudness",
                    "endpoints": [
                        "/loudness/predict-loudness/",
                    ],
                },
            },
        }

    @unified.get("/health")
    def health():  # type: ignore[misc]
        return {"status": "healthy"}

    return unified


app = create_unified_app()


if __name__ == "__main__":
    # Run a single server/port for both services
    uvicorn.run(app, host="0.0.0.0", port=8000)


