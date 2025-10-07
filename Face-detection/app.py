import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import math
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
from tensorflow.keras.models import load_model

# ================== CONFIG (env-overridable) ==================
# Get the absolute path to this file's directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH       = os.getenv("MODEL_PATH", os.path.join(_BASE_DIR, "model/emotion_detection_model.h5"))
UPLOAD_FOLDER    = os.getenv("UPLOAD_FOLDER", os.path.join(_BASE_DIR, "uploads"))
OUTPUT_FOLDER    = os.getenv("OUTPUT_FOLDER", os.path.join(_BASE_DIR, "output"))
SAMPLE_FPS       = float(os.getenv("FER_SAMPLE_FPS", "3"))      # infer this many frames/sec
EMA_ALPHA        = float(os.getenv("FER_EMA_ALPHA", "0.6"))     # smoothing factor
PAD_RATIO        = float(os.getenv("FER_PAD_RATIO", "0.06"))    # % of face box cropped out (borders)
MIN_CONFIDENCE   = float(os.getenv("FER_MIN_CONF", "0.35"))     # gate to accept/update label
MAX_FACES_COUNT  = int(os.getenv("FER_MAX_FACES", "1"))         # we expect 1; >1 triggers warning

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ================== Mediapipe (fallback to Haar) ==================
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    FACE_DET = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    USE_MP = True
except Exception:
    USE_MP = False
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         "haarcascade_frontalface_default.xml")

# ================== TimeDistributed shim so legacy 5-D models load ==================
def _make_maybe_td(base_cls):
    class MaybeTD(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__()
            self._base_kwargs = dict(kwargs)
            self._inner = base_cls(**kwargs)
            self._wrapped = False

        def build(self, input_shape):
            if isinstance(input_shape, (tuple, list)) and len(input_shape) == 5:
                self._inner = tf.keras.layers.TimeDistributed(base_cls(**self._base_kwargs))
                self._wrapped = True
            self._inner.build(input_shape)
            super().build(input_shape)

        def call(self, x, **kw):
            return self._inner(x, **kw)

        def get_config(self):
            cfg = {"_base_class": base_cls.__name__, "_wrapped": self._wrapped}
            cfg.update(self._base_kwargs)
            return cfg
    MaybeTD.__name__ = f"MaybeTD_{base_cls.__name__}"
    return MaybeTD

Conv2D_TD          = _make_maybe_td(tf.keras.layers.Conv2D)
MaxPool2D_TD       = _make_maybe_td(tf.keras.layers.MaxPooling2D)
AvgPool2D_TD       = _make_maybe_td(tf.keras.layers.AveragePooling2D)
BatchNorm_TD       = _make_maybe_td(tf.keras.layers.BatchNormalization)
Dropout_TD         = _make_maybe_td(tf.keras.layers.Dropout)
Dense_TD           = _make_maybe_td(tf.keras.layers.Dense)
Flatten_TD         = _make_maybe_td(tf.keras.layers.Flatten)
Activation_TD      = _make_maybe_td(tf.keras.layers.Activation)
SeparableConv2D_TD = _make_maybe_td(tf.keras.layers.SeparableConv2D)

CUSTOM_OBJECTS = {
    "Conv2D": Conv2D_TD,
    "MaxPooling2D": MaxPool2D_TD,
    "AveragePooling2D": AvgPool2D_TD,
    "BatchNormalization": BatchNorm_TD,
    "Dropout": Dropout_TD,
    "Dense": Dense_TD,
    "Flatten": Flatten_TD,
    "Activation": Activation_TD,
    "SeparableConv2D": SeparableConv2D_TD,
    "TimeDistributed": tf.keras.layers.TimeDistributed,
}

# ================== Model loader (same logic as your test script) ==================
def load_emotion_model(path: str):
    model = load_model(path, compile=False, safe_mode=False, custom_objects=CUSTOM_OBJECTS)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 5:
        _, T, H, W, C = input_shape
        expects_time = True
    elif len(input_shape) == 4:
        _, H, W, C = input_shape
        expects_time = False
    else:
        raise ValueError(f"Unexpected input shape from model: {input_shape}")

    return model, (H, W, C), expects_time

MODEL, IN_SHAPE, EXPECTS_TIME = load_emotion_model(MODEL_PATH)
print(f"[INIT] Loaded model: {MODEL_PATH} | input(H,W,C)={IN_SHAPE} | expects_time={EXPECTS_TIME}")

# ================== Helpers ==================
def ema_update(prev: Optional[np.ndarray], curr: np.ndarray, alpha: float) -> np.ndarray:
    curr = curr.astype(np.float32)
    if prev is None:
        return curr
    return (alpha * prev + (1.0 - alpha) * curr).astype(np.float32)

def detect_faces(bgr_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if bgr_img is None or bgr_img.size == 0:
        return []
    h, w = bgr_img.shape[:2]
    rects = []
    if USE_MP:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        res = FACE_DET.process(rgb)
        if res and res.detections:
            for det in res.detections:
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * w); y = int(box.ymin * h)
                ww = int(box.width * w); hh = int(box.height * h)
                if ww > 0 and hh > 0:
                    rects.append((max(0, x), max(0, y), min(ww, w), min(hh, h)))
    else:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        rects = [(int(x), int(y), int(w_), int(h_)) for (x, y, w_, h_) in faces]
    return rects

def preprocess_face(bgr_crop: np.ndarray, H: int, W: int, C: int) -> np.ndarray:
    if C == 1:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (W, H)).astype("float32") / 255.0
        face = np.expand_dims(face, -1)
    else:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        face = cv2.resize(rgb, (W, H)).astype("float32") / 255.0
    return face

def predict_emotion(x_in: np.ndarray) -> np.ndarray:
    if EXPECTS_TIME:
        x_b = x_in[np.newaxis, np.newaxis, ...]  # (1,1,H,W,C)
    else:
        x_b = x_in[np.newaxis, ...]              # (1,H,W,C)
    probs = MODEL.predict(x_b, verbose=0)
    if EXPECTS_TIME:
        probs = probs[:, -1, :]
    return probs[0].astype(np.float32)

def probs_to_dict(vec: np.ndarray) -> Dict[str, float]:
    return {lab: round(float(p) * 100.0, 1) for lab, p in zip(EMO_LABELS, vec)}

def draw_label(frame, box, label, prob):
    (x, y, w, h) = box
    color = (0, 255, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} ({int(prob * 100)}%)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y0 = max(y - th - 8, 0)
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y0 + th + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y0 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# ========= Coaching helpers =========
POSITIVE = {"Happy": 1.0, "Surprise": 0.6}
NEGATIVE = {"Angry": 0.9, "Disgust": 0.9, "Sad": 0.7, "Fear": 0.7}
NEUTRAL_W = 0.45
SWITCH_NORM = 20  # ~switches per 10 minutes baseline

def shannon_entropy(pcts: dict[str, float]) -> float:
    total = sum(pcts.values()) or 1.0
    probs = [max(1e-9, v / total) for v in pcts.values()]
    return -sum(p * math.log(p, 7) for p in probs)  # normalized [0..1]

def label_switch_rate(emotion_timeline: list[dict], fps_hint: float, sample_step: int) -> float:
    labels = [e.get("fer") for e in emotion_timeline if e.get("face") is True and e.get("fer")]
    if len(labels) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
    seconds_between_samples = max(1e-6, sample_step / (fps_hint or 24.0))
    minutes = len(labels) * seconds_between_samples / 60.0
    return 0.0 if minutes <= 0 else switches / minutes

def engagement_score(fer_pcts, visible_s, away_s, switch_per_min, multi_face) -> int:
    pos = sum(fer_pcts.get(k, 0.0) * v for k, v in POSITIVE.items()) / 100.0
    neg = sum(fer_pcts.get(k, 0.0) * v for k, v in NEGATIVE.items()) / 100.0
    neu = fer_pcts.get("Neutral", 0.0) / 100.0
    vis_ratio = visible_s / max(1.0, (visible_s + away_s))
    stability = 1.0 - min(1.0, switch_per_min / SWITCH_NORM)
    penalty_multi = 0.1 if multi_face else 0.0
    raw = (0.55*pos - 0.35*neg - 0.15*(NEUTRAL_W*neu) + 0.20*vis_ratio + 0.15*stability - penalty_multi)
    return int(max(0, min(100, round(raw * 100))))

def top3_labels(fer_pcts):
    return sorted(fer_pcts.items(), key=lambda kv: kv[1], reverse=True)[:3]

def segments_to_review(emotion_timeline: list[dict], min_span_s=6):
    out = []
    # Neutral spans
    run = []
    for e in emotion_timeline:
        if e.get("face") is True and e.get("fer") == "Neutral":
            run.append(e["time"])
        else:
            if run and run[-1] - run[0] >= min_span_s:
                out.append({"type": "neutral_span", "start": round(run[0], 2), "end": round(run[-1], 2)})
            run = []
    if run and run[-1] - run[0] >= min_span_s:
        out.append({"type": "neutral_span", "start": round(run[0], 2), "end": round(run[-1], 2)})

    # Negative spikes (needs prob >= 0.6)
    for e in emotion_timeline:
        if e.get("face") is True and e.get("fer") in NEGATIVE and (e.get("prob") or 0) >= 0.6:
            out.append({"type": "negative_spike", "time": e["time"], "label": e["fer"], "prob": e["prob"]})
    return out

def build_coaching(fer_pcts, visible_s, away_s, switch_per_min, multi_face, emotion_timeline, start_time, end_time):
    ent = shannon_entropy(fer_pcts)
    eng = engagement_score(fer_pcts, visible_s, away_s, switch_per_min, multi_face)
    t3 = top3_labels(fer_pcts)

    strengths, areas, tips = [], [], []

    if fer_pcts.get("Happy", 0) >= 30:
        strengths.append("Warm presence — strong *Happy* signal.")
    if ent >= 0.55:
        strengths.append("Good facial variety (not monotone).")
    if away_s == 0:
        strengths.append("Consistent camera contact.")

    if fer_pcts.get("Neutral", 0) >= 40:
        areas.append("High Neutral — add more expressive cues (brows, smiles, nods).")
        tips.append("Use intentional pauses + smile at section starts to break neutrality.")
    neg_total = sum(fer_pcts.get(k, 0) for k in NEGATIVE)
    if neg_total >= 15:
        areas.append("Noticeable negative micro-expressions (Angry/Disgust/Fear/Sad).")
        tips.append("Relax jaw/forehead; breathe out before key points.")
    if switch_per_min >= 10:
        areas.append("Expressions change very frequently.")
        tips.append("Hold expressions for ~2–3 seconds to let the audience register them.")
    if away_s > visible_s:
        areas.append("You looked away from camera more than at it.")
        tips.append("Keep eyes near lens when delivering key lines.")
    if multi_face:
        areas.append("Multiple faces detected — record alone for cleaner analysis.")

    moments = segments_to_review(emotion_timeline)
    summary = (
        f"Top 3: {', '.join([f'{k} {v:.1f}%' for k, v in t3])}. "
        f"Engagement score **{eng}%**. "
        f"Face visible {visible_s}s / Away {away_s}s. "
        f"Expression variety (entropy) {ent:.2f}. "
        f"Switches ≈ {switch_per_min:.1f}/min."
    )

    return {
        "summary": summary,
        "score": eng,
        "strengths": strengths,
        "areas_to_improve": areas,
        "tips": tips,
        "moments_to_review": moments,
        "metrics": {
            "entropy": round(ent, 3),
            "switches_per_min": round(switch_per_min, 2),
            "top3": t3,
            "range": [start_time, end_time],
        },
    }
# ========= end helpers =========

# ================== FastAPI ==================
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/video/{filename:path}")
async def get_annotated_video(filename: str):
    from pathlib import Path
    output_dir = Path(OUTPUT_FOLDER).resolve()
    file_path = output_dir / filename
    if not file_path.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(str(file_path))

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    alpha: float = Query(EMA_ALPHA),
    sample_fps: float = Query(SAMPLE_FPS),
    pad: float = Query(PAD_RATIO),
    min_conf: float = Query(MIN_CONFIDENCE),
):
    from pathlib import Path
    
    try:
        # Ensure directories exist (resolve to absolute paths)
        upload_dir = Path(UPLOAD_FOLDER).resolve()
        output_dir = Path(OUTPUT_FOLDER).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return JSONResponse({"error": f"Failed to create directories: {str(e)}"}, status_code=500)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recorded_{timestamp}.webm"
    video_path = upload_dir / video_filename
    
    # Save uploaded video
    try:
        video_data = await video.read()
        video_path.write_bytes(video_data)
        print(f"✅ Received video: {video_path} ({len(video_data)} bytes)")
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return JSONResponse({"error": f"Failed to save video: {str(e)}"}, status_code=500)

    # Try to open video with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ OpenCV cannot open: {video_path}")
        return JSONResponse({"error": "Cannot read video file. Format may not be supported by OpenCV."}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 1000:
        print(f"⚠️ Invalid FPS detected: {fps}, defaulting to 30")
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        print(f"⚠️ Invalid frame count: {total_frames}, will process until end")
        total_frames = 0
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    annotated_name = f"annotated_{timestamp}.mp4"
    annotated_path = output_dir / annotated_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    step = max(1, int(round(fps / max(0.1, sample_fps))))
    print(f"[INFO] OpenCV FPS={fps:.2f}, frames={total_frames}, step={step}, α={alpha}, pad={pad}, min_conf={min_conf}")

    fer_counts = {k: 0 for k in EMO_LABELS}
    timeline = []
    emotion_timeline = []   # [{time, face, fer, prob}]
    multiple_face_error = False
    detected_once = False
    face_visible_seconds = 0
    face_away_seconds = 0

    H, W, C = IN_SHAPE
    last_box = None
    last_label = None
    last_prob = None
    fer_ema_vec = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        do_infer = (frame_idx % step == 0)
        timestamp_sec = round(frame_idx / (fps or 1.0), 2)

        if do_infer:
            faces = detect_faces(frame)

            if len(faces) == 0:
                face_away_seconds += 1
                last_box = None
                last_label = None
                last_prob = None
                timeline.append({"time": timestamp_sec, "status": "Not Visible"})
                emotion_timeline.append({"time": timestamp_sec, "face": False, "fer": None, "prob": None})

            elif len(faces) > MAX_FACES_COUNT:
                multiple_face_error = True
                last_box = max(faces, key=lambda r: r[2] * r[3])
                last_label = "Multiple"
                last_prob = None
                timeline.append({"time": timestamp_sec, "status": "Multiple Faces"})
                emotion_timeline.append({"time": timestamp_sec, "face": "multiple", "fer": None, "prob": None})

            else:
                detected_once = True
                face_visible_seconds += 1
                timeline.append({"time": timestamp_sec, "status": "Visible"})

                (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                pad_px = int(pad * min(w, h))
                x0 = max(x + pad_px, 0); y0 = max(y + pad_px, 0)
                x1 = min(x + w - pad_px, frame.shape[1]); y1 = min(y + h - pad_px, frame.shape[0])

                crop = frame[y0:y1, x0:x1]
                if crop.size > 0:
                    xin = preprocess_face(crop, H, W, C)
                    probs = predict_emotion(xin)               # raw model probs

                    # Confidence gate: only update EMA if we're confident
                    if float(np.max(probs)) >= min_conf:
                        fer_ema_vec = ema_update(fer_ema_vec, probs, alpha)
                    # If EMA is still None (first weak detections), fall back to current probs
                    vec = fer_ema_vec if fer_ema_vec is not None else probs
                    idx = int(np.argmax(vec))
                    fer_label = EMO_LABELS[idx]
                    fer_prob  = float(vec[idx])

                    last_box = (x0, y0, x1 - x0, y1 - y0)
                    last_label = fer_label
                    last_prob = fer_prob

                    # Count only confident frames
                    if float(np.max(probs)) >= min_conf:
                        fer_counts[fer_label] += 1

                    emotion_timeline.append(
                        {"time": timestamp_sec, "face": True, "fer": fer_label, "prob": round(fer_prob, 4)}
                    )
                else:
                    emotion_timeline.append({"time": timestamp_sec, "face": True, "fer": None, "prob": None})

            if frame_idx % (step * 15) == 0:
                print(f"[ANALYZE] t={timestamp_sec}s, faces={len(faces) if do_infer else 0}")

        # Draw overlay on every frame
        overlay = frame.copy()
        if last_box:
            draw_label(overlay, last_box, last_label or "No face", last_prob if last_prob else 0.0)
        else:
            cv2.putText(overlay, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)

        writer.write(overlay)
        frame_idx += 1

    try:
        cap.release()
        writer.release()
    except Exception as e:
        print(f"⚠️ Error releasing video resources: {e}")

    # --------- Advanced coaching / metrics ----------
    try:
        total_vis = sum(fer_counts.values()) or 1
        fer_percent = {k: round(v / total_vis * 100.0, 1) for k, v in fer_counts.items()}
        start_time = timeline[0]["time"] if timeline else 0
        end_time = timeline[-1]["time"] if timeline else 0

        switch_rate = label_switch_rate(emotion_timeline, fps_hint=fps, sample_step=step)
        coaching = build_coaching(
            fer_percent,
            face_visible_seconds,
            face_away_seconds,
            switch_rate,
            multiple_face_error,
            emotion_timeline,
            start_time,
            end_time,
        )

        print(f"📤 Analysis complete. Annotated video: {annotated_path}")

        return {
            "face_detected": bool(detected_once),
            "multiple_faces": bool(multiple_face_error),
            "fer_emotions": fer_percent,
            "face_timeline": timeline,
            "emotion_timeline": emotion_timeline,  # includes probs for richer charts
            "start_time": start_time,
            "end_time": end_time,
            "visible_seconds": int(face_visible_seconds),
            "away_seconds": int(face_away_seconds),

            # Use coaching tips as suggestions for the UI
            "suggestions": coaching["tips"],
            # Full coaching payload (score, summary, strengths, areas, tips, moments_to_review, metrics)
            "coaching": coaching,

            "annotated_video": annotated_name,
            "annotated_url": f"/video/{annotated_name}",
            "params": {"alpha": alpha, "sample_fps": sample_fps, "pad": pad, "min_conf": min_conf},
        }
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)

@app.post("/analyze_frame")
async def analyze_frame(
    frame: UploadFile = File(...),
    alpha: float = Query(EMA_ALPHA),
    pad: float = Query(PAD_RATIO),
    min_conf: float = Query(MIN_CONFIDENCE),
):
    """
    Multipart form: 'frame' (jpg/png)
    Optional query params: alpha, pad, min_conf
    Returns: { face_detected, top_emotion, prob, probs{label:percent} }
    """
    data = await frame.read()
    if not data:
        return {"face_detected": False, "top_emotion": None, "probs": {}}

    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return {"face_detected": False, "top_emotion": None, "probs": {}}

    faces = detect_faces(img)
    if len(faces) == 0:
        return {"face_detected": False, "top_emotion": None, "probs": {}}

    (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
    pad_px = int(pad * min(w, h))
    x0 = max(x + pad_px, 0); y0 = max(y + pad_px, 0)
    x1 = min(x + w - pad_px, img.shape[1]); y1 = min(y + h - pad_px, img.shape[0])

    crop = img[y0:y1, x0:x1]
    H, W, C = IN_SHAPE
    xin = preprocess_face(crop, H, W, C)
    probs = predict_emotion(xin)

    # EMA across single call doesn't persist; return raw + gated top
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    if top_prob < min_conf:
        top_label = "Uncertain"
    else:
        top_label = EMO_LABELS[top_idx]

    return {
        "face_detected": True,
        "top_emotion": top_label,
        "prob": round(top_prob, 4),
        "probs": probs_to_dict(probs),
        "params": {"alpha": alpha, "pad": pad, "min_conf": min_conf}
    }

if __name__ == "__main__":
    import uvicorn
    # On Windows we avoid the reloader to keep the model from reloading twice.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
