import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- Face detection (MediaPipe -> fallback to Haar) ----------------
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    FACE_DET = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    USE_MP = True
except Exception:
    USE_MP = False
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ---------------- TimeDistributed shim for legacy 5-D models ----------------
def make_maybe_time_distributed(base_cls):
    class MaybeTD(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__()
            # create the "inner" base layer using the same kwargs the original layer had
            self._base_kwargs = dict(kwargs)
            self._inner = base_cls(**kwargs)
            self._wrapped = False

        def build(self, input_shape):
            # If a 5D input appears, wrap the inner layer with TimeDistributed on-the-fly
            if isinstance(input_shape, (tuple, list)) and len(input_shape) == 5:
                self._inner = tf.keras.layers.TimeDistributed(base_cls(**self._base_kwargs))
                self._wrapped = True
            self._inner.build(input_shape)
            super().build(input_shape)

        def call(self, x, **kw):
            return self._inner(x, **kw)

        def compute_output_shape(self, input_shape):
            try:
                return self._inner.compute_output_shape(input_shape)
            except Exception:
                # Fallback – not strictly required in TF2 eager
                return tf.TensorShape(self._inner.compute_output_signature(input_shape).shape)

        def get_config(self):
            # Keep the original layer's config so the object remains re-serializable
            cfg = {"_base_class": base_cls.__name__, "_wrapped": self._wrapped}
            cfg.update(self._base_kwargs)
            return cfg
    MaybeTD.__name__ = f"MaybeTD_{base_cls.__name__}"
    return MaybeTD

# Register shims for common layers that appear in CNNs
Conv2D_TD          = make_maybe_time_distributed(tf.keras.layers.Conv2D)
MaxPool2D_TD       = make_maybe_time_distributed(tf.keras.layers.MaxPooling2D)
AvgPool2D_TD       = make_maybe_time_distributed(tf.keras.layers.AveragePooling2D)
BatchNorm_TD       = make_maybe_time_distributed(tf.keras.layers.BatchNormalization)
Dropout_TD         = make_maybe_time_distributed(tf.keras.layers.Dropout)
Dense_TD           = make_maybe_time_distributed(tf.keras.layers.Dense)
Flatten_TD         = make_maybe_time_distributed(tf.keras.layers.Flatten)
Activation_TD      = make_maybe_time_distributed(tf.keras.layers.Activation)
SeparableConv2D_TD = make_maybe_time_distributed(tf.keras.layers.SeparableConv2D)

CUSTOM_OBJECTS = {
    # When the loader sees "Conv2D" in the H5 config, instantiate our shim instead.
    "Conv2D": Conv2D_TD,
    "MaxPooling2D": MaxPool2D_TD,
    "AveragePooling2D": AvgPool2D_TD,
    "BatchNormalization": BatchNorm_TD,
    "Dropout": Dropout_TD,
    "Dense": Dense_TD,
    "Flatten": Flatten_TD,
    "Activation": Activation_TD,
    "SeparableConv2D": SeparableConv2D_TD,
    # In case the original actually used TimeDistributed explicitly
    "TimeDistributed": tf.keras.layers.TimeDistributed,
}

# ---------------- Model loader ----------------
def load_emotion_model(path):
    """
    Loads legacy .h5 models that may contain plain 2D layers but expect 5D (time) inputs.
    Returns (model, (H,W,C), expects_time)
    """
    model = load_model(path, compile=False, safe_mode=False, custom_objects=CUSTOM_OBJECTS)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 5:
        # (None, T, H, W, C)
        _, T, H, W, C = input_shape
        expects_time = True
    elif len(input_shape) == 4:
        # (None, H, W, C)
        _, H, W, C = input_shape
        expects_time = False
    else:
        raise ValueError(f"Unexpected input shape from model: {input_shape}")

    return model, (H, W, C), expects_time

# ---------------- Utilities ----------------
def ema_update(prev_vec, new_vec, alpha=0.6):
    if prev_vec is None:
        return new_vec.astype(np.float32)
    return (alpha * prev_vec + (1.0 - alpha) * new_vec).astype(np.float32)

def detect_faces(bgr):
    h, w = bgr.shape[:2]
    rects = []
    if USE_MP:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = FACE_DET.process(rgb)
        if res and res.detections:
            for det in res.detections:
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * w); y = int(box.ymin * h)
                ww = int(box.width * w); hh = int(box.height * h)
                x, y = max(0, x), max(0, y)
                ww, hh = max(0, ww), max(0, hh)
                if ww > 0 and hh > 0:
                    rects.append((x, y, ww, hh))
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        rects = [(int(x), int(y), int(w_), int(h_)) for (x, y, w_, h_) in faces]
    return rects

def preprocess_face(bgr_crop, H, W, C):
    if C == 1:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (W, H)).astype("float32") / 255.0
        face = np.expand_dims(face, -1)
    else:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        face = cv2.resize(rgb, (W, H)).astype("float32") / 255.0
    return face

def predict_emotion(model, x, expects_time):
    if expects_time:
        x = x[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, C)
    else:
        x = x[np.newaxis, ...]              # (1, H, W, C)
    probs = model.predict(x, verbose=0)
    if expects_time:
        probs = probs[:, -1, :]            # last timestep
    return probs[0]

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

# ---------------- Runners ----------------
def run_webcam(model, in_shape, expects_time, cam_index=0):
    H, W, C = in_shape
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    print("[INFO] Webcam started. Press 'q' to quit.")
    fer_ema = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect_faces(frame)
        if faces:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            pad = int(0.06 * min(w, h))
            x0 = max(x + pad, 0); y0 = max(y + pad, 0)
            x1 = min(x + w - pad, frame.shape[1]); y1 = min(y + h - pad, frame.shape[0])
            crop = frame[y0:y1, x0:x1]
            if crop.size > 0:
                xin = preprocess_face(crop, H, W, C)
                probs = predict_emotion(model, xin, expects_time)
                fer_ema = ema_update(fer_ema, probs, 0.6)
                pred = int(np.argmax(fer_ema))
                draw_label(frame, (x0, y0, x1 - x0, y1 - y0), EMO_LABELS[pred], float(fer_ema[pred]))
        else:
            cv2.putText(frame, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)
        cv2.imshow("Emotion Webcam (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_image(model, in_shape, expects_time, path):
    if not os.path.isfile(path):
        print(f"ERROR: Image not found: {path}"); return
    img = cv2.imread(path)
    if img is None:
        print("ERROR: Could not read image."); return
    H, W, C = in_shape
    faces = detect_faces(img)
    if faces:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(0.06 * min(w, h))
        x0 = max(x + pad, 0); y0 = max(y + pad, 0)
        x1 = min(x + w - pad, img.shape[1]); y1 = min(y + h - pad, img.shape[0])
        crop = img[y0:y1, x0:x1]
        xin = preprocess_face(crop, H, W, C)
        probs = predict_emotion(model, xin, expects_time)
        pred = int(np.argmax(probs))
        draw_label(img, (x0, y0, x1 - x0, y1 - y0), EMO_LABELS[pred], float(probs[pred]))
    else:
        cv2.putText(img, "No face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.imshow("Emotion Image - Press any key", img)
    cv2.waitKey(0); cv2.destroyAllWindows()

def run_video(model, in_shape, expects_time, path):
    if not os.path.isfile(path):
        print(f"ERROR: Video not found: {path}"); return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("ERROR: Could not open video."); return
    print("[INFO] Playing video. Press 'q' to quit.")
    H, W, C = in_shape; fer_ema = None
    while True:
        ok, frame = cap.read()
        if not ok: break
        faces = detect_faces(frame)
        if faces:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            pad = int(0.06 * min(w, h))
            x0 = max(x + pad, 0); y0 = max(y + pad, 0)
            x1 = min(x + w - pad, frame.shape[1]); y1 = min(y + h - pad, frame.shape[0])
            crop = frame[y0:y1, x0:x1]
            xin = preprocess_face(crop, H, W, C)
            probs = predict_emotion(model, xin, expects_time)
            fer_ema = ema_update(fer_ema, probs, 0.6)
            pred = int(np.argmax(fer_ema))
            draw_label(frame, (x0, y0, x1 - x0, y1 - y0), EMO_LABELS[pred], float(fer_ema[pred]))
        else:
            cv2.putText(frame, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)
        cv2.imshow("Emotion Video (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the .h5 model")
    ap.add_argument("--mode", required=True, choices=["webcam", "image", "video"])
    ap.add_argument("--input", help="Path to image/video if mode != webcam")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    args = ap.parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model, in_shape, expects_time = load_emotion_model(args.model)
    print(f"[INFO] Model input (H,W,C)={in_shape}, expects_time={expects_time}")

    if args.mode == "webcam":
        run_webcam(model, in_shape, expects_time, cam_index=args.cam)
    elif args.mode == "image":
        if not args.input:
            print("ERROR: --input path is required for mode=image"); return
        run_image(model, in_shape, expects_time, args.input)
    else:
        if not args.input:
            print("ERROR: --input path is required for mode=video"); return
        run_video(model, in_shape, expects_time, args.input)

if __name__ == "__main__":
    main()
