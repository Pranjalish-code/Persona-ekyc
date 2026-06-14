# livecheck.py
# Liveness check: Static (LBP+FFT) + Motion + rPPG (POS) + Blink+Mouth minimum gate (offline)
# NOTE: Random 3s challenge is handled in app.py real-time.

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


# =========================
# Utils
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)


# =========================
# Face detection (OpenCV Haar)
# =========================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_bbox(gray, min_size=(80, 80)) -> Optional[Tuple[int, int, int, int]]:
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def smooth_bbox(prev, curr, alpha=0.7):
    if prev is None:
        return curr
    px, py, pw, ph = prev
    cx, cy, cw, ch = curr
    x = int(alpha * px + (1 - alpha) * cx)
    y = int(alpha * py + (1 - alpha) * cy)
    w = int(alpha * pw + (1 - alpha) * cw)
    h = int(alpha * ph + (1 - alpha) * ch)
    return x, y, w, h


def crop_face(frame_bgr, bbox, out_size=160, pad=0.25):
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + pad)

    x0 = int(clamp(cx - side / 2, 0, w - 1))
    y0 = int(clamp(cy - side / 2, 0, h - 1))
    x1 = int(clamp(cx + side / 2, 0, w - 1))
    y1 = int(clamp(cy + side / 2, 0, h - 1))

    face = frame_bgr[y0:y1, x0:x1]
    if face.size == 0:
        return None
    face = cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return face


# =========================
# Static: LBP + FFT
# =========================
def lbp_hist(gray, R=1):
    H, W = gray.shape
    g = gray.astype(np.int32)
    center = g[R:H - R, R:W - R]
    code = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neigh = g[R + dy:H - R + dy, R + dx:W - R + dx]
        code |= ((neigh >= center) << i).astype(np.uint8)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def fft_screen_score(gray):
    g = gray.astype(np.float32)
    g = g - g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(F))
    H, W = mag.shape
    cy, cx = H // 2, W // 2

    r0 = int(0.06 * min(H, W))
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    ring1 = (R > r0) & (R < 0.35 * min(H, W))
    ring2 = (R >= 0.35 * min(H, W)) & (R < 0.48 * min(H, W))

    e2 = mag[ring2].mean() if ring2.any() else 0.0
    eT = mag.mean() + 1e-8

    flat = mag[ring1].ravel()
    if flat.size > 50:
        k = max(10, int(0.001 * flat.size))
        topk = np.partition(flat, -k)[-k:]
        peak_ratio = (topk.mean() / (flat.mean() + 1e-8))
    else:
        peak_ratio = 1.0

    screen_like = 0.6 * (e2 / eT) + 0.4 * (peak_ratio / 3.0)
    return float(screen_like), {
        "fft_e2_over_total": float(e2 / eT),
        "peak_ratio": float(peak_ratio),
    }


def static_texture_features(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray_fft = cv2.GaussianBlur(gray, (3, 3), 0)  # FFT stability only

    hist = lbp_hist(gray)
    lbp_entropy = float(-np.sum(hist * np.log(hist + 1e-8)))

    screen_score, fft_info = fft_screen_score(gray_fft)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    hf = float(np.mean(np.abs(lap)))
    return {
        "lbp_entropy": lbp_entropy,
        "hf_laplacian_mean": hf,
        "screen_fft_score": float(screen_score),
        **fft_info,
    }


# =========================
# Temporal: motion + replay cadence
# =========================
def temporal_motion_features(
    face_grays: List[np.ndarray],
    centers: List[Tuple[float, float]],
    face_ws: List[float],
) -> Dict[str, float]:
    if len(centers) < 5:
        return {"jitter_med_norm": 0.0, "jitter_std_norm": 0.0, "dup_frame_rate": 0.0}

    c = np.array(centers, dtype=np.float32)
    v = np.diff(c, axis=0)
    speed = np.sqrt((v[:, 0] ** 2 + v[:, 1] ** 2))

    wmed = float(np.median(face_ws)) if len(face_ws) else 160.0
    jitter_med = float(np.median(speed) / (wmed + 1e-6))
    jitter_std = float(np.std(speed) / (wmed + 1e-6))

    diffs = []
    for i in range(1, len(face_grays)):
        a = face_grays[i - 1].astype(np.float32)
        b = face_grays[i].astype(np.float32)
        diffs.append(float(np.mean(np.abs(a - b))))
    diffs = np.array(diffs, dtype=np.float32)
    dup_rate = float(np.mean(diffs < 1.2)) if len(diffs) else 0.0

    return {
        "jitter_med_norm": jitter_med,
        "jitter_std_norm": jitter_std,
        "dup_frame_rate": dup_rate,
    }


# =========================
# rPPG: POS method + quality
# =========================
def bandpass_fft(x, fs, f_lo=0.7, f_hi=3.0):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    X = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    Xf = np.zeros_like(X)
    Xf[mask] = X[mask]
    y = np.fft.irfft(Xf, n=n)
    return y, freqs, np.abs(X), mask


def _pos_from_roi(face_bgr_seq: List[np.ndarray], fs: float, x0: int, x1: int):
    rgb = []
    for f in face_bgr_seq:
        roi = f[:, x0:x1]
        b, g, r = cv2.split(roi)
        rgb.append([np.mean(r), np.mean(g), np.mean(b)])
    rgb = np.array(rgb, dtype=np.float32)
    rgb = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-6)

    X = rgb.T
    S1 = X[1] - X[2]
    S2 = X[1] + X[2] - 2 * X[0]
    alpha = (np.std(S1) / (np.std(S2) + 1e-6))
    h = S1 - alpha * S2
    h = zscore(h)

    y, freqs, mag, mask = bandpass_fft(h, fs, 0.7, 3.0)
    return y, freqs, mag, mask


def pos_rppg(face_bgr_seq: List[np.ndarray], fs: float):
    if len(face_bgr_seq) < 30:
        return None

    W = face_bgr_seq[0].shape[1]
    mid = W // 2

    y_left, freqs, _, mask = _pos_from_roi(face_bgr_seq, fs, 0, mid)
    y_right, _, _, _ = _pos_from_roi(face_bgr_seq, fs, mid, W)

    if np.std(y_left) < 1e-6 or np.std(y_right) < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_left, y_right)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    y = 0.5 * (y_left + y_right)
    y_n = zscore(y)
    n = len(y_n)
    X = np.fft.rfft(y_n * np.hanning(n))
    mag = np.abs(X)
    mag_bp = mag[mask]
    freqs_bp = freqs[mask]
    if len(mag_bp) < 5:
        return None

    peak_i = int(np.argmax(mag_bp))
    peak_f = float(freqs_bp[peak_i])
    bpm = float(peak_f * 60.0)

    peak_mag = float(mag_bp[peak_i])
    noise_mag = float(np.median(mag_bp) + 1e-8)
    peak_ratio = peak_mag / noise_mag
    snr_db = 20.0 * np.log10(peak_mag / (noise_mag + 1e-8))

    return {
        "bpm": bpm,
        "peak_ratio": float(peak_ratio),
        "snr_db": float(snr_db),
        "corr_lr": float(corr),
    }


# =========================
# Offline blink + mouth minimum gate (FaceMesh)
# =========================
try:
    import mediapipe as mp
    _MP_OK = True
except Exception:
    mp = None
    _MP_OK = False


def _euclid(p, q):
    return float(np.linalg.norm(np.array(p, dtype=np.float32) - np.array(q, dtype=np.float32)))


def _ear(lm_xy, idx6):
    p1, p2, p3, p4, p5, p6 = [lm_xy[i] for i in idx6]
    num = _euclid(p2, p6) + _euclid(p3, p5)
    den = 2.0 * _euclid(p1, p4) + 1e-6
    return num / den


def _mar(lm_xy, idx4):
    left, right, upper, lower = [lm_xy[i] for i in idx4]
    num = _euclid(upper, lower)
    den = _euclid(left, right) + 1e-6
    return num / den


def blink_mouth_metrics(
    face_crops: List[np.ndarray],
    ear_thr: float = 0.21,
    ear_consec: int = 2,
    mar_thr: float = 0.55,
    mar_consec: int = 2,
) -> Dict[str, float]:
    if not _MP_OK:
        return {"mp_ok": False, "blink_count": 0, "mouth_open_count": 0, "ear_median": 0.0, "mar_median": 0.0}

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [61, 291, 13, 14]

    ears, mars = [], []
    blink_count = 0
    mouth_open_count = 0

    eye_closed_run = 0
    mouth_open_run = 0
    was_eye_closed = False
    was_mouth_open = False

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for face in face_crops:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue

            h, w = face.shape[:2]
            pts = res.multi_face_landmarks[0].landmark
            lm_xy = [(int(p.x * w), int(p.y * h)) for p in pts]

            ear_val = 0.5 * (_ear(lm_xy, LEFT_EYE) + _ear(lm_xy, RIGHT_EYE))
            mar_val = _mar(lm_xy, MOUTH)
            ears.append(float(ear_val))
            mars.append(float(mar_val))

            if ear_val < ear_thr:
                eye_closed_run += 1
            else:
                if eye_closed_run >= ear_consec and not was_eye_closed:
                    blink_count += 1
                    was_eye_closed = True
                eye_closed_run = 0
                was_eye_closed = False

            if mar_val > mar_thr:
                mouth_open_run += 1
            else:
                if mouth_open_run >= mar_consec and not was_mouth_open:
                    mouth_open_count += 1
                    was_mouth_open = True
                mouth_open_run = 0
                was_mouth_open = False

    return {
        "mp_ok": True,
        "blink_count": int(blink_count),
        "mouth_open_count": int(mouth_open_count),
        "ear_median": float(np.median(ears)) if ears else 0.0,
        "mar_median": float(np.median(mars)) if mars else 0.0,
    }


# =========================
# Output structure
# =========================
@dataclass
class LivenessResult:
    decision: str
    score: float
    components: Dict[str, float]
    info: Dict[str, float]


# =========================
# Main callable
# =========================
def analyze_video(
    video_path: str,
    max_seconds: float = 8.5,
    sample_fps: float = 12.0,
    face_size: int = 160,
) -> LivenessResult:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    step = max(1, int(round(fps / sample_fps)))
    face_crops, face_grays, centers, face_ws, static_feats_list = [], [], [], [], []

    prev_bbox = None
    frame_idx = 0
    max_frames = int(max_seconds * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx > max_frames:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = detect_face_bbox(gray)
        if bbox is None:
            frame_idx += 1
            continue

        bbox = smooth_bbox(prev_bbox, bbox, alpha=0.75)
        prev_bbox = bbox
        x, y, w, h = bbox

        centers.append((x + w / 2.0, y + h / 2.0))
        face_ws.append(float(w))

        face = crop_face(frame, bbox, out_size=face_size, pad=0.25)
        if face is None:
            frame_idx += 1
            continue

        face_crops.append(face)
        face_grays.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))

        if len(face_crops) in [1, max(2, len(face_crops) // 2), max(3, len(face_crops) - 1)]:
            static_feats_list.append(static_texture_features(face))

        frame_idx += 1

    cap.release()

    if len(face_crops) < 20:
        return LivenessResult(
            decision="FAIL_NO_FACE",
            score=0.0,
            components={"P_static": 0.0, "P_motion": 0.0, "P_rppg": 0.0},
            info={"reason": "not_enough_face_frames", "frames_used": int(len(face_crops))},
        )

    eff_fs = float(sample_fps)

    if len(static_feats_list) == 0:
        static_feats_list = [static_texture_features(face_crops[len(face_crops) // 2])]

    lbp_ent = float(np.mean([d["lbp_entropy"] for d in static_feats_list]))
    hf_lap = float(np.mean([d["hf_laplacian_mean"] for d in static_feats_list]))
    screen_fft = float(np.mean([d["screen_fft_score"] for d in static_feats_list]))
    fft_peak_ratio = float(np.mean([d["peak_ratio"] for d in static_feats_list]))
    P_screen = float(sigmoid((0.55 - screen_fft) / 0.08))

    static_raw = (
        1.3 * (lbp_ent - 3.9)
        + 0.60 * (hf_lap - 3.0)
        - 0.9 * (screen_fft - 0.55)
        - 0.12 * (fft_peak_ratio - 2.0)
    )
    P_static = float(sigmoid(static_raw))

    tm = temporal_motion_features(face_grays, centers, face_ws)
    jitter = float(tm["jitter_med_norm"])
    dup_rate = float(tm["dup_frame_rate"])

    motion_raw = 1.8 * (0.06 - abs(jitter - 0.03)) - 3.0 * (dup_rate - 0.10)
    P_motion = float(sigmoid(motion_raw))

    r = pos_rppg(face_crops, fs=eff_fs)
    if r is None:
        P_rppg, Q_rppg = 0.0, 0.0
        bpm, snr_db, peak_r, corr_lr = 0.0, -999.0, 0.0, 0.0
    else:
        bpm = float(r["bpm"])
        snr_db = float(r["snr_db"])
        peak_r = float(r["peak_ratio"])
        corr_lr = float(r.get("corr_lr", 0.0))

        plausible = 1.0 if (40.0 <= bpm <= 180.0) else 0.0
        Q_base = sigmoid((snr_db - 6.0) / 2.5) * sigmoid((peak_r - 2.0) / 0.8) * plausible
        P_base = sigmoid((snr_db - 7.0) / 2.0) * sigmoid((peak_r - 2.2) / 0.7) * plausible

        coh = sigmoid((corr_lr - 0.30) / 0.15)
        Q_rppg = float(Q_base * coh)
        P_rppg = float(P_base * coh)

    # Fusion
    q_min = 0.3
    if Q_rppg >= q_min:
        score = 0.30 * P_static + 0.40 * P_motion + 0.30 * P_rppg 
        mode = "with_rppg"
    else:
        score = 0.50 * P_static + 0.50 * P_motion 
        mode = "no_rppg"

    # Offline minimum action gate (not the random challenge)
    bm = blink_mouth_metrics(face_crops)
    min_blinks = 0
    min_mouth_opens = 0

    if bm["mp_ok"]:
        blink_ok = bm["blink_count"] >= min_blinks
        mouth_ok = bm["mouth_open_count"] >= min_mouth_opens
    else:
        blink_ok = True
        mouth_ok = True

    tau_live = 0.50
    base_live = (score >= tau_live)
    decision = "LIVE" if (base_live ) else "SPOOF"

    spoof_hint = "unknown"
    if decision == "SPOOF" and bm["mp_ok"]:
        if (not blink_ok) and (not mouth_ok):
            spoof_hint = "no_blink_and_no_mouth_open"
        elif not blink_ok:
            spoof_hint = "no_blink_detected"
        elif not mouth_ok:
            spoof_hint = "no_mouth_open_detected"

    if decision == "SPOOF" and spoof_hint == "unknown":
        if (screen_fft > 0.70) and (P_static < 0.35):
            spoof_hint = "replay_screen_likely"
        elif P_static > 0.9:
            spoof_hint = "AI_likely"
        elif (dup_rate > 0.22) and (screen_fft > 0.60):
            spoof_hint = "replay_or_stutter_likely"
        elif hf_lap < 2.0:
            spoof_hint = "print_or_blur_likely"
        elif Q_rppg < 0.15:
            spoof_hint = "no_rppg_signal_likely"
        else:
            spoof_hint = "inconsistent_motion_texture"

    return LivenessResult(
        decision=decision,
        score=float(score),
        components={"P_static": P_static, "P_motion": P_motion, "P_rppg": P_rppg},
        info={
            "mode": mode,
            "lbp_entropy": lbp_ent,
            "hf_laplacian_mean": hf_lap,
            "screen_fft_score": screen_fft,
            "fft_peak_ratio": fft_peak_ratio,
            "jitter_med_norm": jitter,
            "dup_frame_rate": dup_rate,
            "bpm": bpm,
            "snr_db": snr_db,
            "rppg_peak_ratio": peak_r,
            "corr_lr": corr_lr,
            "Q_rppg": Q_rppg,
            "spoof_hint": spoof_hint,
            "frames_used": int(len(face_crops)),
            "sample_fps": float(eff_fs),
            "tau_live": float(tau_live),
            "q_min": float(q_min),
            "mp_facemesh": bool(bm["mp_ok"]),
            "blink_count": int(bm["blink_count"]),
            "mouth_open_count": int(bm["mouth_open_count"]),
            "ear_median": float(bm["ear_median"]),
            "mar_median": float(bm["mar_median"]),
            "min_blinks": int(min_blinks),
            "min_mouth_opens": int(min_mouth_opens),
        },
    )
