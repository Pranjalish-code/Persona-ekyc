# arcface.py
import os
import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# ---------------------------
# 1) Init model
# ---------------------------
_face_app = None

def get_face_app(det_size=(640, 640), providers=None, ctx_id=0):
    """
    Lazy init InsightFace (RetinaFace + ArcFace).
    For Streamlit Cloud / CPU-only: providers=["CPUExecutionProvider"]
    For GPU server: providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    """
    global _face_app
    if _face_app is None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers
        )
        _face_app.prepare(ctx_id=ctx_id, det_size=det_size)
    return _face_app

# ---------------------------
# 2) Robust image reader
# ---------------------------
def read_image_any(input_data):
    """
    Accepts:
    - file path (str)
    - bytes / bytearray (uploaded file bytes)
    - numpy array (BGR/RGB)

    Returns:
    - BGR np.ndarray
    """
    if isinstance(input_data, str):
        # robust read for weird filenames / jpeg
        try:
            data = np.fromfile(input_data, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = cv2.imread(input_data)

        if img is None:
            raise ValueError(f"Image not readable: {input_data}")
        return img

    if isinstance(input_data, (bytes, bytearray)):
        arr = np.frombuffer(input_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Uploaded image bytes could not be decoded.")
        return img

    if isinstance(input_data, np.ndarray):
        img = input_data
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected color image (H, W, 3).")
        return img

    raise TypeError("Unsupported input type for image.")

# ---------------------------
# 3) Embedding + similarity
# ---------------------------
def get_embedding(
    input_data,
    det_size=(640, 640),
    providers=None,
    ctx_id=0,
    pick="best"
):
    """
    Returns ArcFace embedding (512-d) or None.
    pick:
      - "best": max(det_score, area)
      - "largest": max(area)
    """
    app = get_face_app(det_size=det_size, providers=providers, ctx_id=ctx_id)
    img = read_image_any(input_data)

    faces = app.get(img)
    if not faces:
        return None

    if pick == "largest":
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    else:
        faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
        return faces[0].embedding

    emb = face.embedding
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)

def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return None
    return float(np.dot(emb1, emb2) / (norm(emb1) * norm(emb2) + 1e-8))

def verify_face_match(
    id_input,
    selfie_input,
    threshold=0.35,
    det_size=(640, 640),
    providers=None
):
    """
    Convenience wrapper:
    returns dict with embeddings status + similarity + decision
    """
    id_emb = get_embedding(id_input, det_size=det_size, providers=providers)
    sf_emb = get_embedding(selfie_input, det_size=det_size, providers=providers)

    sim = cosine_similarity(id_emb, sf_emb)
    if sim is None:
        return {
            "ok": False,
            "similarity": None,
            "decision": "NO_FACE",
            "reason": "Face not detected in ID or selfie/video frame."
        }

    decision = "MATCH" if sim >= threshold else "NO_MATCH"
    return {
        "ok": True,
        "similarity": sim,
        "threshold": threshold,
        "decision": decision
    }

# ---------------------------
# 4) Optional: choose threshold by FAR (simple)
# ---------------------------
def threshold_from_scores(scores, labels, target_far=1e-3):
    """
    Utility for offline calibration (NOT needed for deployment).
    Given genuine/impostor scores+labels, choose threshold for FAR target.
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    # impostor scores = label 0
    imp = scores[labels == 0]
    if len(imp) == 0:
        raise ValueError("No impostor scores (label=0) provided.")

    # FAR = fraction of impostors >= threshold
    # Want FAR <= target_far -> threshold at (1 - target_far) quantile of impostors
    thr = float(np.quantile(imp, 1.0 - target_far))
    return thr
