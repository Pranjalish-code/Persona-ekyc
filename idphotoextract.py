# idphotoextract.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------
# 1) Model init
# ---------------------------
_face_app = None

def get_face_app(det_size=(1280, 1280)):
    """
    Lazy init FaceAnalysis once (important for Streamlit performance).
    """
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=det_size)
    return _face_app

# ---------------------------
# 2) Helpers
# ---------------------------
def rotate_image(img_bgr, angle):
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr

def read_image_any(input_data):
    """
    Accepts:
    - file path (str)
    - bytes (uploaded file bytes)
    - numpy array (BGR or RGB)
    Returns: BGR image (np.ndarray)
    """
    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            raise ValueError("Image path unreadable / not found.")
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
            raise ValueError("Expected a color image (H,W,3).")
        return img

    raise TypeError("Unsupported input type for image.")

# ---------------------------
# 3) Main function
# ---------------------------
def crop_face_from_id_robust(
    input_data,
    margin=0.30,
    det_size=(1280, 1280),
    upscale_if_small=True,
    min_side_for_upscale=900,
    upscale_factor=1.5,
    return_debug=False
):
    """
    Streamlit-ready robust face crop.

    input_data: path OR bytes OR numpy image
    margin: extra padding around detected face bbox
    det_size: detector size for insightface
    return_debug: if True returns (crop, debug_dict)

    Returns:
        crop_bgr (np.ndarray) OR (crop_bgr, debug)
    """
    app = get_face_app(det_size=det_size)

    img0 = read_image_any(input_data)
    if img0 is None:
        raise ValueError("Image not found or unreadable")

    best_face = None
    best_img = None
    used_angle = None

    # Try multiple rotations (IDs often rotated)
    for angle in [0, 90, 180, 270]:
        img = rotate_image(img0, angle)

        h, w = img.shape[:2]
        if upscale_if_small and max(h, w) < min_side_for_upscale:
            img = cv2.resize(img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

        faces = app.get(img)
        if not faces:
            continue

        # Choose largest face by area
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        best_face = face
        best_img = img
        used_angle = angle
        break

    # Fallback: center crop
    if best_face is None:
        h, w = img0.shape[:2]
        cx1, cy1 = int(w * 0.2), int(h * 0.2)
        cx2, cy2 = int(w * 0.8), int(h * 0.8)
        crop = img0[cy1:cy2, cx1:cx2]
        if return_debug:
            return crop, {
                "mode": "fallback_center_crop",
                "angle": None,
                "bbox": None
            }
        return crop

    img = best_img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, best_face.bbox)

    # Add margin
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    crop = img[y1:y2, x1:x2]

    if return_debug:
        return crop, {
            "mode": "face_detected",
            "angle": used_angle,
            "bbox": [x1, y1, x2, y2],
            "det_score": float(getattr(best_face, "det_score", 0.0))
        }

    return crop
