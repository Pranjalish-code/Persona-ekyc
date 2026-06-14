# app.py
import os
import time
import tempfile
import random
from typing import List, Optional
from collections import deque
from dataclasses import asdict

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from idphotoextract import crop_face_from_id_robust
from arcface import get_embedding, cosine_similarity
from livecheck import analyze_video


# ---------------------------
# Page
# ---------------------------
st.set_page_config(page_title="Persona-style eKYC (ID + Live Challenge)", layout="wide")
st.title("Persona-style eKYC Demo")
st.caption("ID photo + Live webcam challenge (random blink/mouth within 5 seconds, exact) + liveness + ArcFace match.")


# ---------------------------
# Session state
# ---------------------------
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "source_label" not in st.session_state:
    st.session_state.source_label = None
if "id_emb" not in st.session_state:
    st.session_state.id_emb = None
if "id_face_crop" not in st.session_state:
    st.session_state.id_face_crop = None
if "challenge_passed" not in st.session_state:
    st.session_state.challenge_passed = False
if "last_challenge_text" not in st.session_state:
    st.session_state.last_challenge_text = ""


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Settings")

    st.subheader("Face Match (ArcFace)")
    sim_threshold = st.slider("Cosine similarity threshold", 0.10, 0.80, 0.35, 0.01)

    st.subheader("Liveness analysis")
    max_seconds = st.slider("Analyze duration (seconds)", 4.0, 15.0, 8.5, 0.5)
    sample_fps = st.slider("Sample FPS", 6.0, 20.0, 12.0, 1.0)

    st.subheader("Providers")
    use_gpu = st.checkbox("Use GPU (CUDA) if available", value=False)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    st.subheader("Challenge")
    # ✅ Change requested: force challenge window to 5s (no slider needed)
    challenge_duration = 5.0

    # Keep these sliders (small change) but they now represent count range for the chosen TYPE
    blink_min = st.slider("Min random blinks", 1, 3, 1, 1)
    blink_max = st.slider("Max random blinks", 1, 5, 3, 1)
    mouth_min = st.slider("Min random mouth opens", 1, 2, 1, 1)
    mouth_max = st.slider("Max random mouth opens", 1, 3, 2, 1)


# ---------------------------
# Helpers
# ---------------------------
def uploaded_image_to_bgr(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_uploaded_video_to_temp(uploaded_file) -> str:
    name = uploaded_file.name or "upload.mp4"
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in [".mp4", ".mov", ".mkv", ".avi"]:
        suffix = ".mp4"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def write_frames_to_mp4(frames_bgr: List[np.ndarray], fps=12.0) -> str:
    if not frames_bgr:
        raise ValueError("No frames to write.")
    h, w = frames_bgr[0].shape[:2]
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    for fr in frames_bgr:
        if fr is None:
            continue
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h))
        vw.write(fr)
    vw.release()
    return out_path


def best_face_frame_from_video(video_path: str, max_scan_seconds: float = 6.0) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_scan_seconds * fps)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count > max_frames:
            break

        emb = get_embedding(frame, providers=providers)
        if emb is not None:
            cap.release()
            return frame

    cap.release()
    return None


# ---------------------------
# WebRTC processor: realtime blink/mouth + random EXACT TYPE challenge
# ---------------------------
class LiveActionProcessor:
    def __init__(self, target_fps: float, max_frames: int = 200):
        self.target_fps = float(target_fps)
        self.last_t = 0.0
        self.frames = deque(maxlen=max_frames)

        # counters (cumulative)
        self.blink_count = 0
        self.mouth_open_count = 0

        # run states
        self.eye_closed_run = 0
        self.mouth_open_run = 0
        self.was_eye_closed = False
        self.was_mouth_open = False

        # thresholds
        self.ear_thr = 0.21
        self.ear_consec = 2
        self.mar_thr = 0.55
        self.mar_consec = 2

        # challenge state
        self.challenge_active = False
        self.challenge_passed = False
        self.challenge_failed = False
        self.challenge_text = ""
        self.challenge_start_t = 0.0

        # ✅ exact type challenge fields
        self.challenge_type = None  # "blink" or "mouth"
        self.target_count = 0

        self.base_blinks = 0
        self.base_mouths = 0
        self.challenge_duration = 5.0  # ✅ fixed 5s

        # MediaPipe init
        self._mp_ok = False
        self._face_mesh = None
        try:
            import mediapipe as mp
            self._mp_ok = True
            self._mp = mp
            self._face_mesh = self._mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            self._mp_ok = False
            self._face_mesh = None

        # landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14]

    # ✅ Changed: start_random_challenge now chooses ONE TYPE and requires EXACT count
    def start_random_challenge(self, bmin, bmax, mmin, mmax, duration_s: float):
        self.challenge_duration = 5.0  # ✅ force 5s, ignore passed duration

        bmin = int(min(bmin, bmax))
        bmax = int(max(bmin, bmax))
        mmin = int(min(mmin, mmax))
        mmax = int(max(mmin, mmax))

        # pick type
        self.challenge_type = random.choice(["blink", "mouth"])
        if self.challenge_type == "blink":
            self.target_count = random.randint(bmin, bmax)
            self.challenge_text = f"Blink EXACTLY {self.target_count}x"
        else:
            self.target_count = random.randint(mmin, mmax)
            self.challenge_text = f"Open mouth EXACTLY {self.target_count}x"

        self.base_blinks = int(self.blink_count)
        self.base_mouths = int(self.mouth_open_count)

        self.challenge_passed = False
        self.challenge_failed = False
        self.challenge_active = True
        self.challenge_start_t = time.time()

    def reset_challenge(self):
        self.challenge_active = False
        self.challenge_passed = False
        self.challenge_failed = False
        self.challenge_text = ""
        self.challenge_type = None
        self.target_count = 0

    def _euclid(self, p, q):
        return float(np.linalg.norm(np.array(p, dtype=np.float32) - np.array(q, dtype=np.float32)))

    def _ear(self, lm_xy, idx6):
        p1, p2, p3, p4, p5, p6 = [lm_xy[i] for i in idx6]
        num = self._euclid(p2, p6) + self._euclid(p3, p5)
        den = 2.0 * self._euclid(p1, p4) + 1e-6
        return num / den

    def _mar(self, lm_xy, idx4):
        left, right, upper, lower = [lm_xy[i] for i in idx4]
        num = self._euclid(upper, lower)
        den = self._euclid(left, right) + 1e-6
        return num / den

    def _get_lm_xy(self, bgr):
        if not self._mp_ok or self._face_mesh is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        h, w = bgr.shape[:2]
        pts = res.multi_face_landmarks[0].landmark
        return [(int(p.x * w), int(p.y * h)) for p in pts]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # buffer frames
        if now - self.last_t >= (1.0 / max(self.target_fps, 1.0)):
            self.frames.append(img.copy())
            self.last_t = now

        # detect blink/mouth
        if self._mp_ok:
            lm_xy = self._get_lm_xy(img)
            if lm_xy is not None:
                ear = 0.5 * (self._ear(lm_xy, self.LEFT_EYE) + self._ear(lm_xy, self.RIGHT_EYE))
                mar = self._mar(lm_xy, self.MOUTH)

                # blink event
                if ear < self.ear_thr:
                    self.eye_closed_run += 1
                else:
                    if self.eye_closed_run >= self.ear_consec and not self.was_eye_closed:
                        self.blink_count += 1
                        self.was_eye_closed = True
                    self.eye_closed_run = 0
                    self.was_eye_closed = False

                # mouth open event
                if mar > self.mar_thr:
                    self.mouth_open_run += 1
                else:
                    if self.mouth_open_run >= self.mar_consec and not self.was_mouth_open:
                        self.mouth_open_count += 1
                        self.was_mouth_open = True
                    self.mouth_open_run = 0
                    self.was_mouth_open = False

        # ✅ challenge evaluate + overlay (EXACT, TYPE-only, 5 seconds)
        if self.challenge_active:
            elapsed = now - self.challenge_start_t
            remaining = self.challenge_duration - elapsed

            d_blinks = int(self.blink_count) - int(self.base_blinks)
            d_mouths = int(self.mouth_open_count) - int(self.base_mouths)

            if self.challenge_type == "blink":
                progress = d_blinks
                other = d_mouths
            else:
                progress = d_mouths
                other = d_blinks

            # strict: doing other action => fail
            if other > 0:
                self.challenge_failed = True
                self.challenge_active = False

            # exact: exceed => fail immediately
            elif progress > int(self.target_count):
                self.challenge_failed = True
                self.challenge_active = False

            # time end => must be exactly target
            elif remaining <= 0:
                if progress == int(self.target_count):
                    self.challenge_passed = True
                else:
                    self.challenge_failed = True
                self.challenge_active = False

            # reached exactly early => pass immediately
            elif progress == int(self.target_count):
                self.challenge_passed = True
                self.challenge_active = False

            cv2.putText(img, "RANDOM CHALLENGE (EXACT)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(img, self.challenge_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, f"Time left: {max(0.0, remaining):.1f}s", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if self.challenge_type == "blink":
                prog_text = f"Progress: blink {progress}/{self.target_count} (mouth {other})"
            else:
                prog_text = f"Progress: mouth {progress}/{self.target_count} (blink {other})"
            cv2.putText(img, prog_text, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # counters overlay always
        cv2.putText(img, f"Blinks: {int(self.blink_count)}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"Mouth opens: {int(self.mouth_open_count)}", (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if self.challenge_passed:
            cv2.putText(img, "CHALLENGE: PASS", (20, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        elif self.challenge_failed:
            cv2.putText(img, "CHALLENGE: FAIL", (20, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------
# Layout
# ---------------------------
left, right = st.columns(2, gap="large")

# ========== LEFT: ID upload ==========
with left:
    st.subheader("1) Upload Government ID Photo")
    id_file = st.file_uploader("ID image (jpg/png)", type=["jpg", "jpeg", "png"], key="idimg")

    if id_file is not None:
        id_bgr = uploaded_image_to_bgr(id_file)
        st.image(bgr_to_rgb(id_bgr), caption="Original ID", use_container_width=True)

        try:
            id_face_crop, dbg = crop_face_from_id_robust(id_bgr, return_debug=True)
            st.session_state.id_face_crop = id_face_crop
            st.image(bgr_to_rgb(id_face_crop), caption="Cropped ID face", use_container_width=True)
            st.write("Crop debug:", dbg)
        except Exception as e:
            st.error(f"ID crop failed: {e}")
            st.session_state.id_face_crop = None

        if st.session_state.id_face_crop is not None:
            id_emb = get_embedding(st.session_state.id_face_crop, providers=providers)
            if id_emb is None:
                st.error("No face embedding extracted from ID crop.")
                st.session_state.id_emb = None
            else:
                st.session_state.id_emb = id_emb
                st.success("ID embedding extracted ✅")


# ========== RIGHT: Video upload OR webcam ==========
with right:
    st.subheader("2) Provide KYC Video")
    tab_up, tab_cam = st.tabs(["Upload video", "Live webcam"])

    with tab_up:
        vid_file = st.file_uploader("KYC video (mp4/mov/mkv/avi)", type=["mp4", "mov", "mkv", "avi"], key="vid")
        if vid_file is not None:
            st.session_state.video_path = save_uploaded_video_to_temp(vid_file)
            st.session_state.source_label = "uploaded_video"
            st.session_state.challenge_passed = True  # no random challenge for upload
            st.video(st.session_state.video_path)

    with tab_cam:
        st.markdown("### 🎯 Live Random Challenge")
        st.info("Press **Start Random Challenge** → do EXACTLY the asked action within **5 seconds** → then save clip → verify.")

        RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

        webrtc_ctx = webrtc_streamer(
            key="webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: LiveActionProcessor(
                sample_fps,
                max_frames=int(float(sample_fps) * float(max_seconds)) + 80
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        vp = webrtc_ctx.video_processor if webrtc_ctx else None

        buffered = len(vp.frames) if vp is not None else 0
        blink_live = int(getattr(vp, "blink_count", 0)) if vp is not None else 0
        mouth_live = int(getattr(vp, "mouth_open_count", 0)) if vp is not None else 0

        st.write(f"Buffered frames: **{buffered}**")
        st.write(f"Realtime blinks: **{blink_live}**")
        st.write(f"Realtime mouth opens: **{mouth_live}**")

        # Challenge status
        if vp is not None:
            status = "— not started"
            if vp.challenge_active:
                status = "⏳ ACTIVE"
            elif vp.challenge_passed:
                status = "✅ PASS"
            elif vp.challenge_failed:
                status = "❌ FAIL"
            st.write("Challenge status:", status)

        cA, cB, cC = st.columns(3)

        with cA:
            if st.button("🎲 Start Random Challenge"):
                if vp is None:
                    st.error("Start the webcam first.")
                else:
                    vp.start_random_challenge(blink_min, blink_max, mouth_min, mouth_max, float(challenge_duration))
                    st.session_state.last_challenge_text = vp.challenge_text
                    st.session_state.challenge_passed = False
                    st.success(f"Challenge started: {vp.challenge_text} (within {challenge_duration:.1f}s)")

        with cB:
            if st.button("🔁 Reset Challenge"):
                if vp is not None:
                    vp.reset_challenge()
                st.session_state.challenge_passed = False
                st.session_state.last_challenge_text = ""
                st.success("Challenge reset.")

        with cC:
            if st.button("🧹 Clear buffer"):
                if vp is not None:
                    vp.frames.clear()
                st.success("Cleared buffer.")

        st.caption(f"Last challenge: {st.session_state.last_challenge_text or '—'}")

        s1, s2 = st.columns(2)
        with s1:
            if st.button("💾 Save webcam clip"):
                if vp is None:
                    st.error("Start the webcam first.")
                else:
                    # Important: enforce challenge PASS before saving
                    if not bool(getattr(vp, "challenge_passed", False)):
                        st.session_state.challenge_passed = False
                        st.error("❌ Challenge NOT passed. Complete the random challenge first.")
                    else:
                        frames = list(vp.frames)
                        if len(frames) < 30:
                            st.warning("Record longer (need at least ~30 frames).")
                        else:
                            st.session_state.video_path = write_frames_to_mp4(frames, fps=float(sample_fps))
                            st.session_state.source_label = "live_webcam"
                            st.session_state.challenge_passed = True
                            st.success("Webcam clip saved ✅")
                            st.video(st.session_state.video_path)

        with s2:
            if st.button("✅ Mark challenge pass (debug)"):
                # remove this button in production; it's only for dev testing
                st.session_state.challenge_passed = True
                st.warning("Debug: forced challenge_passed=True")


# ---------------------------
# Verify
# ---------------------------
st.divider()
st.subheader("3) Verify")

run_btn = st.button(
    "Verify Now",
    type="primary",
    disabled=(st.session_state.id_emb is None or st.session_state.video_path is None),
)

if run_btn:
    # Enforce random challenge for webcam
    if st.session_state.source_label == "live_webcam" and not st.session_state.challenge_passed:
        st.error("❌ Random challenge not passed. Marking SPOOF.")
        st.stop()

    with st.spinner("Running liveness + face match..."):
        video_path = str(st.session_state.video_path)

        # Liveness
        try:
            liv = analyze_video(
                video_path,
                max_seconds=float(max_seconds),
                sample_fps=float(sample_fps),
                face_size=160,
            )
        except Exception as e:
            st.error(f"Liveness failed: {e}")
            st.stop()

        # Face match frame
        best_frame = best_face_frame_from_video(video_path, max_scan_seconds=min(6.0, float(max_seconds)))
        if best_frame is None:
            st.error("Could not find a face frame in the video for ArcFace match.")
            st.stop()

        vid_emb = get_embedding(best_frame, providers=providers)
        if vid_emb is None:
            st.error("Face embedding not found from best video frame.")
            st.stop()

        sim = cosine_similarity(st.session_state.id_emb, vid_emb)
        match_ok = (sim is not None) and (sim >= float(sim_threshold))

        live_ok = (liv.decision == "LIVE")
        final_ok = match_ok and live_ok

    a, b, c = st.columns(3, gap="large")

    with a:
        st.markdown("### Face Match (ArcFace)")
        st.metric("Cosine similarity", f"{sim:.4f}" if sim is not None else "None")
        st.write("Threshold:", float(sim_threshold))
        st.write("Decision:", "✅ MATCH" if match_ok else "❌ NO MATCH")
        st.image(bgr_to_rgb(best_frame), caption="Video frame used for match", use_container_width=True)

    with b:
        st.markdown("### Liveness")
        st.metric("Liveness score", f"{liv.score:.3f}")
        st.write("Decision:", "✅ LIVE" if live_ok else "❌ SPOOF")
        st.json(asdict(liv))

    with c:
        st.markdown("### Final eKYC Decision")
        if final_ok:
            st.success("✅ VERIFIED")
        else:
            st.error("❌ FAILED")

        if not live_ok:
            st.warning(f"Liveness hint: {liv.info.get('spoof_hint', 'unknown')}")
        if not match_ok:
            st.warning("Face mismatch: similarity below threshold.")

st.caption("Production note: remove the debug button and keep challenge as mandatory for live webcam.")
