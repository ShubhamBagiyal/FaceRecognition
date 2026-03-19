# app.py
import os
import io
import csv
import time
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import onnxruntime as ort

# -------------------------
# File paths
# -------------------------
DATA_DIR = Path("face_attendance_data")
DATA_DIR.mkdir(exist_ok=True)
PROTOS_PATH   = DATA_DIR / "prototypes.npz"
DB_JSON       = DATA_DIR / "students_db.json"
ATTENDANCE_CSV = DATA_DIR / "attendance_full.csv"
VISUALIZATION_IMG = DATA_DIR / "attendance_visualized.jpg"

IMG_SIZE          = 112      # ArcFace input size
THRESHOLD_DEFAULT = 0.55     # cosine-similarity threshold

# -------------------------
# Model paths (from insightface buffalo_s, already downloaded)
# -------------------------
_INSIGHTFACE_DIR = Path.home() / ".insightface" / "models" / "buffalo_s"
SCRFD_MODEL_PATH   = _INSIGHTFACE_DIR / "det_500m.onnx"
ARCFACE_MODEL_PATH = _INSIGHTFACE_DIR / "w600k_mbf.onnx"

# -------------------------
# 5-point reference landmarks for ArcFace alignment
# (standard 112x112 crop used by InsightFace)
# -------------------------
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# -------------------------
# Model loaders (cached)
# -------------------------
@st.cache_resource
def load_scrfd() -> ort.InferenceSession:
    if not SCRFD_MODEL_PATH.exists():
        st.error(f"SCRFD model not found at {SCRFD_MODEL_PATH}. Run: python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_s').prepare(ctx_id=0)\"")
        st.stop()
    return ort.InferenceSession(str(SCRFD_MODEL_PATH), providers=["CPUExecutionProvider"])


@st.cache_resource
def load_arcface() -> ort.InferenceSession:
    if not ARCFACE_MODEL_PATH.exists():
        st.error(f"ArcFace model not found at {ARCFACE_MODEL_PATH}. Run: python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_s').prepare(ctx_id=0)\"")
        st.stop()
    return ort.InferenceSession(str(ARCFACE_MODEL_PATH), providers=["CPUExecutionProvider"])


SCRFD_SESSION  = load_scrfd()
ARC_SESSION    = load_arcface()

# Cache input/output names so we don't query every call
_SCRFD_INPUT   = SCRFD_SESSION.get_inputs()[0].name
_ARC_INPUT     = ARC_SESSION.get_inputs()[0].name


# -------------------------
# Image utilities
# -------------------------
def imgfile_to_bgr(img_file) -> np.ndarray:
    """Convert an uploaded file-like object to a BGR numpy array."""
    img = Image.open(img_file).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def prewhiten(img: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] as expected by InsightFace ArcFace."""
    x = img.astype(np.float32) / 255.0
    return (x - 0.5) / 0.5


# -------------------------
# SCRFD Face Detection
# SCRFD ONNX has 9 output tensors (3 strides × 3 heads: score, bbox, kps).
# We decode each stride manually.
# -------------------------
_SCRFD_STRIDES = [8, 16, 32]
_SCRFD_NUM_ANCHORS = 2          # anchors per location (scrfd_500m)


def _generate_anchors(height: int, width: int, stride: int) -> np.ndarray:
    """Generate anchor center points for one stride level."""
    shifts_x = (np.arange(width)  + 0.5) * stride
    shifts_y = (np.arange(height) + 0.5) * stride
    xx, yy = np.meshgrid(shifts_x, shifts_y)
    centers = np.stack([xx.ravel(), yy.ravel()], axis=1)      # (H*W, 2)
    centers = np.repeat(centers, _SCRFD_NUM_ANCHORS, axis=0)   # (H*W*2, 2)
    return centers.astype(np.float32)


def detect_faces_scrfd(
    bgr: np.ndarray,
    conf_thresh: float = 0.5,
    nms_thresh: float  = 0.4,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Run SCRFD detection.

    Returns
    -------
    bboxes : list of [x1,y1,x2,y2]  (pixel coords, clipped to image)
    landmarks : list of (5,2) float32 arrays
    """
    h, w = bgr.shape[:2]

    # Resize to 640×640 as expected by scrfd_500m
    inp_size  = 640
    img_resized = cv2.resize(bgr, (inp_size, inp_size))
    scale_x   = w / inp_size
    scale_y   = h / inp_size

    img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 128.0                                  # InsightFace normalisation
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]            # NCHW

    outputs = SCRFD_SESSION.run(None, {_SCRFD_INPUT: img})
    # outputs order: score_8, score_16, score_32,
    #                bbox_8,  bbox_16,  bbox_32,
    #                kps_8,   kps_16,   kps_32

    all_boxes  = []
    all_scores = []
    all_kps    = []

    for i, stride in enumerate(_SCRFD_STRIDES):
        score_pred = outputs[i].reshape(-1)                      # (H*W*2,)
        bbox_pred  = outputs[i + 3].reshape(-1, 4)               # (H*W*2, 4)
        kps_pred   = outputs[i + 6].reshape(-1, 5, 2)            # (H*W*2, 5, 2)

        fh = inp_size // stride
        fw = inp_size // stride
        centers = _generate_anchors(fh, fw, stride)              # (H*W*2, 2)

        # Decode bboxes
        x1 = (centers[:, 0] - bbox_pred[:, 0] * stride) * scale_x
        y1 = (centers[:, 1] - bbox_pred[:, 1] * stride) * scale_y
        x2 = (centers[:, 0] + bbox_pred[:, 2] * stride) * scale_x
        y2 = (centers[:, 1] + bbox_pred[:, 3] * stride) * scale_y

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Decode landmarks
        kps = centers[:, np.newaxis, :] + kps_pred * stride      # (N,5,2)
        kps[:, :, 0] *= scale_x
        kps[:, :, 1] *= scale_y

        mask = score_pred >= conf_thresh
        all_boxes.append(boxes[mask])
        all_scores.append(score_pred[mask])
        all_kps.append(kps[mask])

    if not any(len(b) for b in all_boxes):
        return [], []

    all_boxes  = np.concatenate(all_boxes,  axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_kps    = np.concatenate(all_kps,    axis=0)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        all_boxes[:, :4].tolist(),
        all_scores.tolist(),
        conf_thresh,
        nms_thresh,
    )
    if len(indices) == 0:
        return [], []

    indices = indices.flatten()
    bboxes    = []
    landmarks = []
    for idx in indices:
        x1, y1, x2, y2 = all_boxes[idx]
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        bboxes.append([x1, y1, x2, y2])
        landmarks.append(all_kps[idx].astype(np.float32))

    return bboxes, landmarks


# -------------------------
# Face Alignment (5-point, similarity transform)
# -------------------------
def align_face_landmark(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align face using 5 landmarks to the reference 112×112 crop.
    Falls back to simple bbox crop if transform fails.
    """
    try:
        M, _ = cv2.estimateAffinePartial2D(
            landmarks, REFERENCE_LANDMARKS, method=cv2.LMEDS
        )
        if M is None:
            raise ValueError("Transform estimation failed")
        aligned = cv2.warpAffine(bgr, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)
        return aligned
    except Exception:
        # Fallback: plain resize
        return cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))


def align_face_bbox(bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Fallback crop+resize when no landmarks are available."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = bgr
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


# -------------------------
# ArcFace embedding
# -------------------------
def emb_from_aligned(face_bgr: np.ndarray) -> np.ndarray:
    """Compute L2-normalised ArcFace embedding from an aligned 112×112 BGR crop."""
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = prewhiten(img)
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    emb = ARC_SESSION.run(None, {_ARC_INPUT: img})[0][0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


# -------------------------
# Prototype store
# -------------------------
def load_prototypes() -> Tuple[List[str], List[str], np.ndarray]:
    if PROTOS_PATH.exists():
        data   = np.load(str(PROTOS_PATH), allow_pickle=True)
        ids    = data["ids"].tolist()
        names  = data["names"].tolist()
        protos = data["prototypes"].astype(np.float32)
        # Re-normalise in case of numerical drift
        norms  = np.linalg.norm(protos, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        protos = protos / norms
        return ids, names, protos
    return [], [], np.zeros((0, 512), dtype=np.float32)


def save_prototypes(ids: List[str], names: List[str], prototypes: np.ndarray) -> None:
    np.savez_compressed(
        str(PROTOS_PATH),
        ids=np.array(ids, dtype=object),
        names=np.array(names, dtype=object),
        prototypes=prototypes.astype(np.float32),
    )


# -------------------------
# Student DB (JSON)
# -------------------------
def load_db() -> dict:
    if DB_JSON.exists():
        with open(DB_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"students": {}}


def save_db(db: dict) -> None:
    with open(DB_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


# -------------------------
# Enrollment
# -------------------------
def enroll_student(
    student_id: str,
    student_name: str,
    uploaded_files: List[io.BytesIO],
) -> bool:
    if not uploaded_files:
        st.error("No images uploaded.")
        return False

    ids, names, protos = load_prototypes()
    db = load_db()

    # Duplicate check
    if student_id in ids:
        st.warning(
            f"Student ID **{student_id}** already enrolled. "
            "Delete the existing entry first or use a different ID."
        )
        return False

    embeddings: List[np.ndarray] = []
    temp_paths: List[str]        = []

    for up in uploaded_files:
        bgr = imgfile_to_bgr(up)

        # Save temp file for debugging purposes (optional)
        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tf.name, bgr)
        tf.close()
        temp_paths.append(tf.name)

        bboxes, landmarks = detect_faces_scrfd(bgr)
        if not bboxes:
            st.warning(f"No face detected in one image — skipping.")
            continue

        # Pick the largest face
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        best  = int(np.argmax(areas))

        if landmarks and len(landmarks) > best:
            aligned = align_face_landmark(bgr, landmarks[best])
        else:
            aligned = align_face_bbox(bgr, bboxes[best])

        emb = emb_from_aligned(aligned)
        embeddings.append(emb)

    # Clean temp files
    for p in temp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    if not embeddings:
        st.error("No usable embeddings extracted — enrollment aborted.")
        return False

    proto = np.mean(np.vstack(embeddings), axis=0)
    proto /= np.linalg.norm(proto) + 1e-8

    ids.append(student_id)
    names.append(student_name)
    protos_new = np.vstack([protos, proto]) if protos.size else proto[np.newaxis, :]
    save_prototypes(ids, names, protos_new)

    db["students"][student_id] = {
        "id":          student_id,
        "name":        student_name,
        "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images":    len(embeddings),
    }
    save_db(db)

    st.success(f"✅ Enrolled **{student_name}** ({student_id}) with {len(embeddings)} images.")
    return True


# -------------------------
# Attendance processing
# -------------------------
def process_class_photo(
    photo_bgr: np.ndarray,
    threshold: float,
) -> Tuple[List[np.ndarray], List[List[int]], List[dict]]:
    ids, names, protos = load_prototypes()
    bboxes, landmarks = detect_faces_scrfd(photo_bgr)

    crops: List[np.ndarray] = []
    suggested: List[dict]   = []

    for i, bbox in enumerate(bboxes):
        if landmarks and i < len(landmarks):
            crop = align_face_landmark(photo_bgr, landmarks[i])
        else:
            crop = align_face_bbox(photo_bgr, bbox)
        crops.append(crop)

        emb = emb_from_aligned(crop)

        if protos.size == 0:
            suggested.append({"student_id": None, "name": None, "score": 0.0})
            continue

        sims  = emb.dot(protos.T)
        idx   = int(np.argmax(sims))
        score = float(sims[idx])

        if score >= threshold:
            suggested.append({"student_id": ids[idx], "name": names[idx], "score": score})
        else:
            suggested.append({"student_id": None, "name": None, "score": score})

    return crops, bboxes, suggested


def finalize_attendance(
    matches: List[dict],
    proto_ids: List[str],
    proto_names: List[str],
    photo_bgr: np.ndarray,
) -> Tuple[List[dict], np.ndarray]:
    present_ids = {m["assigned_id"] for m in matches if m.get("assigned_id")}
    timestamp   = time.strftime("%Y-%m-%d %H:%M:%S")
    rows: List[dict] = []

    # Present
    for sid in present_ids:
        m = next((x for x in matches if x.get("assigned_id") == sid), None)
        rows.append({
            "timestamp":  timestamp,
            "student_id": sid,
            "name":       m.get("assigned_name", "") if m else "",
            "score":      round(m.get("score", 0.0), 4) if m else 0.0,
            "bbox":       str(m.get("bbox", "")) if m else "",
            "status":     "present",
        })

    # Absent
    for sid, name in zip(proto_ids, proto_names):
        if sid not in present_ids:
            rows.append({
                "timestamp":  timestamp,
                "student_id": sid,
                "name":       name,
                "score":      0.0,
                "bbox":       "",
                "status":     "absent",
            })

    # Unknown faces
    for m in matches:
        if not m.get("assigned_id"):
            rows.append({
                "timestamp":  timestamp,
                "student_id": "",
                "name":       "unknown",
                "score":      round(m.get("score", 0.0), 4),
                "bbox":       str(m.get("bbox", "")),
                "status":     "unknown",
            })

    # Write CSV
    fieldnames = ["timestamp", "student_id", "name", "score", "bbox", "status"]
    with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Visualization
    vis = photo_bgr.copy()
    for m in matches:
        bbox = m.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        color = (0, 200, 0) if m.get("assigned_id") else (0, 0, 220)
        label = f"{m.get('assigned_name', 'unknown')} {m.get('score', 0.0):.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
            cv2.LINE_AA,
        )

    # Absent panel on the right
    absent_names = [n for i, n in zip(proto_ids, proto_names) if i not in present_ids]
    h_img, w_img = vis.shape[:2]
    panel_w  = 360
    panel    = np.ones((h_img, panel_w, 3), dtype=np.uint8) * 245
    line_h   = 22
    max_lines = max(1, (h_img - 60) // line_h)

    cv2.putText(panel, "ABSENT", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2, cv2.LINE_AA)
    y = 60
    for idx, nm in enumerate(absent_names[:max_lines]):
        cv2.putText(panel, f"{idx + 1}. {nm}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        y += line_h
    if len(absent_names) > max_lines:
        cv2.putText(
            panel, f"... +{len(absent_names) - max_lines} more",
            (10, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA,
        )

    combined = np.concatenate([vis, panel], axis=1)
    cv2.imwrite(str(VISUALIZATION_IMG), combined)
    return rows, combined


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Face Attendance — SCRFD + ArcFace", layout="wide")
st.title("🎓 Face Attendance — Enroll · Match · Review")

menu = st.sidebar.selectbox(
    "Choose action",
    ["Enroll student", "Take attendance", "Review last attendance"],
)

# Always load fresh (don't cache these — they change after enroll)
proto_ids, proto_names, prototypes = load_prototypes()
db = load_db()

# =============================================================
# ENROLL
# =============================================================
if menu == "Enroll student":
    st.header("Enroll a student")
    with st.form("enroll_form"):
        student_id   = st.text_input("Student ID (must be unique)")
        student_name = st.text_input("Student name")
        uploaded     = st.file_uploader(
            "Upload face photos (3–8 recommended, clear frontal)",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png", "bmp"],
        )
        submitted = st.form_submit_button("Enroll")

    if submitted:
        sid  = student_id.strip()
        sname = student_name.strip() or sid
        if not sid:
            st.error("Student ID is required.")
        elif not uploaded:
            st.error("Please upload at least one image.")
        else:
            with st.spinner("Enrolling…"):
                enroll_student(sid, sname, uploaded)

    # Show enrolled students
    if db["students"]:
        st.divider()
        st.subheader("Currently enrolled students")
        st.dataframe(
            pd.DataFrame(db["students"].values()),
            use_container_width=True,
        )

# =============================================================
# TAKE ATTENDANCE
# =============================================================
elif menu == "Take attendance":
    st.header("Take attendance from a class photo")

    if not proto_ids:
        st.warning("No students enrolled yet. Go to 'Enroll student' first.")
        st.stop()

    threshold     = st.sidebar.slider("Matching threshold", 0.0, 1.0, float(THRESHOLD_DEFAULT), 0.01)
    uploaded_photo = st.file_uploader("Upload class/group photo", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_photo:
        photo_bgr = imgfile_to_bgr(uploaded_photo)
        st.image(
            cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB),
            caption="Uploaded photo",
            use_container_width=True,
        )

        with st.spinner("Detecting and recognising faces…"):
            crops, bboxes, suggested = process_class_photo(photo_bgr, threshold)

        if not crops:
            st.warning("⚠️ No faces detected in the photo.")
        else:
            st.success(f"Detected **{len(crops)}** face(s)")
            matches: List[dict] = []
            cols = st.columns(3)

            for i, (crop, bbox, sug) in enumerate(zip(crops, bboxes, suggested)):
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                col = cols[i % 3]
                col.image(crop_rgb, width=180, caption=f"Face #{i + 1}")

                default_label = (
                    f"{sug['student_id']} — {sug['name']} ({sug['score']:.2f})"
                    if sug["student_id"]
                    else f"<Unknown>  (best score: {sug['score']:.2f})"
                )
                col.caption(f"Suggested: {default_label}")

                options  = (
                    ["<Keep suggested>"]
                    + [f"{sid} — {name}" for sid, name in zip(proto_ids, proto_names)]
                    + ["<Unknown>"]
                )
                sel = col.selectbox(f"Assign Face #{i + 1}", options, index=0, key=f"match_{i}")

                if sel == "<Keep suggested>":
                    assigned_id   = sug["student_id"]
                    assigned_name = sug["name"]
                elif sel == "<Unknown>":
                    assigned_id   = ""
                    assigned_name = ""
                else:
                    parts         = sel.split(" — ", 1)
                    assigned_id   = parts[0]
                    assigned_name = parts[1] if len(parts) > 1 else parts[0]

                matches.append({
                    "assigned_id":   assigned_id,
                    "assigned_name": assigned_name,
                    "score":         sug["score"],
                    "bbox":          bbox,
                })

            st.divider()
            if st.button("✅ Finalize attendance and save", type="primary"):
                with st.spinner("Saving…"):
                    rows_out, vis = finalize_attendance(matches, proto_ids, proto_names, photo_bgr)
                st.success(f"Attendance saved → `{ATTENDANCE_CSV}`")
                st.image(
                    cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                    caption="Attendance visualization",
                    use_container_width=True,
                )
                st.dataframe(pd.DataFrame(rows_out), use_container_width=True)

# =============================================================
# REVIEW
# =============================================================
elif menu == "Review last attendance":
    st.header("Review last saved attendance")

    if not ATTENDANCE_CSV.exists():
        st.info("No attendance record found yet.")
    else:
        df = pd.read_csv(ATTENDANCE_CSV)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Present",  int((df["status"] == "present").sum()))
        c2.metric("Absent",   int((df["status"] == "absent").sum()))
        c3.metric("Unknown",  int((df["status"] == "unknown").sum()))

        st.divider()
        st.dataframe(df, use_container_width=True)

        if VISUALIZATION_IMG.exists():
            st.image(Image.open(VISUALIZATION_IMG), use_container_width=True)