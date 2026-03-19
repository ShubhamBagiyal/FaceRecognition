# app.py
import os
import io
import time
import json
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import onnxruntime as ort

# -------------------------
# File paths
# -------------------------
DATA_DIR = Path("face_attendance_data")
DATA_DIR.mkdir(exist_ok=True)
PROTOS_PATH = DATA_DIR / "prototypes.npz"
DB_JSON = DATA_DIR / "students_db.json"
ATTENDANCE_CSV = DATA_DIR / "attendance_full.csv"
VISUALIZATION_IMG = DATA_DIR / "attendance_visualized.jpg"

IMG_SIZE = 112  # ArcFace input
BATCH_SIZE = 16
THRESHOLD_DEFAULT = 0.55  # ArcFace matching threshold

# -------------------------
# SCRFD detector (InsightFace)
# -------------------------
@st.cache_resource
def load_scrfd():
    model_path = "scrfd_500m.onnx"
    if not os.path.exists(model_path):
        import urllib.request
        url = "https://github.com/deepinsight/insightface/releases/download/v0.0.3/scrfd_500m.onnx"
        urllib.request.urlretrieve(url, model_path)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess

SCRFD_SESSION = load_scrfd()

# -------------------------
# ArcFace embedder
# -------------------------
@st.cache_resource
def load_arcface():
    model_path = "arcface_r100.onnx"
    if not os.path.exists(model_path):
        import urllib.request
        url = "https://github.com/deepinsight/insightface/releases/download/v0.0.2/arcface_r100.onnx"
        urllib.request.urlretrieve(url, model_path)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess

ARCSESSION = load_arcface()

# -------------------------
# Utilities
# -------------------------
def imgfile_to_bgr(img_file) -> np.ndarray:
    img = Image.open(img_file).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def write_temp_image(bgr_img) -> str:
    tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tf.name, bgr_img)
    tf.close()
    return tf.name

def prewhiten(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return (x - 0.5) / 0.5

def align_face(bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
    crop = bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

def emb_from_bgr(face_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = prewhiten(img)
    img = np.transpose(img, (2,0,1))[np.newaxis,:].astype(np.float32)
    emb = ARCSESSION.run(None, {"input": img})[0][0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)

def load_prototypes():
    if PROTOS_PATH.exists():
        data = np.load(str(PROTOS_PATH), allow_pickle=True)
        ids = data["ids"].tolist()
        names = data["names"].tolist()
        protos = data["prototypes"].astype(np.float32)
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        protos = protos / norms
        return ids, names, protos
    return [], [], np.zeros((0,512), dtype=np.float32)

def save_prototypes(ids: List[str], names: List[str], prototypes: np.ndarray):
    np.savez_compressed(str(PROTOS_PATH),
                        ids=np.array(ids, dtype=object),
                        names=np.array(names, dtype=object),
                        prototypes=prototypes.astype(np.float32))

def load_db():
    if DB_JSON.exists():
        with open(DB_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"students": {}}

def save_db(db):
    with open(DB_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

# -------------------------
# SCRFD Face Detection
# -------------------------
def detect_faces_scrfd(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = np.transpose(img, (2,0,1))[np.newaxis,:]  # NCHW
    img /= 255.0
    input_name = SCRFD_SESSION.get_inputs()[0].name
    output = SCRFD_SESSION.run(None, {input_name: img})[0]
    bboxes = []
    for box in output:
        score = box[4]
        if score > 0.5:
            x1, y1, x2, y2 = box[:4].astype(int)
            bboxes.append([x1,y1,x2,y2])
    return bboxes

# -------------------------
# Enrollment
# -------------------------
def enroll_student_from_upload(student_id: str, student_name: str, uploaded_files: List[io.BytesIO]):
    if not uploaded_files:
        raise ValueError("No images uploaded for enrollment.")
    db = load_db()
    ids, names, protos = load_prototypes()
    embeddings = []
    saved_paths = []

    for up in uploaded_files:
        bgr = imgfile_to_bgr(up)
        temp_path = write_temp_image(bgr)
        saved_paths.append(temp_path)
        bboxes = detect_faces_scrfd(bgr)
        if not bboxes:
            continue
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
        idx = int(np.argmax(areas))
        aligned = align_face(bgr, bboxes[idx])
        emb = emb_from_bgr(aligned)
        embeddings.append(emb)

    if not embeddings:
        st.error("No usable face embeddings extracted — enrollment aborted.")
        return False

    proto = np.mean(np.vstack(embeddings), axis=0)
    proto /= np.linalg.norm(proto)
    ids.append(student_id)
    names.append(student_name)
    protos_new = np.vstack([protos, proto]) if protos.size else proto[None,:]
    save_prototypes(ids, names, protos_new)
    db["students"][student_id] = {
        "id": student_id,
        "name": student_name,
        "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(embeddings)
    }
    save_db(db)
    st.success(f"Enrolled {student_name} with {len(embeddings)} embeddings.")
    for p in saved_paths:
        try: os.remove(p)
        except: pass
    return True

# -------------------------
# Attendance
# -------------------------
def process_class_photo(photo_bgr: np.ndarray, threshold: float):
    ids, names, protos = load_prototypes()
    bboxes = detect_faces_scrfd(photo_bgr)
    crops = [align_face(photo_bgr, bb) for bb in bboxes]
    embs = [emb_from_bgr(c) for c in crops]
    suggested = []
    for i, e in enumerate(embs):
        if protos.size == 0:
            suggested.append({"student_id": None, "name": None, "score": 0.0})
            continue
        sims = e.dot(protos.T)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= threshold:
            suggested.append({"student_id": ids[idx], "name": names[idx], "score": score})
        else:
            suggested.append({"student_id": None, "name": None, "score": score})
    return crops, bboxes, suggested

def finalize_attendance(matches: List[dict], proto_ids: List[str], proto_names: List[str], photo_bgr: np.ndarray):
    present_ids = set([m["assigned_id"] for m in matches if m.get("assigned_id")])
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    import csv
    rows = []

    # Present
    for sid in present_ids:
        m = next((x for x in matches if x.get("assigned_id")==sid), None)
        rows.append({"timestamp": timestamp, "student_id": sid, "name": m.get("assigned_name",""),
                     "score": m.get("score",0.0), "bbox": m.get("bbox",""), "status":"present"})
    # Absent
    for sid,name in zip(proto_ids,proto_names):
        if sid not in present_ids:
            rows.append({"timestamp": timestamp, "student_id": sid, "name": name, "score":0.0, "bbox":"", "status":"absent"})
    # Unknown
    for m in matches:
        if not m.get("assigned_id"):
            rows.append({"timestamp": timestamp, "student_id": "", "name": "", "score": m.get("score",0.0),
                         "bbox": m.get("bbox",""), "status":"unknown"})

    # Save CSV
    with open(ATTENDANCE_CSV,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","student_id","name","score","bbox","status"])
        w.writeheader()
        for r in rows: w.writerow(r)

    # Visualization
    vis = photo_bgr.copy()
    for m in matches:
        bbox = m.get("bbox")
        if not bbox: continue
        x1,y1,x2,y2 = bbox
        color = (0,255,0) if m.get("assigned_id") else (0,0,255)
        label = f"{m.get('assigned_name','unknown')} {m.get('score',0.0):.2f}"
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
        cv2.putText(vis,label,(x1,max(10,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)

    # Absent panel
    absent_names = [n for i,n in zip(proto_ids,proto_names) if i not in present_ids]
    h_img = vis.shape[0]; panel_w=360
    panel = 255*np.ones((h_img,panel_w,3),dtype=np.uint8)
    cv2.putText(panel,"ABSENT",(8,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
    y=60; line_h=20; max_lines = max(1,(h_img-60)//line_h)
    for idx,nm in enumerate(absent_names[:max_lines]):
        cv2.putText(panel,f"{idx+1}. {nm}",(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        y+=line_h
    if len(absent_names)>max_lines:
        cv2.putText(panel,f"... +{len(absent_names)-max_lines} more",(8,y+6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    combined = np.concatenate([vis,panel],axis=1)
    cv2.imwrite(str(VISUALIZATION_IMG),combined)
    return rows, combined

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Face Attendance (SCRFD+ArcFace)", layout="wide")
st.title("Face Attendance — Enroll, Match, Review")

menu = st.sidebar.selectbox("Choose action", ["Enroll student", "Take attendance", "Review last attendance"])
proto_ids, proto_names, prototypes = load_prototypes()
db = load_db()

if menu=="Enroll student":
    st.header("Enroll a single student")
    with st.form("enroll_form"):
        student_id = st.text_input("Student ID (unique)")
        student_name = st.text_input("Student name")
        uploaded = st.file_uploader("Upload photos (3-8 recommended)", accept_multiple_files=True, type=["jpg","jpeg","png","bmp"])
        submit = st.form_submit_button("Enroll")
    if submit:
        if not student_id.strip():
            st.error("Student ID required")
        elif not uploaded:
            st.error("Upload at least one image")
        else:
            st.info("Enrolling...")
            enroll_student_from_upload(student_id.strip(), student_name.strip() or student_id.strip(), uploaded)

elif menu=="Take attendance":
    st.header("Take attendance from a class photo")
    threshold = st.sidebar.slider("Matching threshold", 0.0, 1.0, float(THRESHOLD_DEFAULT), 0.01)
    uploaded_photo = st.file_uploader("Upload class/group photo", type=["jpg","jpeg","png","bmp"])
    if uploaded_photo:
        photo_bgr = imgfile_to_bgr(uploaded_photo)
        st.image(cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded photo", use_column_width=True)
        crops, bboxes, suggested = process_class_photo(photo_bgr, threshold)
        if not crops:
            st.warning("No faces detected")
        else:
            st.success(f"Detected {len(crops)} faces")
            matches=[]
            cols = st.columns(3)
            for i,(crop,bbox,sug) in enumerate(zip(crops,bboxes,suggested)):
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                col = cols[i%3]
                col.image(crop_rgb, width=200, caption=f"Face #{i+1}")
                options = ["<Keep suggested>"] + [f"{sid} — {name}" for sid,name in zip(proto_ids,proto_names)] + ["<Unknown>"]
                sel = col.selectbox(f"Assign Face #{i+1}", options, index=0, key=f"match_sel_{i}")
                if sel=="<Keep suggested>":
                    assigned_id = sug["student_id"]; assigned_name = sug["name"]
                elif sel=="<Unknown>":
                    assigned_id = ""; assigned_name=""
                else:
                    assigned_id = sel.split(" — ",1)[0]; assigned_name=sel.split(" — ",1)[1] if " — " in sel else assigned_id
                matches.append({"assigned_id":assigned_id,"assigned_name":assigned_name,"score":sug["score"],"bbox":bbox})
            if st.button("Finalize attendance and save"):
                rows_out, vis = finalize_attendance(matches, proto_ids, proto_names, photo_bgr)
                st.success(f"Attendance finalized. CSV: {ATTENDANCE_CSV}")
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Attendance visualization", use_column_width=True)

elif menu=="Review last attendance":
    st.header("Review last saved attendance")
    if ATTENDANCE_CSV.exists():
        import pandas as pd
        df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(df)
        if VISUALIZATION_IMG.exists():
            st.image(Image.open(VISUALIZATION_IMG), use_column_width=True)
