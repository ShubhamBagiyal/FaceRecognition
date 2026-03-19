# 🎓 Face Attendance System

An AI-powered face recognition attendance system built with **SCRFD** (face detection) and **ArcFace** (face recognition) using Streamlit as the UI.

---

## ✨ Features

- **Enroll students** using multiple face photos for better accuracy
- **Take attendance** from a single class/group photo
- **Manual override** — review and correct face matches before saving
- **Attendance visualization** — annotated photo with present/absent panel
- **CSV export** of every attendance session
- Landmark-based face alignment for high accuracy

---

## 🧠 How It Works

1. **Detection** — SCRFD detects all faces in a photo
2. **Alignment** — 5-point landmark alignment normalizes each face to 112×112
3. **Embedding** — ArcFace converts each face into a 512-dim vector
4. **Matching** — Cosine similarity matches detected faces to enrolled students

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/ShubhamBagiyal/facerecognition.git
cd facerecognition
```

### 2. Create and activate a conda environment
```bash
conda create -n face_attendance python=3.11
conda activate face_attendance
```

### 3. Install dependencies
```bash
pip install streamlit opencv-python-headless numpy pandas Pillow onnxruntime insightface scikit-image
```

### 4. Download the face models
```bash
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_s').prepare(ctx_id=0)"
```
This downloads the models to `~/.insightface/models/buffalo_s/` automatically.

### 5. Run the app
```bash
streamlit run newapp.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ How to Use

### Enroll a Student
1. Go to **"Enroll student"** in the sidebar
2. Enter Student ID and Name
3. Upload 3–8 clear frontal face photos
4. Click **Enroll**

### Take Attendance
1. Go to **"Take attendance"**
2. Upload a class/group photo
3. Review detected faces and correct any wrong matches
4. Click **Finalize attendance and save**

### Review Attendance
1. Go to **"Review last attendance"**
2. View the attendance table and annotated photo
3. See present/absent/unknown counts

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `opencv-python-headless` | Image processing |
| `onnxruntime` | Running SCRFD & ArcFace models |
| `insightface` | Model download & face utilities |
| `numpy` | Array operations |
| `pandas` | Attendance CSV handling |
| `Pillow` | Image loading |
| `scikit-image` | Image utilities |

---

## 📁 Project Structure

```
detection/
├── newapp.py          # Main Streamlit app
├── face_attendance_data/
│   ├── prototypes.npz     # Enrolled student embeddings
│   ├── students_db.json   # Student records
│   └── attendance_full.csv # Attendance log
```

> **Note:** `.onnx` model files are not included in this repo. They are automatically downloaded via insightface on first run.
