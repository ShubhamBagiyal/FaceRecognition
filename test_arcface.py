import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# Initialize SCRFD + ArcFace pipeline
app = FaceAnalysis(name="buffalo_l")  # "buffalo_l" = SCRFD detector + ArcFace ResNet100
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 = CPU (good for Mac M1)

def compare_faces(img_path1, img_path2, threshold=0.35):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    faces1 = app.get(img1)
    faces2 = app.get(img2)

    if not faces1 or not faces2:
        print(f"❌ Face not detected in {img_path1} or {img_path2}")
        return

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    # Cosine similarity
    cosine_sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    print(f"🔍 {img_path1} vs {img_path2} → Similarity: {cosine_sim:.3f}")

    if cosine_sim > threshold:
        print("✅ Likely same person")
    else:
        print("❌ Different person")

# ---- Tests ----
compare_faces("face1.jpg", "face2.jpg")  # same person
compare_faces("face1.jpg", "face3.jpg")  # different person
