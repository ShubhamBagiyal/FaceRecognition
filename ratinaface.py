from retinaface import RetinaFace
import cv2
import os

# Path to test image
img_path = "../data/test/images.jpeg"  # Adjust relative path if needed
print("Exists:", os.path.exists(img_path))
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

# Load image
img = cv2.imread(img_path)

# Detect faces
faces = RetinaFace.detect_faces(img_path)

# Draw rectangles on detected faces
for key in faces:
    identity = faces[key]
    x1, y1, x2, y2 = identity['facial_area']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show result
cv2.imshow("RetinaFace", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
