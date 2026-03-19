import cv2
import os

img_path = "../data/test/images.jpeg"
print("Exists:", os.path.exists(img_path))
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load image
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Show result
cv2.imshow("Haar Cascade", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
