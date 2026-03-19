import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from retinaface import RetinaFace

# ================= Setup =================

attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])

# Load embeddings if exist
if os.path.exists("live_embeddings.npy") and os.path.exists("live_labels.npy"):
    embeddings = np.load("live_embeddings.npy", allow_pickle=True).tolist()
    labels = np.load("live_labels.npy", allow_pickle=True).tolist()
else:
    embeddings = []
    labels = []

logged_names = set()

# HaarCascade as fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================= Functions =================

def start_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Try RetinaFace detection
        faces = RetinaFace.detect_faces(frame, threshold=0.5)

        if isinstance(faces, dict):  # Faces found
            for key in faces.keys():
                facial_area = faces[key]["facial_area"]
                x1, y1, x2, y2 = facial_area
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Simulate recognition (just take "Unknown" for now)
                name = "Student"

                if name not in logged_names:
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    logged_names.add(name)

        else:
            # Fallback HaarCascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in haar_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def enroll_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible")
        return

    name = simpledialog.askstring("Enroll Face", "Enter name:")
    if not name:
        cap.release()
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Failed to capture image")
        return

    faces = RetinaFace.detect_faces(frame, threshold=0.5)
    if not isinstance(faces, dict):
        messagebox.showwarning("Warning", "No face detected")
        return

    # Save dummy embedding (in real case you’d use FaceNet)
    embeddings.append(np.random.rand(128))
    labels.append(name)

    np.save("live_embeddings.npy", np.array(embeddings, dtype=object))
    np.save("live_labels.npy", np.array(labels, dtype=object))

    messagebox.showinfo("Success", f"{name} enrolled successfully!")


def view_attendance():
    if not os.path.exists(attendance_file):
        messagebox.showinfo("Attendance", "No attendance records yet")
        return

    with open(attendance_file, "r") as f:
        records = f.read()

    top = tk.Toplevel(root)
    top.title("Attendance Records")
    text = tk.Text(top, wrap="word", width=60, height=20)
    text.insert("1.0", records)
    text.pack()


# ================= GUI =================

root = tk.Tk()
root.title("FaceTrack - Smart Attendance System")
root.geometry("400x300")

btn1 = tk.Button(root, text="Start Recognition", command=start_recognition, height=2, width=25)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="Enroll New Face", command=enroll_face, height=2, width=25)
btn2.pack(pady=10)

btn3 = tk.Button(root, text="View Attendance", command=view_attendance, height=2, width=25)
btn3.pack(pady=10)

btn4 = tk.Button(root, text="Exit", command=root.quit, height=2, width=25)
btn4.pack(pady=10)

root.mainloop()
