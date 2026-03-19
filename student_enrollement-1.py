import sys
import json
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
                               QPushButton, QVBoxLayout, QFormLayout, QMessageBox, 
                               QFileDialog, QHBoxLayout, QDialog, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QScrollArea, QFrame)
from PySide6.QtGui import QPixmap, QFont, QImage
from PySide6.QtCore import Qt, QThread, Signal, QObject
import cv2
import numpy as np

# --- All heavy imports are at the top for cleaner code ---
from deepface import DeepFace
from retinaface import RetinaFace 

# NOTE: The database_connector.py file is still required for this script to work.
from database_connector import get_db_connection

# --- Worker for Model Loading ---
class ModelLoaderWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def run(self):
        """Pre-loads ALL heavy AI models to prevent hangs during operation."""
        try:
            print("Loading AI models...")
            blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # 1. Pre-load Facenet model from DeepFace
            _ = DeepFace.represent(
                img_path=blank_image,
                model_name='Facenet', detector_backend='opencv', enforce_detection=False
            )
            
            # 2. Pre-load RetinaFace model
            _ = RetinaFace.detect_faces(blank_image)

            print("AI Models Initialized Successfully.")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

# --- Worker for AI Processing (IMPROVED) ---
class EmbeddingWorker(QObject):
    finished = Signal(str, object)

    def __init__(self, image_data_list):
        super().__init__()
        self.image_data_list = image_data_list

    def _align_face(self, img_bgr, det):
        """Improved face alignment with better landmark handling"""
        IMG_SIZE = 160
        DESIRED_LEFT_EYE = (0.275 * IMG_SIZE, 0.35 * IMG_SIZE)
        DESIRED_RIGHT_EYE = (0.725 * IMG_SIZE, 0.35 * IMG_SIZE)
        DESIRED_NOSE = (0.5 * IMG_SIZE, 0.625 * IMG_SIZE)
        
        landmarks = det.get("landmarks", {})
        
        # Use facial landmarks for precise alignment if available
        if "left_eye" in landmarks and "right_eye" in landmarks and "nose" in landmarks:
            src_points = np.array([landmarks["left_eye"], landmarks["right_eye"], landmarks["nose"]], dtype=np.float32)
            dst_points = np.array([DESIRED_LEFT_EYE, DESIRED_RIGHT_EYE, DESIRED_NOSE], dtype=np.float32)
            try:
                M = cv2.getAffineTransform(src_points, dst_points)
                aligned = cv2.warpAffine(img_bgr, M, (IMG_SIZE, IMG_SIZE), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                return aligned
            except Exception:
                pass  # Fall back to basic cropping
        
        # Fallback: basic cropping with padding
        facial_area = det['facial_area']
        if isinstance(facial_area, dict):
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        elif isinstance(facial_area, list):
            x1, y1, x2, y2 = facial_area
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
        else:
            return None
            
        # Add padding to the crop
        pad_x = int(0.15 * w)
        pad_y = int(0.15 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_bgr.shape[1], x + w + pad_x)
        y2 = min(img_bgr.shape[0], y + h + pad_y)
        
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
            
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    def _preprocess_face(self, face_img):
        """Apply preprocessing to normalize the face image"""
        # Convert to float and normalize
        face_img = face_img.astype(np.float32) / 255.0
        # Simple whitening
        face_img = (face_img - 0.5) / 0.5
        return face_img

    def run(self):
        embeddings = []
        successful_images = 0
        
        try:
            for img_data in self.image_data_list:
                rgb_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                
                # Detect faces with error handling
                try:
                    detected_faces = RetinaFace.detect_faces(rgb_img)
                except Exception as e:
                    print(f"Face detection failed: {e}")
                    continue
                
                if not isinstance(detected_faces, dict) or not detected_faces:
                    continue
                
                # Find the largest face
                best_face_key = max(detected_faces.keys(), 
                                  key=lambda k: (detected_faces[k]['facial_area'][2] - 
                                               detected_faces[k]['facial_area'][0]) * (detected_faces[k]['facial_area'][3] - 
                                               detected_faces[k]['facial_area'][1]))
                
                face_data = detected_faces[best_face_key]
                aligned_face = self._align_face(img_data, face_data)
                
                if aligned_face is None:
                    continue
                
                # Preprocess the face
                preprocessed_face = self._preprocess_face(aligned_face)
                
                # Get embedding
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=preprocessed_face, 
                        model_name='Facenet', 
                        detector_backend='skip', 
                        enforce_detection=True
                    )
                    
                    if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                        if 'embedding' in embedding_objs[0]:
                            embeddings.append(embedding_objs[0]['embedding'])
                            successful_images += 1
                        else:
                            # Handle different return format
                            embeddings.append(embedding_objs[0])
                            successful_images += 1
                except Exception as e:
                    print(f"Embedding extraction failed: {e}")
                    continue
            
            if not embeddings:
                self.finished.emit("error", "No valid faces could be detected in any selected photos.")
                return
            
            # Create prototype embedding from all successful images
            prototype_embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(prototype_embedding)
            
            if norm > 0:
                normalized_prototype = prototype_embedding / norm
                self.finished.emit("success", (normalized_prototype.tolist(), successful_images))
            else:
                self.finished.emit("error", "A zero-vector embedding was generated.")
                
        except Exception as e:
            self.finished.emit("error", f"An unexpected error occurred: {e}")

# --- Dialog to display enrolled students ---
class EnrolledStudentsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enrolled Students")
        self.setGeometry(200, 200, 600, 500)
        self.setStyleSheet("background-color: #2c3e50; color: white;")
        
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget { gridline-color: #566573; } 
            QHeaderView::section { background-color: #34495e; padding: 5px; }
        """)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["SAP ID", "Full Name", "Phone Number", "Images Used"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.table)
        
        self.delete_button = QPushButton("Delete Selected Student")
        self.delete_button.setStyleSheet("""
            background-color: #e74c3c; 
            padding: 10px; 
            border-radius: 5px; 
            font-weight: bold;
        """)
        self.delete_button.clicked.connect(self.delete_student)
        layout.addWidget(self.delete_button)
        
        self.setLayout(layout)
        self.load_student_data()

    def load_student_data(self):
        db = get_db_connection()
        if db is None: 
            return
            
        try:
            cursor = db.cursor()
            cursor.execute("SELECT sap_id, full_name, phone_number, images_used FROM students ORDER BY full_name")
            records = cursor.fetchall()
            
            self.table.setRowCount(len(records))
            for row_idx, row_data in enumerate(records):
                for col_idx, col_data in enumerate(row_data):
                    self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load students: {e}")
        finally:
            if db and db.is_connected(): 
                db.close()

    def delete_student(self):
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Selection Error", "Please select a student to delete.")
            return
            
        sap_id = self.table.item(current_row, 0).text()
        name = self.table.item(current_row, 1).text()
        
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion", 
            f"Are you sure you want to delete {name} ({sap_id})?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            db = get_db_connection()
            if db is None: 
                return
                
            try:
                cursor = db.cursor()
                cursor.execute("DELETE FROM students WHERE sap_id = %s", (sap_id,))
                db.commit()
                self.table.removeRow(current_row)
                QMessageBox.information(self, "Success", f"Student {name} deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Database Error", f"Failed to delete student: {e}")
            finally:
                if db.is_connected(): 
                    db.close()

# --- Main Application Window ---
class StudentEnrollmentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Student Enrollment System")
        self.setGeometry(100, 100, 550, 700)
        self._set_stylesheet()
        
        self.photo_preview_widgets = []
        self.initialize_ui()
        self.initialize_models()

    def _set_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2c3e50; }
            QLabel { color: #ecf0f1; font-size: 14px; }
            QLineEdit { 
                background-color: #34495e; 
                color: #ecf0f1; 
                border-radius: 5px; 
                padding: 8px; 
                font-size: 14px; 
            }
            QPushButton { 
                background-color: #3498db; 
                color: white; 
                font-size: 16px; 
                font-weight: bold; 
                border-radius: 5px; 
                padding: 12px; 
            }
            QPushButton:hover { background-color: #5dade2; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)

    def initialize_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(50, 30, 50, 40)
        main_layout.setSpacing(15)
        
        title = QLabel("Student Enrollment")
        title.setFont(QFont("Helvetica", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        form_layout = QFormLayout()
        self.fields = { 
            "Full Name": QLineEdit(), 
            "SAP ID": QLineEdit(), 
            "Password": QLineEdit(), 
            "Phone Number": QLineEdit() 
        }
        self.fields["Password"].setEchoMode(QLineEdit.EchoMode.Password)
        
        for label, editor in self.fields.items():
            form_layout.addRow(label, editor)
        
        main_layout.addLayout(form_layout)
        
        photo_select_layout = QHBoxLayout()
        self.select_photo_button = QPushButton("Select Photos (3-5 recommended)...")
        self.select_photo_button.clicked.connect(self.select_photos)
        photo_select_layout.addWidget(self.select_photo_button)
        
        self.photo_path_label = QLabel("No photos selected.")
        self.photo_path_label.setStyleSheet("color: #bdc3c7; font-style: italic;")
        photo_select_layout.addWidget(self.photo_path_label, 1)
        
        main_layout.addLayout(photo_select_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(150)
        self.scroll_area.setStyleSheet("background-color: #34495e; border-radius: 5px;")
        
        self.preview_container = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.preview_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.scroll_area.setWidget(self.preview_container)
        
        main_layout.addWidget(self.scroll_area)
        
        self.enroll_button = QPushButton("Enroll Student")
        self.enroll_button.clicked.connect(self.enroll_student)
        
        self.view_students_button = QPushButton("View Enrolled Students")
        self.view_students_button.setStyleSheet("background-color: #95a5a6; color: #2c3e50;")
        self.view_students_button.clicked.connect(self.view_enrolled_students)
        
        main_layout.addWidget(self.enroll_button)
        main_layout.addWidget(self.view_students_button)
        
        self.setCentralWidget(central_widget)

    def initialize_models(self):
        self.enroll_button.setText("Initializing AI Models...")
        self.enroll_button.setEnabled(False)
        self.view_students_button.setEnabled(False)
        
        self.model_loader_thread = QThread()
        self.model_loader_worker = ModelLoaderWorker()
        self.model_loader_worker.moveToThread(self.model_loader_thread)
        
        self.model_loader_thread.started.connect(self.model_loader_worker.run)
        self.model_loader_worker.finished.connect(self.on_models_loaded)
        self.model_loader_worker.error.connect(self.on_models_load_error)
        
        self.model_loader_thread.finished.connect(self.model_loader_thread.deleteLater)
        
        self.model_loader_thread.start()

    def on_models_loaded(self):
        self.enroll_button.setText("Enroll Student")
        self.enroll_button.setEnabled(True)
        self.view_students_button.setEnabled(True)
        self.model_loader_thread.quit()
        self.model_loader_thread.wait(1000)

    def on_models_load_error(self, error_message):
        QMessageBox.critical(self, "Model Error", f"Failed to initialize AI models: {error_message}")
        self.enroll_button.setText("Initialization Failed")
        self.model_loader_thread.quit()
        self.model_loader_thread.wait(1000)

    def select_photos(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Student Photos (3-5 recommended)", 
            "", 
            "Image Files (*.png *.jpg *.jpeg)"
        )
        
        if not file_names:
            return
            
        failed_files = []
        for file_name in file_names:
            try:
                with open(file_name, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img_data is not None:
                    self.add_photo_preview(img_data)
                else:
                    failed_files.append(os.path.basename(file_name))
            except Exception:
                failed_files.append(os.path.basename(file_name))
        
        self.update_photo_count_label()
        
        if failed_files:
            QMessageBox.warning(
                self, 
                "Image Load Warning", 
                "Could not read the following files:\n" + "\n".join(failed_files)
            )

    def add_photo_preview(self, img_data):
        thumbnail_widget = QFrame()
        thumbnail_widget.setStyleSheet("background-color: #2c3e50; padding: 5px; border-radius: 5px;")
        
        thumbnail_layout = QVBoxLayout(thumbnail_widget)
        pixmap = self.get_pixmap_from_cv(img_data)
        
        label = QLabel()
        label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        remove_button = QPushButton("Remove")
        remove_button.setStyleSheet("font-size: 10px; padding: 4px; background-color: #e74c3c;")
        remove_button.clicked.connect(lambda: self.remove_photo(thumbnail_widget))
        
        thumbnail_layout.addWidget(label)
        thumbnail_layout.addWidget(remove_button)
        
        self.preview_layout.addWidget(thumbnail_widget)
        self.photo_preview_widgets.append((thumbnail_widget, img_data))

    def remove_photo(self, widget_to_remove):
        self.photo_preview_widgets = [item for item in self.photo_preview_widgets if item[0] != widget_to_remove]
        widget_to_remove.deleteLater()
        self.update_photo_count_label()
    
    def update_photo_count_label(self):
        count = len(self.photo_preview_widgets)
        status_text = f"{count} photo(s) selected." if count > 0 else "No photos selected."
        
        if 0 < count < 3:
            status_text += " (Recommend 3-5 photos for better recognition)"
            
        self.photo_path_label.setText(status_text)

    def get_pixmap_from_cv(self, img_data):
        rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap(qt_image)
        
    def enroll_student(self):
        student_data = {key.lower().replace(" ", "_"): editor.text() for key, editor in self.fields.items()}
        photo_data_list = [item[1] for item in self.photo_preview_widgets]
        
        if not all([student_data["full_name"], student_data["sap_id"], student_data["password"]]):
            QMessageBox.critical(self, "Error", "All fields including password are required.")
            return
            
        if not photo_data_list:
            QMessageBox.critical(self, "Error", "At least one photo is required.")
            return
            
        if len(photo_data_list) < 3:
            reply = QMessageBox.question(
                self,
                "Few Photos Warning",
                "Using fewer than 3 photos may result in poor recognition accuracy. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.enroll_button.setText("Processing...")
        self.enroll_button.setEnabled(False)
        
        self.embedding_thread = QThread()
        self.embedding_worker = EmbeddingWorker(photo_data_list)
        self.embedding_worker.moveToThread(self.embedding_thread)
        
        self.embedding_thread.started.connect(self.embedding_worker.run)
        self.embedding_worker.finished.connect(self.on_embedding_finished)
        
        self.embedding_thread.finished.connect(self.embedding_thread.deleteLater)
        
        self.embedding_thread.start()

    def view_enrolled_students(self):
        dialog = EnrolledStudentsDialog(self)
        dialog.exec()

    def on_embedding_finished(self, status, result):
        try:
            if status == "success":
                embedding, successful_images = result
                student_data = {key.lower().replace(" ", "_"): editor.text() for key, editor in self.fields.items()}
                self._save_to_database(student_data, json.dumps(embedding), successful_images)
            else:
                QMessageBox.critical(self, "Enrollment Failed", str(result))
        finally:
            self.enroll_button.setText("Enroll Student")
            self.enroll_button.setEnabled(True)
            if hasattr(self, 'embedding_thread') and self.embedding_thread:
                self.embedding_thread.quit()
                self.embedding_thread.wait(1000)

    def _save_to_database(self, data, embedding, successful_images):
        db_connection = get_db_connection()
        if db_connection is None: 
            QMessageBox.critical(self, "Database Error", "Could not connect to database.")
            return
            
        try:
            cursor = db_connection.cursor()
            sql = """INSERT INTO students 
                     (full_name, sap_id, password, phone_number, student_embedding, images_used) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
                     
            cursor.execute(sql, (
                data['full_name'], 
                data['sap_id'], 
                data['password'], 
                data['phone_number'], 
                embedding,
                successful_images
            ))
            
            db_connection.commit()
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Student '{data['full_name']}' enrolled successfully with {successful_images} images!"
            )
            
            for editor in self.fields.values(): 
                editor.clear()
            for widget, _ in self.photo_preview_widgets:
                widget.deleteLater()
            self.photo_preview_widgets = []
            self.update_photo_count_label()
            
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to save student: {e}")
        finally:
            if db_connection and db_connection.is_connected(): 
                db_connection.close()
                
    def closeEvent(self, event):
        """Clean up threads when the application closes."""
        if hasattr(self, 'model_loader_thread') and self.model_loader_thread:
            self.model_loader_thread.quit()
            self.model_loader_thread.wait(1000)
            
        if hasattr(self, 'embedding_thread') and self.embedding_thread:
            self.embedding_thread.quit()
            self.embedding_thread.wait(1000)
            
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StudentEnrollmentApp()
    window.show()
    sys.exit(app.exec())

