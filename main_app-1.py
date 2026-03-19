import sys
import json
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
                               QPushButton, QVBoxLayout, QFormLayout, QMessageBox, 
                               QDialog, QHBoxLayout, QTableWidget, QTableWidgetItem,
                               QHeaderView, QDialogButtonBox, QInputDialog, QFileDialog,
                               QSlider, QStackedWidget)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QThread, Signal, QObject
import cv2
import numpy as np

# --- All heavy imports are at the top for cleaner code ---
from retinaface import RetinaFace
from deepface import DeepFace

# NOTE: The database_connector.py file is still required for this script to work.
from database_connector import get_db_connection

# --- Login Dialog Window (Upgraded for Multi-Role) ---
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Login")
        self.setModal(True)
        self.setFixedSize(300, 150)
        
        self.username = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        
        form_layout = QFormLayout()
        form_layout.addRow("Username:", self.username)
        form_layout.addRow("Password:", self.password)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)
        
        self.user_info = None
        self.user_role = None

    def accept(self):
        db = get_db_connection()
        if db is None:
            QMessageBox.critical(self, "Database Error", "Could not connect to the database.")
            return
        try:
            cursor = db.cursor(dictionary=True)
            username = self.username.text()
            password = self.password.text()

            cursor.execute("SELECT * FROM faculty WHERE username = %s AND password = %s", (username, password))
            user = cursor.fetchone()
            if user:
                self.user_info = user
                self.user_role = 'faculty'
                super().accept()
                return

            cursor.execute("SELECT * FROM admins WHERE username = %s AND password = %s", (username, password))
            user = cursor.fetchone()
            if user:
                self.user_info = user
                self.user_role = 'admin'
                super().accept()
                return

            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        finally:
            if db.is_connected():
                cursor.close()
                db.close()

# --- Photo Analysis Worker (IMPROVED) ---
class AnalysisWorker(QObject):
    finished = Signal(list) 
    error = Signal(str)

    def __init__(self, image_data, known_students_matrix, known_student_ids):
        super().__init__()
        self.image_data = image_data
        self.known_students_matrix = known_students_matrix
        self.known_student_ids = known_student_ids
        self.recognition_threshold = 0.75 

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
        try:
            rgb_img = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
            detected_faces_data = RetinaFace.detect_faces(rgb_img)
            
            if not isinstance(detected_faces_data, dict): 
                detected_faces_data = {}
                
            annotated_results, aligned_faces, original_face_objects = [], [], []
            
            for face_key, face_data in detected_faces_data.items():
                aligned = self._align_face(self.image_data, face_data)
                if aligned is not None:
                    aligned_faces.append(aligned)
                    original_face_objects.append(face_data)

            if not aligned_faces:
                self.finished.emit([])
                return

            # Process each face individually with better error handling
            embeddings_list = []
            valid_face_indices = []
            
            for i, aligned_face in enumerate(aligned_faces):
                try:
                    # Preprocess the face
                    preprocessed_face = self._preprocess_face(aligned_face)
                    
                    # Get embedding
                    embedding_result = DeepFace.represent(
                        img_path=preprocessed_face, 
                        model_name='Facenet', 
                        detector_backend='skip', 
                        enforce_detection=True
                    )
                    
                    # Handle different return formats
                    if isinstance(embedding_result, list) and len(embedding_result) > 0:
                        if isinstance(embedding_result[0], dict) and 'embedding' in embedding_result[0]:
                            embeddings_list.append(embedding_result[0]['embedding'])
                            valid_face_indices.append(i)
                        else:
                            # Try to access the embedding directly
                            embeddings_list.append(embedding_result[0])
                            valid_face_indices.append(i)
                    elif isinstance(embedding_result, dict) and 'embedding' in embedding_result:
                        embeddings_list.append(embedding_result['embedding'])
                        valid_face_indices.append(i)
                    else:
                        print(f"Unexpected embedding format for face {i}")
                        continue
                except Exception as e:
                    print(f"Failed to get embedding for face {i}: {e}")
                    continue

            if not embeddings_list:
                self.finished.emit([])
                return
                
            detected_embeddings = np.array(embeddings_list)
            norms = np.linalg.norm(detected_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            detected_embeddings /= norms
            
            if detected_embeddings.size > 0 and self.known_students_matrix.size > 0:
                sims = detected_embeddings.dot(self.known_students_matrix.T)
                best_match_indices = np.argmax(sims, axis=1)
                best_match_scores = np.max(sims, axis=1)
            else:
                best_match_indices, best_match_scores = [], []
            
            # Create results only for valid faces
            for i, valid_idx in enumerate(valid_face_indices):
                face_obj = original_face_objects[valid_idx]
                confidence = float(best_match_scores[i]) if i < len(best_match_scores) else 0.0
                
                if i < len(best_match_indices) and confidence >= self.recognition_threshold:
                    student_idx = best_match_indices[i]
                    student_id = self.known_student_ids[student_idx]
                else:
                    student_id = -1
                    
                annotated_results.append({
                    "student_id": student_id,
                    "confidence": confidence,
                    "facial_area": face_obj['facial_area']
                })

            self.finished.emit(annotated_results)
        except Exception as e:
            self.error.emit(f"Failed during analysis: {e}")

# --- Unified Dashboard Window ---
class MainDashboard(QMainWindow):
    def __init__(self, user_info, user_role):
        super().__init__()
        self.user_info = user_info
        self.user_role = user_role
        
        self.setWindowTitle(f"{user_role.capitalize()} Dashboard - Welcome, {self.user_info['full_name']}")
        self.setGeometry(100, 100, 1200, 800)
        self._set_stylesheet()

        self.known_students_dict, self.known_students_matrix, self.known_student_ids = self.load_known_students()
        self.analysis_results = None 
        self.image_path = None
        self.session_id = None
        self.current_threshold = 0.75

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Main Content View) ---
        self.left_panel_widget = QStackedWidget()
        
        # --- Right Panel (Controls) ---
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 10, 10, 10)
        right_panel.setSpacing(15)
        
        # --- Dynamically build UI based on role ---
        if self.user_role == 'faculty':
            self._setup_faculty_ui(right_panel)
        elif self.user_role == 'admin':
            self._setup_admin_ui(right_panel)

        main_layout.addWidget(self.left_panel_widget, 6) # 60% of space
        main_layout.addLayout(right_panel, 4) # 40% of space
        self.setCentralWidget(central_widget)

    def _set_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2c3e50; }
            QLabel { color: #ecf0f1; font-size: 14px; }
            QTableWidget { gridline-color: #566573; }
            QHeaderView::section { background-color: #34495e; padding: 5px; }
            QPushButton {
                background-color: #3498db; color: white; font-size: 16px;
                font-weight: bold; border-radius: 5px; padding: 12px;
            }
            QPushButton:hover { background-color: #5dade2; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)

    # --- UI Setup Functions ---
    def _setup_faculty_ui(self, right_panel):
        self.image_preview_label = QLabel("Select a class photo to begin analysis.")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setStyleSheet("background-color: #34495e; border-radius: 5px;")
        self.left_panel_widget.addWidget(self.image_preview_label)

        self.analyze_button = QPushButton("Select Class Photo & Analyze")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.save_button = QPushButton("Save Attendance to Database")
        self.save_button.clicked.connect(self.save_attendance)
        self.save_button.setEnabled(False)
        
        threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel(f"Confidence Threshold: {self.current_threshold:.2f}")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(int(self.current_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.report_table = QTableWidget()
        self.report_table.setColumnCount(4)
        self.report_table.setHorizontalHeaderLabels(["SAP ID", "Student Name", "Status", "Confidence"])
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        right_panel.addWidget(self.analyze_button)
        right_panel.addLayout(threshold_layout)
        right_panel.addWidget(QLabel("Attendance Report:"))
        right_panel.addWidget(self.report_table)
        right_panel.addWidget(self.save_button)
        right_panel.addStretch()

    def _setup_admin_ui(self, right_panel):
        self.admin_students_table = self.create_students_table()
        self.admin_attendance_table = self.create_attendance_table()
        self.left_panel_widget.addWidget(self.admin_students_table)
        self.left_panel_widget.addWidget(self.admin_attendance_table)

        manage_students_button = QPushButton("Manage Students")
        manage_students_button.clicked.connect(lambda: self.left_panel_widget.setCurrentWidget(self.admin_students_table))
        view_attendance_button = QPushButton("View All Attendance")
        view_attendance_button.clicked.connect(lambda: self.left_panel_widget.setCurrentWidget(self.admin_attendance_table))
        self.delete_student_button = QPushButton("Delete Selected Student")
        self.delete_student_button.setStyleSheet("background-color: #e74c3c;")
        self.delete_student_button.clicked.connect(self.delete_student)
        
        right_panel.addWidget(manage_students_button)
        right_panel.addWidget(view_attendance_button)
        right_panel.addStretch()
        right_panel.addWidget(self.delete_student_button)
        self.load_admin_data()

    def create_students_table(self):
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["SAP ID", "Full Name", "Phone Number"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def create_attendance_table(self):
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Session ID", "Course", "Student Name", "Status", "Timestamp"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    # --- Data Loading and Actions ---
    def load_known_students(self):
        db = get_db_connection()
        if db is None: return {}, np.array([]), []
        students_dict, embeddings_list, student_ids = {}, [], []
        try:
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT student_id, sap_id, full_name, student_embedding FROM students WHERE student_embedding IS NOT NULL")
            records = cursor.fetchall()
            for record in records:
                student_id = record['student_id']
                embedding = np.array(json.loads(record['student_embedding']))
                students_dict[student_id] = {'sap_id': record['sap_id'], 'name': record['full_name']}
                embeddings_list.append(embedding)
                student_ids.append(student_id)
            embeddings_matrix = np.array(embeddings_list)
            print(f"Loaded {len(students_dict)} students from database.")
            return students_dict, embeddings_matrix, student_ids
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Failed to load students: {e}")
            return {}, np.array([]), []
        finally:
            if db and db.is_connected(): 
                db.close()

    def load_admin_data(self):
        db = get_db_connection()
        if db is None: return
        try:
            cursor = db.cursor()
            cursor.execute("SELECT sap_id, full_name, phone_number FROM students ORDER BY full_name")
            records = cursor.fetchall()
            self.admin_students_table.setRowCount(len(records))
            for r_idx, row in enumerate(records):
                for c_idx, item in enumerate(row):
                    self.admin_students_table.setItem(r_idx, c_idx, QTableWidgetItem(str(item)))
                    
            query = """SELECT s.session_id, s.course_name, st.full_name, a.status, a.timestamp
                       FROM attendance a JOIN sessions s ON a.session_id = s.session_id
                       JOIN students st ON a.student_id = st.student_id ORDER BY a.timestamp DESC"""
            cursor.execute(query)
            records = cursor.fetchall()
            self.admin_attendance_table.setRowCount(len(records))
            for r_idx, row in enumerate(records):
                for c_idx, item in enumerate(row):
                    self.admin_attendance_table.setItem(r_idx, c_idx, QTableWidgetItem(str(item)))
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Failed to load admin data: {e}")
        finally:
            if db.is_connected(): 
                db.close()

    def delete_student(self):
        current_row = self.admin_students_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Selection Error", "Please select a student to delete.")
            return
        sap_id = self.admin_students_table.item(current_row, 0).text()
        name = self.admin_students_table.item(current_row, 1).text()
        reply = QMessageBox.question(self, "Confirm Deletion", f"Delete {name} ({sap_id})?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            db = get_db_connection()
            if db is None: return
            try:
                cursor = db.cursor()
                cursor.execute("DELETE FROM students WHERE sap_id = %s", (sap_id,))
                db.commit()
                self.admin_students_table.removeRow(current_row)
                QMessageBox.information(self, "Success", f"Student {name} deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "DB Error", f"Could not delete student: {e}")
            finally:
                if db.is_connected(): 
                    db.close()

    # --- Faculty-Specific Functions ---
    def start_analysis(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Class Photo", "", "Image Files (*.png *.jpg *.jpeg)")
        if not file_name: return
        self.image_path = file_name
        image_data = cv2.imread(file_name)
        if image_data is None:
            QMessageBox.critical(self, "Error", "Could not load the selected image."); return
        
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        self.analyze_button.setText("Analyzing...")
        self.analyze_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(image_data, self.known_students_matrix, self.known_student_ids)
        self.analysis_worker.moveToThread(self.analysis_thread)
        
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        
        # Ensure proper thread cleanup
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
        
        self.analysis_thread.start()

    def on_analysis_finished(self, results):
        self.analysis_results = results
        self.on_threshold_changed(self.threshold_slider.value())
        self.analyze_button.setText("Select Another Photo & Analyze")
        self.analyze_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.analysis_thread.quit()
        self.analysis_thread.wait(1000)

    def on_analysis_error(self, error_message):
        QMessageBox.critical(self, "Analysis Failed", error_message)
        self.analyze_button.setText("Select Class Photo & Analyze")
        self.analyze_button.setEnabled(True)
        self.analysis_thread.quit()
        self.analysis_thread.wait(1000)

    def on_threshold_changed(self, value):
        self.current_threshold = value / 100.0
        self.threshold_label.setText(f"Confidence Threshold: {self.current_threshold:.2f}")
        if self.analysis_results is not None:
            self.display_report()
            self.display_annotated_image()

    def display_report(self):
        if self.analysis_results is None: return
        present_students = {}
        
        for res in self.analysis_results:
            if res['confidence'] >= self.current_threshold and res['student_id'] != -1:
                student_id, confidence = res['student_id'], res['confidence']
                if student_id not in present_students or confidence > present_students[student_id]:
                    present_students[student_id] = confidence
        
        self.report_table.setRowCount(len(self.known_students_dict))
        row = 0
        
        for student_id, student_data in self.known_students_dict.items():
            self.report_table.setItem(row, 0, QTableWidgetItem(student_data['sap_id']))
            self.report_table.setItem(row, 1, QTableWidgetItem(student_data['name']))
            
            if student_id in present_students:
                self.report_table.setItem(row, 2, QTableWidgetItem("Present"))
                self.report_table.setItem(row, 3, QTableWidgetItem(f"{present_students[student_id]:.2f}"))
            else:
                self.report_table.setItem(row, 2, QTableWidgetItem("Absent"))
                self.report_table.setItem(row, 3, QTableWidgetItem("0.00"))
            
            row += 1
        
        self.report_table.sortByColumn(1, Qt.SortOrder.AscendingOrder)

    def display_annotated_image(self):
        if self.analysis_results is None: return
        image = cv2.imread(self.image_path)
        
        for result in self.analysis_results:
            facial_area = result['facial_area']
            
            if isinstance(facial_area, dict):
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                pt1, pt2 = (x, y), (x + w, y + h)
            elif isinstance(facial_area, list):
                x1, y1, x2, y2 = facial_area
                pt1, pt2 = (x1, y1), (x2, y2)
            else:
                continue

            student_id, confidence = result['student_id'], result['confidence']
            
            if confidence >= self.current_threshold and student_id != -1:
                identity = self.known_students_dict[student_id]['name']
                color = (0, 255, 0)
                display_text = f"{identity} ({confidence:.2f})"
            else:
                color = (0, 0, 255)
                display_text = "Unknown"
            
            cv2.rectangle(image, pt1, pt2, color, 2)
            cv2.putText(image, display_text, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def save_attendance(self):
        if self.analysis_results is None: return
        
        course_name, ok = QInputDialog.getText(self, "New Session", "Enter Course Name/Code:")
        if not ok or not course_name: return
        
        db = get_db_connection()
        if db is None: return
        
        try:
            cursor = db.cursor()
            cursor.execute("INSERT INTO sessions (faculty_id, course_name) VALUES (%s, %s)", 
                           (self.user_info['faculty_id'], course_name))
            session_id = cursor.lastrowid
            
            present_students = {}
            for res in self.analysis_results:
                if res['confidence'] >= self.current_threshold and res['student_id'] != -1:
                    student_id = res['student_id']
                    confidence = res['confidence']
                    if student_id not in present_students or confidence > present_students[student_id]:
                        present_students[student_id] = confidence
            
            absent_students = [sid for sid in self.known_students_dict if sid not in present_students]
            
            for student_id, confidence in present_students.items():
                cursor.execute("""INSERT INTO attendance (session_id, student_id, status, confidence_score, method) 
                               VALUES (%s, %s, %s, %s, %s)""",
                               (session_id, student_id, 'present', confidence, 'face'))
            
            for student_id in absent_students:
                cursor.execute("""INSERT INTO attendance (session_id, student_id, status, confidence_score, method) 
                               VALUES (%s, %s, %s, %s, %s)""",
                               (session_id, student_id, 'absent', 0.0, 'face'))
            
            db.commit()
            QMessageBox.information(self, "Success", f"Attendance for session '{course_name}' saved.")
            self.save_button.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Could not save attendance: {e}")
        finally:
            if db.is_connected(): 
                db.close()
                
    def closeEvent(self, event):
        """Clean up threads when the application closes."""
        if hasattr(self, 'analysis_thread') and self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait(1000)
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    main_window = None
    try:
        print("Initializing AI Models... This may take a moment.")
        DeepFace.build_model('Facenet')
        print("Models Initialized.")
        
        login_dialog = LoginDialog()
        if login_dialog.exec():
            main_window = MainDashboard(login_dialog.user_info, login_dialog.user_role)
            main_window.show()
            sys.exit(app.exec())
        else:
            sys.exit(0)
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"Could not initialize AI models.\n\nError: {e}")
        sys.exit(1)