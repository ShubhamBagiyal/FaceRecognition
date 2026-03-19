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

            # Check faculty table first
            cursor.execute("SELECT * FROM faculty WHERE username = %s AND password = %s", (username, password))
            user = cursor.fetchone()
            if user:
                self.user_info = user
                self.user_role = 'faculty'
                super().accept()
                return

            # If not found, check admin table
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
                cursor.close(); db.close()

# --- Photo Analysis Worker (Unchanged) ---
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
        IMG_SIZE = 160
        DESIRED_LEFT_EYE = (0.275 * IMG_SIZE, 0.35 * IMG_SIZE)
        DESIRED_RIGHT_EYE = (0.725 * IMG_SIZE, 0.35 * IMG_SIZE)
        DESIRED_NOSE = (0.5 * IMG_SIZE, 0.625 * IMG_SIZE)
        landmarks = det.get("landmarks", {})
        if "left_eye" in landmarks and "right_eye" in landmarks and "nose" in landmarks:
            src_points = np.array([landmarks["left_eye"], landmarks["right_eye"], landmarks["nose"]], dtype=np.float32)
            dst_points = np.array([DESIRED_LEFT_EYE, DESIRED_RIGHT_EYE, DESIRED_NOSE], dtype=np.float32)
            try:
                M = cv2.getAffineTransform(src_points, dst_points)
                return cv2.warpAffine(img_bgr, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            except Exception: pass
        facial_area = det['facial_area']
        if isinstance(facial_area, dict):
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            crop = img_bgr[y:y+h, x:x+w]
        elif isinstance(facial_area, list):
            x1, y1, x2, y2 = facial_area
            crop = img_bgr[y1:y2, x1:x2]
        else: return None
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) if crop.size > 0 else None

    def run(self):
        try:
            rgb_img = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
            detected_faces_data = RetinaFace.detect_faces(rgb_img)
            if not isinstance(detected_faces_data, dict): detected_faces_data = {}
            annotated_results, aligned_faces, original_face_objects = [], [], []
            for face_key, face_data in detected_faces_data.items():
                aligned = self._align_face(self.image_data, face_data)
                if aligned is not None:
                    aligned_faces.append(aligned)
                    original_face_objects.append(face_data)
            if not aligned_faces:
                self.finished.emit([]); return
            embeddings = DeepFace.represent(img_path=aligned_faces, model_name='Facenet', detector_backend='skip', enforce_detection=True)
            detected_embeddings = np.array([face['embedding'] for face in embeddings])
            norms = np.linalg.norm(detected_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            detected_embeddings /= norms
            if detected_embeddings.size > 0 and self.known_students_matrix.size > 0:
                sims = detected_embeddings.dot(self.known_students_matrix.T)
                best_match_indices = np.argmax(sims, axis=1)
                best_match_scores = np.max(sims, axis=1)
            else:
                best_match_indices, best_match_scores = [], []
            for i, face_obj in enumerate(original_face_objects):
                confidence = float(best_match_scores[i]) if i < len(best_match_scores) else 0.0
                student_id = self.known_student_ids[best_match_indices[i]] if confidence >= self.recognition_threshold else -1
                annotated_results.append({"student_id": student_id, "confidence": confidence, "facial_area": face_obj['facial_area']})
            self.finished.emit(annotated_results)
        except Exception as e:
            self.error.emit(f"Failed during analysis: {e}")

# --- Faculty Dashboard Window (Unchanged logic, now a QWidget) ---
class FacultyDashboard(QWidget):
    def __init__(self, faculty_info, parent=None):
        super().__init__(parent)
        self.faculty_info = faculty_info
        self.known_students_dict, self.known_students_matrix, self.known_student_ids = self.load_known_students()
        self.analysis_results, self.image_path, self.session_id = None, None, None
        self.current_threshold = 0.75

        main_layout = QHBoxLayout(self)
        left_panel = QVBoxLayout()
        self.image_preview_label = QLabel("Select a class photo to begin analysis.")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setStyleSheet("background-color: #34495e; border-radius: 5px;")
        left_panel.addWidget(self.image_preview_label, 1)
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 0, 0, 0)
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
        main_layout.addLayout(left_panel, 6)
        main_layout.addLayout(right_panel, 4)

    def load_known_students(self):
        # Logic is identical to before
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
            print(f"Loaded {len(students_dict)} students for faculty.")
            return students_dict, embeddings_matrix, student_ids
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Failed to load students: {e}")
            return {}, np.array([]), []
        finally:
            if db.is_connected(): cursor.close(); db.close()

    def start_analysis(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Class Photo", "", "Image Files (*.png *.jpg *.jpeg)")
        if not file_name: return
        self.image_path = file_name
        image_data = cv2.imread(file_name)
        if image_data is None:
            QMessageBox.critical(self, "Error", "Could not load the selected image.")
            return
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.analyze_button.setText("Analyzing...")
        self.analyze_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(image_data, self.known_students_matrix, self.known_student_ids)
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_finished(self, results):
        self.analysis_results = results
        self.on_threshold_changed(self.threshold_slider.value())
        self.analyze_button.setText("Select Another Photo & Analyze")
        self.analyze_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.analysis_thread.quit()

    def on_analysis_error(self, error_message):
        QMessageBox.critical(self, "Analysis Failed", error_message)
        self.analyze_button.setText("Select Class Photo & Analyze")
        self.analyze_button.setEnabled(True)
        self.analysis_thread.quit()

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
            if res['confidence'] >= self.current_threshold:
                student_id = res['student_id']
                confidence = res['confidence']
                if student_id != -1 and (student_id not in present_students or confidence > present_students[student_id]):
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
            x1, y1, x2, y2 = result['facial_area']
            pt1, pt2 = (x1, y1), (x2, y2)
            student_id = result['student_id']
            confidence = result['confidence']
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
        self.image_preview_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_attendance(self):
        # Logic is identical to before
        if self.analysis_results is None:
            QMessageBox.warning(self, "No Report", "Please analyze a photo first.")
            return
        course_name, ok = QInputDialog.getText(self, "New Session", "Enter Course Name/Code:")
        if not ok or not course_name: return
        db = get_db_connection()
        if db is None: return
        try:
            cursor = db.cursor()
            cursor.execute("INSERT INTO sessions (faculty_id, course_name) VALUES (%s, %s)",
                           (self.faculty_info['faculty_id'], course_name))
            self.session_id = cursor.lastrowid
            present_students = {}
            for res in self.analysis_results:
                if res['confidence'] >= self.current_threshold:
                    student_id = res['student_id']
                    if student_id != -1:
                        confidence = res['confidence']
                        if student_id not in present_students or confidence > present_students[student_id]:
                            present_students[student_id] = confidence
            absent_students = [sid for sid in self.known_students_dict if sid not in present_students]
            for student_id, confidence in present_students.items():
                cursor.execute("INSERT INTO attendance (session_id, student_id, status, confidence_score, method) VALUES (%s, %s, %s, %s, %s)",
                               (self.session_id, student_id, 'present', confidence, 'face'))
            for student_id in absent_students:
                cursor.execute("INSERT INTO attendance (session_id, student_id, status, confidence_score, method) VALUES (%s, %s, %s, %s, %s)",
                               (self.session_id, student_id, 'absent', 0.0, 'face'))
            db.commit()
            QMessageBox.information(self, "Success", f"Attendance for session '{course_name}' saved.")
            self.save_button.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Could not save attendance: {e}")
        finally:
            if db.is_connected():
                cursor.close(); db.close()

# --- NEW: Admin Dashboard Window ---
class AdminDashboard(QWidget):
    def __init__(self, admin_info, parent=None):
        super().__init__(parent)
        self.admin_info = admin_info

        main_layout = QVBoxLayout(self)
        
        title = QLabel(f"Admin Dashboard - Welcome, {self.admin_info['full_name']}")
        title.setFont(QFont("Helvetica", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.tabs = QStackedWidget() # A widget to hold our different views
        
        # --- Create the different pages for the admin ---
        self.students_page = self.create_students_page()
        self.attendance_page = self.create_attendance_page()
        
        self.tabs.addWidget(self.students_page)
        self.tabs.addWidget(self.attendance_page)
        
        # --- Navigation buttons ---
        nav_layout = QHBoxLayout()
        students_button = QPushButton("Manage Students")
        students_button.clicked.connect(lambda: self.tabs.setCurrentWidget(self.students_page))
        attendance_button = QPushButton("View All Attendance")
        attendance_button.clicked.connect(lambda: self.tabs.setCurrentWidget(self.attendance_page))
        nav_layout.addWidget(students_button)
        nav_layout.addWidget(attendance_button)

        main_layout.addWidget(title)
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.tabs)

    def create_students_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        self.students_table = QTableWidget()
        self.students_table.setColumnCount(3)
        self.students_table.setHorizontalHeaderLabels(["SAP ID", "Full Name", "Phone Number"])
        self.students_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.refresh_students_button = QPushButton("Refresh Student List")
        self.refresh_students_button.clicked.connect(self.load_students_data)
        
        self.delete_student_button = QPushButton("Delete Selected Student")
        self.delete_student_button.setStyleSheet("background-color: #e74c3c;")
        self.delete_student_button.clicked.connect(self.delete_student)

        layout.addWidget(QLabel("Enrolled Students:"))
        layout.addWidget(self.students_table)
        layout.addWidget(self.refresh_students_button)
        layout.addWidget(self.delete_student_button)
        self.load_students_data()
        return page

    def create_attendance_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(5)
        self.attendance_table.setHorizontalHeaderLabels(["Session ID", "Course", "Student Name", "Status", "Timestamp"])
        self.attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.refresh_attendance_button = QPushButton("Refresh Attendance Records")
        self.refresh_attendance_button.clicked.connect(self.load_attendance_data)

        layout.addWidget(QLabel("All Attendance Records:"))
        layout.addWidget(self.attendance_table)
        layout.addWidget(self.refresh_attendance_button)
        self.load_attendance_data()
        return page

    def load_students_data(self):
        # Similar to the one in the enrollment app
        db = get_db_connection()
        if db is None: return
        try:
            cursor = db.cursor()
            cursor.execute("SELECT sap_id, full_name, phone_number FROM students ORDER BY full_name")
            records = cursor.fetchall()
            self.students_table.setRowCount(len(records))
            for r_idx, row in enumerate(records):
                for c_idx, item in enumerate(row):
                    self.students_table.setItem(r_idx, c_idx, QTableWidgetItem(str(item)))
        finally:
            if db.is_connected(): cursor.close(); db.close()
    
    def load_attendance_data(self):
        db = get_db_connection()
        if db is None: return
        try:
            cursor = db.cursor()
            # Join tables to get all necessary info
            query = """
                SELECT s.session_id, s.course_name, st.full_name, a.status, a.timestamp
                FROM attendance a
                JOIN sessions s ON a.session_id = s.session_id
                JOIN students st ON a.student_id = st.student_id
                ORDER BY a.timestamp DESC
            """
            cursor.execute(query)
            records = cursor.fetchall()
            self.attendance_table.setRowCount(len(records))
            for r_idx, row in enumerate(records):
                for c_idx, item in enumerate(row):
                    self.attendance_table.setItem(r_idx, c_idx, QTableWidgetItem(str(item)))
        finally:
            if db.is_connected(): cursor.close(); db.close()

    def delete_student(self):
        # Similar to the one in the enrollment app
        current_row = self.students_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Selection Error", "Please select a student to delete.")
            return
        sap_id = self.students_table.item(current_row, 0).text()
        name = self.students_table.item(current_row, 1).text()
        reply = QMessageBox.question(self, "Confirm Deletion", f"Delete {name} ({sap_id})?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            db = get_db_connection()
            if db is None: return
            try:
                cursor = db.cursor()
                cursor.execute("DELETE FROM students WHERE sap_id = %s", (sap_id,))
                db.commit()
                self.students_table.removeRow(current_row)
            finally:
                if db.is_connected(): cursor.close(); db.close()

# --- Main Application Window ---
class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

    def show_dashboard(self, user_info, user_role):
        if user_role == 'faculty':
            dashboard = FacultyDashboard(user_info)
        elif user_role == 'admin':
            dashboard = AdminDashboard(user_info)
        else:
            QMessageBox.critical(self, "Error", "Unknown user role.")
            return
            
        self.stacked_widget.addWidget(dashboard)
        self.stacked_widget.setCurrentWidget(dashboard)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
    try:
        print("Initializing AI Models... This may take a moment.")
        DeepFace.build_model('Facenet')
        print("Models Initialized.")
        
        login_dialog = LoginDialog()
        if login_dialog.exec():
            main_window = MainApplication()
            main_window.show_dashboard(login_dialog.user_info, login_dialog.user_role)
            main_window.show()
            sys.exit(app.exec())
        else:
            sys.exit(0)
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"Could not initialize AI models.\n\nError: {e}")
        sys.exit(1)

