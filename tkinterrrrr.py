import customtkinter as ctk
import tkinter.messagebox as messagebox
import pandas as pd
import os

# Setup theme
ctk.set_appearance_mode("dark")   # "light" or "dark"
ctk.set_default_color_theme("blue")  # "green", "dark-blue", etc.

class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Face Attendance System")
        self.geometry("900x600")

        # Sidebar (Navigation)
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        self.label = ctk.CTkLabel(self.sidebar, text="Navigation", font=("Helvetica", 16, "bold"))
        self.label.pack(pady=20)

        self.enroll_btn = ctk.CTkButton(self.sidebar, text="Enroll Face", command=self.enroll_face)
        self.enroll_btn.pack(pady=10, fill="x")

        self.recog_btn = ctk.CTkButton(self.sidebar, text="Start Recognition", command=self.start_recognition)
        self.recog_btn.pack(pady=10, fill="x")

        self.view_btn = ctk.CTkButton(self.sidebar, text="View Attendance", command=self.view_attendance)
        self.view_btn.pack(pady=10, fill="x")

        self.exit_btn = ctk.CTkButton(self.sidebar, text="Exit", fg_color="red", hover_color="darkred", command=self.quit)
        self.exit_btn.pack(pady=30, fill="x")

        # Main content area
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.pack(side="right", expand=True, fill="both")

        self.main_label = ctk.CTkLabel(self.main_frame, text="Welcome to Face Attendance System",
                                       font=("Helvetica", 20, "bold"))
        self.main_label.pack(pady=50)

    def enroll_face(self):
        messagebox.showinfo("Enroll", "Face enrollment process will start here.")

    def start_recognition(self):
        messagebox.showinfo("Recognition", "Real-time recognition will run here.")

    def view_attendance(self):
        if not os.path.exists("attendance.csv"):
            messagebox.showwarning("No Data", "Attendance file not found!")
            return
        df = pd.read_csv("attendance.csv")
        top = ctk.CTkToplevel(self)
        top.title("Attendance Records")
        top.geometry("600x400")
        text_box = ctk.CTkTextbox(top, width=580, height=380)
        text_box.pack(pady=10)
        text_box.insert("1.0", df.to_string(index=False))

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
