import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import tensorflow as tf
import ttkbootstrap as tb

# Detect the base directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Automatically locate the model and icon inside the project folder
model_path = os.path.join(BASE_DIR, "skin_cancer_model.keras")
icon_path = os.path.join(BASE_DIR, "skin-cancer.ico")

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Class names
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

class SkinCancerDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Cancer Detection - AI Powered")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Set the window icon
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print("Icon load error:", e)

        self.actual_class = None
        self.setup_ui()

    def setup_ui(self):
        frame = tb.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        title = tb.Label(frame, text="Skin Cancer Detection", font=("Segoe UI", 22, "bold"))
        title.pack(pady=20)

        self.image_label = tb.Label(frame, text="No image selected", bootstyle="info", width=30)
        self.image_label.pack(pady=10)

        self.result_label = tb.Label(frame, text="", font=("Segoe UI", 16), bootstyle="success")
        self.result_label.pack(pady=10)

        btn_upload = tb.Button(frame, text="📁 Select Image", command=self.select_image, bootstyle="primary-outline")
        btn_upload.pack(pady=10)

        footer = tb.Label(frame, text="Developed by: Houssem Bouagal", font=("Segoe UI", 10, "italic"), bootstyle="secondary")
        footer.pack(side="bottom", pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        img = Image.open(file_path).resize((240, 240))
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

        self.actual_class = os.path.basename(os.path.dirname(file_path)).lower()
        if self.actual_class not in class_names:
            self.actual_class = "unknown"

        img_array = cv2.imread(file_path)
        img_array = cv2.resize(img_array, (240, 240))
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        predicted_class = class_names[class_index]

        self.result_label.config(
            text=f"🧠 True Label: {self.actual_class}\n🔍 Predicted Label: {predicted_class}",
            bootstyle="success"
        )

# Launch the app
if __name__ == "__main__":
    app = tb.Window(themename="darkly")
    SkinCancerDetectorApp(app)
    app.mainloop()
