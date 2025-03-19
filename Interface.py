import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

# Define script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "skin_cancer_model.keras")
icon_path = os.path.join(script_dir, "skin-cancer.ico")

# Check required files
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found: {model_path}")

if not os.path.exists(icon_path):
    print(f"⚠️ Warning: Icon not found ({icon_path})")

# Load model
model = load_model(model_path)

# Disease classes
classes = [
    "Actinic Keratoses (AKIEC)", 
    "Basal Cell Carcinoma (BCC)", 
    "Benign Keratosis (BKL)", 
    "Dermatofibroma (DF)", 
    "Melanoma (MEL)", 
    "Melanocytic Nevi (NV)", 
    "Vascular Lesions (VASC)"
]

def load_and_predict():
    """Loads an image, performs prediction, and updates the UI."""
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
    
    if not file_path:
        return  # User canceled

    try:
        # Update button text and cursor
        btn_select.config(state=tk.DISABLED, text="📊 Analyzing...")
        root.config(cursor="watch")  
        root.update()

        # Load and preprocess image
        img = Image.open(file_path).resize((128, 128))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = classes[predicted_index]  
        predicted_proba = prediction[predicted_index] * 100  # Convert to percentage

        # Update UI with result
        result_label.config(
            text=f"🩺 Diagnosis: {predicted_class}\n🎯 Confidence: {predicted_proba:.2f}%",
            fg="white"
        )

        # Display image
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

    finally:
        # Reset button and cursor
        btn_select.config(state=tk.NORMAL, text="📸 Select an Image")
        root.config(cursor="")

def show_team():
    """Displays developer information."""
    team_members = "Developed by:\n- Bouagal Houssem Eddine"
    messagebox.showinfo("Developer", team_members)

# Create main window
root = tk.Tk()
root.title("Skin Cancer Detection")
root.geometry("480x600")
root.configure(bg="#34495E")

# Set window icon
if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

# Widget styles
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10, background="#2980B9")
style.configure("TFrame", background="#34495E")

# Title
header_label = tk.Label(
    root, text="🔬 Skin Cancer Diagnosis", 
    font=("Arial", 18, "bold"), bg="#007BFF", fg="white", pady=10
)
header_label.pack(fill=tk.X)

# Button to upload image
btn_select = ttk.Button(root, text="📸 Select an Image", command=load_and_predict)
btn_select.pack(pady=20)

# Image display area
img_label = tk.Label(root, bg="#34495E", relief=tk.SOLID, bd=2)
img_label.pack(pady=10)

# Prediction result
result_label = tk.Label(
    root, text="🔎 Waiting for analysis...", font=("Arial", 14, "bold"), 
    bg="#34495E", fg="white"
)
result_label.pack(pady=20)

# Progress bar (hidden by default)
progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
progress.pack(pady=10)
progress.pack_forget()  # Initially hidden

# Button to show developers
team_button = tk.Button(
    root, text="💻 Developer", font=("Arial", 12), bg="#FFC107", 
    fg="black", command=show_team
)
team_button.pack(pady=10)

# Run application
root.mainloop()
