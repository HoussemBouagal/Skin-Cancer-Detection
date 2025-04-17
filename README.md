# Skin Cancer Detection using Deep Learning

## 📌 Overview
This project is a deep learning-based skin cancer detection system using Convolutional Neural Networks (CNNs). It allows users to upload an image of a skin lesion and predicts its category with a confidence score. The model is trained on the **Skin Cancer MNIST HAM10000** dataset.

## 🔥 Features
- **Deep Learning Model:** CNN model trained to classify 7 types of skin cancer.
- **Tkinter GUI:** User-friendly interface for easy image upload and prediction.
- **Image Preprocessing:** Automatic resizing and normalization of input images.
- **Real-time Prediction:** Generates a classification result with accuracy percentage.
- **Model Persistence:** Trained model is saved and can be reloaded for further use.

## 🏗️ Technologies Used
- **Python** (TensorFlow, Keras, NumPy, Pandas, Matplotlib, PIL, Tkinter)
- **Machine Learning** (CNN model, ImageDataGenerator)
- **Dataset:** HAM10000 (Skin Cancer MNIST dataset)

## 🚀 Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/skin-cancer-detection.git
   cd skin-cancer-detection
   ```
2. Install dependencies:
   ```sh
   pip install tensorflow numpy pandas matplotlib pillow scikit-learn
   ```
3. Run the application:
   ```sh
   python Interface.py
   ```

## 🎯 How It Works
1. **Train the Model:**
   - The dataset is loaded and preprocessed.
   - A CNN model is trained to classify skin lesions.
   - The trained model is saved as `skin_cancer_model.keras`.
2. **Use the GUI for Predictions:**
   - Run the Tkinter-based GUI.
   - Upload an image of a skin lesion.
   - The model predicts and displays the result with confidence percentage.

## 📊 Model Training Summary
- **Input Size:** 224x224 pixels
- **CNN Layers:** 3 Convolutional + MaxPooling layers
- **Activation Functions:** ReLU, Softmax
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Training Epochs:** 20

## 📸 Demo Screenshot
![GUI Screenshot](screenshot.png)

## 📜 License
This project is licensed under the **MIT License**.

## 🌟 Acknowledgments
- **HAM10000 Dataset**: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **TensorFlow & Keras** for deep learning.
