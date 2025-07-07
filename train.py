import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Paths
base_dir = "E:\\Skin Cancer Detection"
dataset_dir = os.path.join(base_dir, "dataset_Skin_Cancer")
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

model_path = os.path.join(base_dir, "skin_cancer_model.keras")
epoch_file_path = os.path.join(base_dir, "last_epoch.txt")
history_path = os.path.join(base_dir, "training_history.npy")

# Hyperparameters
image_size = (240, 240)
batch_size = 32
initial_lr = 1e-4
fine_tune_lr = 1e-5
initial_epochs = 25
fine_tune_epochs = 15
total_epochs = initial_epochs + fine_tune_epochs

# Image Data Generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=True
)
val_data = val_datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)

# Load/save training history
def load_history(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return {}

def save_history(new_history, path):
    if os.path.exists(path):
        full_history = np.load(path, allow_pickle=True).item()
        for key in new_history.history:
            full_history[key] += new_history.history[key]
    else:
        full_history = new_history.history
    np.save(path, full_history)

# Custom callbacks
class EpochSaver(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        with open(epoch_file_path, "w") as f:
            current_epoch = epoch + self.params.get("initial_epoch", 0)
            f.write(str(current_epoch))

class HistorySaver(Callback):
    def __init__(self, path):
        self.path = path
        self.history_data = load_history(path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history_data.setdefault(key, []).append(value)
        np.save(self.path, self.history_data)

# Resume from saved epoch if exists
if os.path.exists(epoch_file_path):
    with open(epoch_file_path, "r") as f:
        last_epoch = int(f.read())
    print(f"🔁 Resuming from epoch {last_epoch}")
else:
    last_epoch = 0
    print("🔰 Starting training from scratch")

# Load or create model
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"📦 Model loaded from: {model_path}")

    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'name') and 'efficientnet' in layer.name.lower():
            base_model = layer
            model.base_model = base_model
            print(f"🔄 base_model '{layer.name}' restored from loaded model.")
            break
    if not base_model:
        print("⚠️ EfficientNet base model NOT FOUND.")
else:
    base_model = EfficientNetB1(include_top=False, weights='imagenet', input_shape=(240, 240, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.base_model = base_model
    model.compile(optimizer=Adam(learning_rate=initial_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("✅ New EfficientNetB1 model created")

model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=1e-6),
    EpochSaver(),
    HistorySaver(history_path)
]

# Phase 1 - Initial Training
if last_epoch < initial_epochs:
    model.compile(optimizer=Adam(learning_rate=initial_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data,
              validation_data=val_data,
              epochs=initial_epochs,
              initial_epoch=last_epoch,
              callbacks=callbacks)
    last_epoch = initial_epochs

# Phase 2 - Fine-Tuning
if last_epoch >= initial_epochs and last_epoch < total_epochs:
    base_model = getattr(model, 'base_model', None)

    if base_model:
        base_model.trainable = True
        for layer in base_model.layers[:150]:
            layer.trainable = False
        print(f"🔧 Fine-Tuning: {base_model.name} partially unfrozen")
    else:
        print("⚠️ EfficientNet base model not found. Skipping fine-tuning step.")

    model.compile(optimizer=Adam(learning_rate=fine_tune_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data,
              validation_data=val_data,
              epochs=total_epochs,
              initial_epoch=last_epoch,
              callbacks=callbacks)

# Final Evaluation
best_model = tf.keras.models.load_model(model_path)
full_history = load_history(history_path)

test_loss, test_acc = best_model.evaluate(test_data)
train_acc = full_history['accuracy'][-1]
val_acc = full_history['val_accuracy'][-1]

print(f"\n🎯 Final Training Accuracy: {train_acc:.4f}")
print(f"🎯 Final Validation Accuracy: {val_acc:.4f}")
print(f"🧪 Final Test Accuracy: {test_acc:.4f}")

# Plot Accuracy & Loss
epochs_range = range(1, len(full_history['accuracy']) + 1)
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, full_history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, full_history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, full_history['loss'], label='Train Loss')
plt.plot(epochs_range, full_history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
metrics_path = os.path.join(base_dir, "metrics_curves.png")
plt.savefig(metrics_path)
plt.show()
print(f"📊 Accuracy/Loss curves saved to: {metrics_path}")

# Confusion Matrix
y_probs = best_model.predict(test_data)
y_preds = np.argmax(y_probs, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_preds)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
cm_path = os.path.join(base_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"🧠 Confusion Matrix saved to: {cm_path}")

# Classification Report
report = classification_report(y_true, y_preds, target_names=class_names)
print("\n📋 Classification Report:")
print(report)

report_path = os.path.join(base_dir, "classification_report.txt")
with open(report_path, "w", encoding='utf-8') as f:
    f.write(report)
print(f"📝 Classification report saved to: {report_path}")
