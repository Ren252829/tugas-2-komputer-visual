import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import kagglehub
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Unduh Dataset dari Kaggle ===
print("📥 Mengunduh dataset dari Kaggle...")
downloaded_path = kagglehub.dataset_download("yusufberksardoan/traffic-detection-project")
print(f"✅ Dataset berhasil diunduh di: {downloaded_path}")

# === Path Dataset ===
base_path = os.path.join(downloaded_path, "train")
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

# === Mapping Label ===
label_map = {
    "0": "bicycle",
    "1": "bus",
    "2": "car",
    "3": "motorbike",
    "4": "person"
}

image_size = (128, 128)

# === Load Gambar dan Label ===
def load_images_and_labels(image_dir, label_dir):
    images = []
    labels = []
    image_paths = []
    for label_file in tqdm(os.listdir(label_dir), desc="🔄 Loading labels"):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            label_id = lines[0].split()[0]
            label = label_map.get(label_id, None)
            if label is None:
                continue

        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, img_file)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(label)
        image_paths.append(img_path)
    return np.array(images), np.array(labels), image_paths

# === Load Dataset ===
X, y, image_paths = load_images_and_labels(images_path, labels_path)

# === Normalisasi Data dan One-Hot Label ===
X = X.astype("float32") / 255.0
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# === Split Dataset ===
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y_encoded, image_paths, test_size=0.2, random_state=42
)

# === Model CNN ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(lb.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Training ===
print("🚀 Melatih CNN...")
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=32)

# === Evaluasi ===
print("\n📊 Evaluasi Model:")
loss, acc = model.evaluate(X_test, y_test)
print(f"Akurasi: {acc:.4f}")

# === Prediksi dan Laporan ===
y_pred = model.predict(X_test)
y_pred_labels = lb.inverse_transform(y_pred)
y_true_labels = lb.inverse_transform(y_test)

from sklearn.metrics import classification_report
print("\n=== HASIL EVALUASI ===")
print(classification_report(y_true_labels, y_pred_labels))

# === Visualisasi Hasil ===
def show_predictions(images, y_true, y_pred, n=6):
    indices = np.random.choice(len(images), n, replace=False)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        img = cv2.imread(images[idx])
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(img)
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

print("\n📷 Menampilkan hasil klasifikasi acak:")
show_predictions(img_test, y_true_labels, y_pred_labels, n=6)

print("\n❌ Menampilkan gambar salah klasifikasi:")
def show_misclassified(images, y_true, y_pred, n=6):
    wrong_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    if not wrong_indices:
        print("✅ Semua gambar diklasifikasikan dengan benar.")
        return
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(wrong_indices[:n]):
        img = cv2.imread(images[idx])
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(img)
        plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}", color="red")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_misclassified(img_test, y_true_labels, y_pred_labels, n=6)
