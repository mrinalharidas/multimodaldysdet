import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

import kagglehub


# ==============================
# Download Dataset
# ==============================
path = kagglehub.dataset_download("oussamaslmani/dyslexic")
print("Path to dataset files:", path)

non_dyslexia_path = os.path.join(path, "dyslexic", "no")
dyslexia_path = os.path.join(path, "dyslexic", "yes")


# ==============================
# Preprocessing Functions
# ==============================
def preprocess_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Binarize
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Resize
    image = cv2.resize(image, size)

    return image


def load_images(folder_path, label, size=(128, 128)):
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            img = preprocess_image(file_path, size=size)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return images, labels


# ==============================
# Load Dataset
# ==============================
size = (128, 128)

non_dys_images, non_dys_labels = load_images(non_dyslexia_path, 0, size)
dys_images, dys_labels = load_images(dyslexia_path, 1, size)

images = np.array(non_dys_images + dys_images)
labels = np.array(non_dys_labels + dys_labels)

# Normalize
images = images / 255.0

# Expand dims for CNN input
images = np.expand_dims(images, axis=-1)

# Convert grayscale -> 3 channel (for VGG16)
images = np.repeat(images, 3, axis=-1)


# ==============================
# Split Data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)


# ==============================
# Build Model
# ==============================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# Train
# ==============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)


# ==============================
# Evaluate
# ==============================
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Non-Dyslexia", "Dyslexia"]))


# ==============================
# Save Model
# ==============================
model.save("dyslexia_model.h5")
print("\n✅ Model saved as dyslexia_model.h5")
