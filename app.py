from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = "dyslexia_model.h5"

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = cv2.resize(image, size)

    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        file = request.files["image"]

        if file:
            upload_path = "uploaded.jpg"
            file.save(upload_path)

            img = preprocess_image(upload_path)
            pred = model.predict(img)[0][0]

            if pred > 0.5:
                prediction_text = f"🧠 Dyslexia Detected (Confidence: {pred:.4f})"
            else:
                prediction_text = f"✅ Non-Dyslexia (Confidence: {pred:.4f})"

            # Optional: delete uploaded image after prediction
            if os.path.exists(upload_path):
                os.remove(upload_path)

    return render_template("index.html", prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True, port=5001)

