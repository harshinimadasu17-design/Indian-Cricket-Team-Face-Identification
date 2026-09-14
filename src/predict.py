import json
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/cricket_player_model.keras"
CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = (224, 224)

# ==========================================
# Load Model
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# Load Class Names
# ==========================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("Class names:", class_names)

# ==========================================
# Face Detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# Predict Player
# ==========================================

def predict_player(image_path):

    # Read original image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found.")
        return None, 0

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        print("No face detected.")
        return None, 0

    # Select largest face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    # Add margin
    margin = 20

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)

    x2 = min(
        img.shape[1],
        x + w + margin
    )

    y2 = min(
        img.shape[0],
        y + h + margin
    )

    # Crop face
    face = img[
        y1:y2,
        x1:x2
    ]

    # Convert BGR to RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Resize exactly like training
    face = cv2.resize(
        face,
        IMAGE_SIZE
    )

    # Convert to float
    face = face.astype(np.float32)

    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Same preprocessing as train.py
    face = tf.keras.applications.mobilenet_v2.preprocess_input(
        face
    )

    # Prediction
    predictions = model.predict(
        face,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_player = class_names[
        predicted_index
    ]

    confidence = (
        predictions[0][predicted_index] * 100
    )

    return predicted_player, confidence


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    image_path = "test_images/virat_test.png"

    player, confidence = predict_player(
        image_path
    )

    if player is not None:

        print("\n==============================")
        print("PREDICTION RESULT")
        print("==============================")

        print(
            f"Player: {player.replace('_', ' ')}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )