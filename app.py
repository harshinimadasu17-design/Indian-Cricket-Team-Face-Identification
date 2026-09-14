import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ==============================
# Configuration
# ==============================

MODEL_PATH = "models/cricket_player_model.keras"
CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = (224, 224)

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Indian Cricket Player Identification",
    page_icon="🏏",
    layout="centered"
)

# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ==============================
# Load Classes
# ==============================

@st.cache_data
def load_class_names():

    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


model = load_model()
class_names = load_class_names()

# ==============================
# Title
# ==============================

st.title(
    "🏏 Indian Cricket Player Face Identification"
)

st.write(
    "Upload an image of an Indian cricket player "
    "to identify the player using a trained "
    "deep learning model."
)

# ==============================
# Upload Image
# ==============================

uploaded_file = st.file_uploader(
    "Upload Player Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded_file is not None:

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Identify Player"):

        # Resize image
        resized_img = img.resize(
            IMAGE_SIZE
        )

        # Convert to NumPy
        img_array = np.array(
            resized_img
        )

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Preprocess
        img_array = (
            tf.keras.applications
            .mobilenet_v2
            .preprocess_input(
                img_array
            )
        )

        # Prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_player = class_names[
            predicted_index
        ]

        confidence = (
            predictions[0][predicted_index]
            * 100
        )

        # ==============================
        # Result
        # ==============================

        st.success(
            "Player: "
            + predicted_player.replace(
                "_", " "
            )
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # ==============================
        # Top 3 Predictions
        # ==============================

        st.subheader(
            "Top 3 Predictions"
        )

        top_indices = np.argsort(
            predictions[0]
        )[-3:][::-1]

        for index in top_indices:

            player = class_names[
                index
            ].replace(
                "_", " "
            )

            probability = (
                predictions[0][index] * 100
            )

            st.write(
                f"**{player}** — "
                f"{probability:.2f}%"
            )