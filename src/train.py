import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# Paths
# ==============================

TRAIN_PATH = "Dataset_Split/train"
VAL_PATH = "Dataset_Split/val"
MODEL_PATH = "models/cricket_player_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# ==============================
# Settings
# ==============================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 20

os.makedirs("models", exist_ok=True)

# ==============================
# Load Dataset
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# ==============================
# Class Names
# ==============================

class_names = train_dataset.class_names

print("\nClass names:")
for i, name in enumerate(class_names):
    print(i, name)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

# ==============================
# Improve Performance
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==============================
# Data Augmentation
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
])

# ==============================
# MobileNetV2
# ==============================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# ==============================
# Stage 1
# Freeze MobileNetV2
# ==============================

base_model.trainable = False

# ==============================
# Build Model
# ==============================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ==============================
# Stage 1 Compile
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==============================")
print("STAGE 1: TRAINING CLASSIFIER")
print("==============================\n")

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# ==============================
# Stage 1 Training
# ==============================

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=STAGE1_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

# ==============================
# Stage 2
# Fine Tune MobileNetV2
# ==============================

print("\n==============================")
print("STAGE 2: FINE-TUNING")
print("==============================\n")

base_model.trainable = True

# Freeze all except last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# ==============================
# Recompile
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Fine-Tuning
# ==============================

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=STAGE2_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

# ==============================
# Final Evaluation
# ==============================

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Model saved at: {MODEL_PATH}"
)

print(
    f"Class names saved at: {CLASS_NAMES_PATH}"
)