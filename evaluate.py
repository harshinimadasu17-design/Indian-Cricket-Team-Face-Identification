import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ==========================================
# SETTINGS
# ==========================================

TRAIN_PATH = "Dataset_Split/train"
VAL_PATH = "Dataset_Split/val"
MODEL_PATH = "models/cricket_player_model.keras"
CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)

# ==========================================
# LOAD VALIDATION DATASET
# ==========================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================
# ACCURACY
# ==========================================

accuracy = np.mean(y_true == y_pred)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)
# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)

# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(12, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=90,
    values_format="d"
)

plt.title("Indian Cricket Player Face Identification - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================
# FIND CONFUSED PLAYER PAIRS
# ==========================================

print("\n==========================================")
print("MOST CONFUSED PLAYER PAIRS")
print("==========================================")

confusions = []

for actual in range(len(class_names)):

    for predicted in range(len(class_names)):

        if actual != predicted:

            count = cm[actual][predicted]

            if count > 0:

                confusions.append(
                    (
                        count,
                        class_names[actual],
                        class_names[predicted]
                    )
                )

# Sort from highest confusion to lowest

confusions.sort(
    reverse=True
)

if len(confusions) == 0:

    print("No incorrect predictions found.")

else:

    for count, actual, predicted in confusions:

        print(
            f"{actual} -> {predicted}: {count} images"
        )

# ==========================================
# PER-PLAYER ACCURACY
# ==========================================

print("\n==========================================")
print("PER-PLAYER ACCURACY")
print("==========================================")

for i, player in enumerate(class_names):

    total_images = np.sum(cm[i])

    correct_images = cm[i][i]

    if total_images > 0:

        player_accuracy = (
            correct_images / total_images
        ) * 100

        print(
            f"{player}: "
            f"{player_accuracy:.2f}% "
            f"({correct_images}/{total_images})"
        )

# ==========================================
# SAVE RESULTS
# ==========================================

with open(
    "evaluation_results.txt",
    "w"
) as file:

    file.write(
        f"Validation Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "MOST CONFUSED PLAYER PAIRS\n"
    )

    file.write(
        "============================\n"
    )

    for count, actual, predicted in confusions:

        file.write(
            f"{actual} -> {predicted}: "
            f"{count} images\n"
        )

    file.write(
        "\nPER-PLAYER ACCURACY\n"
    )

    file.write(
        "====================\n"
    )

    for i, player in enumerate(class_names):

        total_images = np.sum(cm[i])

        correct_images = cm[i][i]

        if total_images > 0:

            player_accuracy = (
                correct_images / total_images
            ) * 100

            file.write(
                f"{player}: "
                f"{player_accuracy:.2f}% "
                f"({correct_images}/{total_images})\n"
            )

print("\n==========================================")
print("EVALUATION COMPLETED")
print("==========================================")

print("Confusion matrix saved as:")
print("confusion_matrix.png")

print("\nDetailed results saved as:")
print("evaluation_results.txt")