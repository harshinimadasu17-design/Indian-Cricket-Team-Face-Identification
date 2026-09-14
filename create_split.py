import os
import shutil
import random

SOURCE = "Dataset_Faces"
DEST = "Dataset_Split"

TRAIN_DIR = os.path.join(DEST, "train")
VAL_DIR = os.path.join(DEST, "val")

random.seed(42)

# Remove old split if it exists
if os.path.exists(DEST):
    shutil.rmtree(DEST)

os.makedirs(TRAIN_DIR)
os.makedirs(VAL_DIR)

total_train = 0
total_val = 0

print("\nCreating balanced train/validation split...\n")

for player in sorted(os.listdir(SOURCE)):

    source_folder = os.path.join(SOURCE, player)

    if not os.path.isdir(source_folder):
        continue

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    # 80% training, 20% validation
    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_player_folder = os.path.join(TRAIN_DIR, player)
    val_player_folder = os.path.join(VAL_DIR, player)

    os.makedirs(train_player_folder)
    os.makedirs(val_player_folder)

    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(train_player_folder, image)
        )

    for image in val_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(val_player_folder, image)
        )

    print(
        f"{player}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} validation"
    )

    total_train += len(train_images)
    total_val += len(val_images)

print("\n================================")
print("SPLIT COMPLETED")
print("================================")

print(f"Training images: {total_train}")
print(f"Validation images: {total_val}")
print(f"Total images: {total_train + total_val}")