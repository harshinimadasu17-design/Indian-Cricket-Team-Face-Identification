import os

DATASET_PATH = "Dataset_Faces"

total = 0

for player in sorted(os.listdir(DATASET_PATH)):
    folder = os.path.join(DATASET_PATH, player)

    if os.path.isdir(folder):
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        print(f"{player}: {len(images)}")
        total += len(images)

print("\nTOTAL:", total)


