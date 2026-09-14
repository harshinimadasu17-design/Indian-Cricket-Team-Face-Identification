import os
import cv2

# ==========================================
# Paths
# ==========================================

SOURCE_DATASET = "Dataset"
OUTPUT_DATASET = "Dataset_Faces"

# Supported image formats
VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# ==========================================
# Create output folder
# ==========================================

os.makedirs(OUTPUT_DATASET, exist_ok=True)

# ==========================================
# Load Haar Cascade Face Detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Could not load face detector.")
    exit()

# ==========================================
# Counters
# ==========================================

total_images = 0
successful_faces = 0
no_face = 0
invalid_images = 0

# ==========================================
# Process each player folder
# ==========================================

for player_name in sorted(os.listdir(SOURCE_DATASET)):

    player_path = os.path.join(
        SOURCE_DATASET,
        player_name
    )

    # Ignore files; process only folders
    if not os.path.isdir(player_path):
        continue

    # Create corresponding output folder
    output_player_path = os.path.join(
        OUTPUT_DATASET,
        player_name
    )

    os.makedirs(
        output_player_path,
        exist_ok=True
    )

    player_total = 0
    player_success = 0

    print("\n====================================")
    print("Processing:", player_name)
    print("====================================")

    # ======================================
    # Process each image
    # ======================================

    for filename in os.listdir(player_path):

        if not filename.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        total_images += 1
        player_total += 1

        input_path = os.path.join(
            player_path,
            filename
        )

        # Read image
        image = cv2.imread(input_path)

        if image is None:
            print("Invalid image:", filename)
            invalid_images += 1
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================
        # Detect faces
        # ==================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # No face detected
        if len(faces) == 0:
            print("No face:", filename)
            no_face += 1
            continue

        # ==================================
        # Select largest face
        # ==================================

        largest_face = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        x, y, w, h = largest_face

        # ==================================
        # Add small margin around face
        # ==================================

        margin = 20

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)

        x2 = min(
            image.shape[1],
            x + w + margin
        )

        y2 = min(
            image.shape[0],
            y + h + margin
        )

        face = image[
            y1:y2,
            x1:x2
        ]

        # ==================================
        # Check cropped face
        # ==================================

        if face.size == 0:
            print("Invalid face crop:", filename)
            invalid_images += 1
            continue

        # ==================================
        # Save cropped face
        # ==================================

        output_path = os.path.join(
            output_player_path,
            filename
        )

        success = cv2.imwrite(
            output_path,
            face
        )

        if success:
            successful_faces += 1
            player_success += 1
        else:
            print("Could not save:", filename)
            invalid_images += 1

    print(
        f"{player_name}: "
        f"{player_success}/{player_total} faces saved"
    )


# ==========================================
# Final Report
# ==========================================

print("\n")
print("==========================================")
print("       DATASET PREPARATION COMPLETED")
print("==========================================")

print(
    f"Total images found: {total_images}"
)

print(
    f"Faces successfully saved: {successful_faces}"
)

print(
    f"No face detected: {no_face}"
)

print(
    f"Invalid/failed images: {invalid_images}"
)

print(
    f"Output dataset: {OUTPUT_DATASET}"
)

print("==========================================")
