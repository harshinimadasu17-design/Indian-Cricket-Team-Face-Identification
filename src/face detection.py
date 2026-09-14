import cv2


def detect_face(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        print("No face detected.")
        return None

    # Select largest face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face = image[
        y:y+h,
        x:x+w
    ]

    return face


if __name__ == "__main__":

    image_path = "test_images/virat_2.jpg"

    face = detect_face(image_path)

    if face is not None:

        cv2.imwrite(
            "test_images/detected_face.jpg",
            face
        )

        print(
            "Face detected and saved."
        )