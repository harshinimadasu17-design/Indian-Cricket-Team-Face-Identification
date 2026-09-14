# Indian Cricket Team Member Face Identification

An AI-based computer vision project that identifies selected Indian cricket players from facial images using **OpenCV, TensorFlow/Keras, and MobileNetV2**. The project also provides a simple **Streamlit web interface** for uploading an image and getting the predicted player name, confidence score, and top-3 predictions.

## Problem Statement

Develop a machine learning/computer vision system that can identify a person from an image based on facial features. The system is trained using images of selected Indian cricket team members and classifies a new input image into the corresponding player's name.

## Objectives

* Collect and prepare images of selected Indian cricket players.
* Detect and extract faces from images.
* Preprocess images for model training.
* Build a deep learning image classification model.
* Identify a cricket player from a new face image.
* Provide an easy-to-use Streamlit interface.

## Players / Classes

The model is trained to classify **12 players**:

1. Hardik Pandya
2. Jasprit Bumrah
3. KL Rahul
4. Mohammed Siraj
5. Ravindra Jadeja
6. Rishabh Pant
7. Rohit Sharma
8. Shubman Gill
9. Suryakumar Yadav
10. Venkatesh Iyer
11. Virat Kohli
12. Yuzvendra Chahal

## Technologies Used

* **Python**
* **OpenCV** – face detection and image processing
* **TensorFlow / Keras** – deep learning
* **MobileNetV2** – transfer learning and feature extraction
* **NumPy**
* **Pillow**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit** – web interface

## System Workflow


Input Image
     ↓
Face Detection
     ↓
Face Cropping
     ↓
Image Preprocessing
     ↓
MobileNetV2
     ↓
Feature Extraction
     ↓
Player Classification
     ↓
Player Name + Confidence


## Dataset

* Images were collected from cricket-player image datasets.
* Face detection was performed using the **Haar Cascade classifier**.
* **499 face images** were successfully prepared.
* Dataset split:

  * **396 training images**
  * **103 validation images**
* Images were resized to **224 × 224 pixels**.
* Data augmentation was used during training.

## Model

The project uses **MobileNetV2 pretrained on ImageNet**.

### Model Configuration

* Input size: `224 × 224`
* Number of classes: `12`
* Transfer learning: Yes
* Fine-tuning: Yes
* Data augmentation: Yes
* Output: 12-class Softmax classification

## Streamlit Application

The application allows the user to:

1. Upload a cricket player's image.
2. Detect and process the face.
3. Run the trained model.
4. Display the predicted player.
5. Display the confidence score.
6. Display the top-3 predictions.

## Project Structure


Indian-Cricket-Team-Face-Identification/
│
├── Dataset/
├── Dataset_Faces/
├── Dataset_Split/
│   ├── train/
│   └── val/
│
├── models/
│   └── cricket_player_model.keras
│
├── src/
│   ├── face detection.py
│   ├── predict.py
│   └── train.py
│
├── test_images/
│
├── app.py
├── class_names.json
├── prepare_dataset.py
├── create_split.py
├── check_dataset.py
├── evaluate.py
├── requirements.txt
└── README.md


## Installation

Clone the repository:

```bash
git clone https://github.com/harshinimadasu17-design/Indian-Cricket-Team-Face-Identification.git
```

Navigate to the project folder:

```bash
cd Indian-Cricket-Team-Face-Identification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## Results

The model was evaluated using a separate validation dataset containing images from all 12 classes.

* Training images: **396**
* Validation images: **103**
* Validation accuracy: **approximately 48%**
* Evaluation: Classification report and confusion matrix
* The Streamlit application successfully performs image upload and prediction.

The accuracy varies between players because of differences in image quality, facial pose, lighting, and the limited number of training images.

## Limitations

* The dataset contains a limited number of images per player.
* Different lighting and facial poses can affect prediction.
* Some players have visually similar features in certain images.
* Prediction accuracy can vary depending on the input image.

## Future Scope

* Increase the number and diversity of training images.
* Improve face alignment and preprocessing.
* Use larger and more advanced face-recognition models.
* Improve model training and hyperparameter tuning.
* Add more Indian cricket players.
* Deploy the application online.

## Conclusion

The **Indian Cricket Team Member Face Identification** project demonstrates the use of computer vision and deep learning to classify cricket players from facial images. By combining **OpenCV face detection, MobileNetV2 transfer learning, and Streamlit**, the project provides an end-to-end AI-based face classification system.
