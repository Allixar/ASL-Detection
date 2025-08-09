# ASL to Speech Detection (Compatible with Python 3.12)

A Python-based American Sign Language (ASL) hand gesture recognition system that converts recognized signs into speech. Built using **OpenCV**, **MediaPipe**, **TensorFlow Lite**, and **pyttsx3**, it can operate in real time from a webcam or process images from a dataset.

## Features
- **Real-time ASL recognition** using your webcam
- **Speech output** for recognized gestures via `pyttsx3`
- **MediaPipe hand tracking** for accurate landmark detection
- **TensorFlow Lite model** for fast, on-device gesture classification
- **Data logging** for collecting new training samples
- **Visual feedback** with bounding boxes, finger outlines, and gesture labels

## Requirements

Install the dependencies:
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
opencv-python
mediapipe
tensorflow==2.17.1
Pillow
numpy
pandas
seaborn
scikit-learn
matplotlib
```

> If you need extra OpenCV modules, use:
> ```
> pip install opencv-contrib-python
> ```

## Project Structure
```
ASL-Detection/
├── app.py                                   # Main application
├── utils/
│   └── cvfpscalc.py                         # FPS calculator
├── model/
│   ├── dataset/                             # Gesture image datasets
│   └── keypoint_classifier/
│       ├── keypoint_classifier.py           # TFLite classifier script
│       ├── keypoint_classifier.tflite       # Pre-trained TFLite model
│       └── keypoint_classifier_label.csv    # Gesture label list
└── assets/
    └── om606.png                            # Loading screen image
```

## Usage

### Run with default settings
```bash
python app.py
```

### Run with custom camera and resolution
```bash
python app.py --device 0 --width 1280 --height 720
```

## Controls
- **n** → Inference mode (real-time ASL to speech)
- **k** → Capture landmarks from the camera for training
- **d** → Process landmarks from the dataset
- **ESC** → Exit the application

## Dataset
Your dataset should be placed in:
```
model/dataset/dataset 1/
```
with separate folders for each gesture class. Each folder name should match the gesture label in `keypoint_classifier_label.csv`.

## How it Works
1. Captures frames from the webcam or dataset images.
2. Detects hand landmarks using MediaPipe.
3. Normalizes and preprocesses landmark coordinates.
4. Classifies gestures with a TensorFlow Lite model.
5. Displays bounding boxes, finger joints, and gesture labels.
6. Speaks the recognized gesture if it changes.

## License
This project is intended for educational and research purposes only.
