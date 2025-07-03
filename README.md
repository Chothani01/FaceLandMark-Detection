# 👤💻 Face Mesh Detection using MediaPipe and OpenCV

This project demonstrates **real-time face mesh detection** using [MediaPipe](https://google.github.io/mediapipe/) and OpenCV in Python.  
It includes two scripts: a **class-based modular version** and a **simple direct version**, to help you understand and build flexible face mesh solutions.

---

## 📂 Files

### ✅ `FaceMeshModule.py` (Class-based)

- Contains a `FaceMeshDetector` class.
- Supports multiple faces (default: 2).
- Allows you to easily adjust:
  - `staticMode`
  - `maxFaces`
  - `refine_landmarks`
  - `minDetectionCon`
  - `minTrackCon`
- Provides:
  - `findFaceMesh(img, draw=True)`: Detects and optionally draws mesh.
  - `findFaceLandmarks(img)`: Returns list of landmark points for each face.
- Prints landmarks of the first detected face in the console.

---

### ✅ `FaceMeshBasic.py` (Simple direct example)

- Direct usage of MediaPipe `FaceMesh`.
- Tracks one face by default.
- Prints all landmark coordinates continuously.
- Draws face mesh contours on the webcam feed.

---

## 🚀 Features

- Real-time face mesh tracking (468 landmarks).
- FPS display on the frame.
- Configurable for one or multiple faces.
- Option to refine landmarks for iris tracking.

---

## ⚙️ Installation

```bash
pip install opencv-python mediapipe numpy
