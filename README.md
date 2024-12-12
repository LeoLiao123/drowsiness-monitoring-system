# Drowsiness Monitoring System
A real-time drowsiness detection system using computer vision and facial landmarks detection. The system can identify users, monitor their alertness levels, and provide timely warnings when signs of drowsiness are detected.

Demo : https://www.youtube.com/watch?v=TBz_t0KP9rc

## Features

* Real-time face detection and recognition
* Drowsiness detection through eye aspect ratio (EAR) analysis
* Yawning detection through mouth aspect ratio analysis
* User verification system
* Data publishing to Ubidots IoT platform
* Real-time visual feedback and alerts

## Prerequisites

* Python 3.8+
* Webcam or USB camera
* CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/drowsiness_monitoring_system.git
cd drowsiness_monitoring_system
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download required model files:

* Download `shape_predictor_68_face_landmarks.dat` from dlib's official website
* Place it in the `resources/` directory



## Model Training

1. Prepare your training data:

* Create folders named face01, face02, etc. for each person
* Place facial images of each person in their respective folders
* Recommended: 20-30 images per person in different lighting conditions

2. Train the face recognition model:

```bash
cd model_training
python train.py
```

## Configuration

Configure the following settings in `config/settings.py`

* Ubidots API key
* Detection thresholds
* Camera settings
* User database

## Usage

```bash
python main.py
```
## Reference

[1] https://steam.oxxostudio.tw/category/python/ai/ai-face-recognizer.html

[2] http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

[3] [https://blog.csdn.net/cungudafa/article/details/103496881](https://www.cnblogs.com/lushuang55/p/17396900.html)