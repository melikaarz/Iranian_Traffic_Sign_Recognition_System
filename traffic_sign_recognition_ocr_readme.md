# 🚦 Iranian Traffic Sign Recognition System with Bilingual OCR (PTSD)

An end-to-end computer vision pipeline designed for **detecting, classifying, and recognizing text on Iranian traffic signs** using the **Persian Traffic Sign Dataset (PTSD)**. The project integrates deep learning recognition models (CNN / YOLO), adaptive image contrast enhancement (CLAHE), and a bilingual optical character recognition (OCR) engine for digit and text extraction.

---

## 🌟 Key Features

- **43-Class Recognition**: Full coverage of regulatory, warning, mandatory, and speed-limit traffic signs used across Iranian roadways.
- **Adaptive Contrast Enhancement (CLAHE)**: Specialized preprocessing step optimizing the readability of symbols and numeric digits under challenging illumination conditions.
- **Bilingual OCR Engine (Persian & English)**: GPU-accelerated text and digit extraction via EasyOCR with dynamic confidence filtering.
- **Data Augmentation Pipeline**: Geometric transformations, controlled zoom, rotation, and lighting adjustments to improve generalization and mitigate overfitting.
- **Automated Dataset Ingestion**: Automatic dataset downloading via `kagglehub` and programmatic splitting into balanced `train`, `val`, and `test` directories.

---

## 🏗 System Architecture & Pipeline

```plaintext
      [ Input Roadway Image ]
                 │
                 ▼
 [ Preprocessing & CLAHE Enhancement ]
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
[ Sign Classification ]  [ Cropped Sign Region ]
   (43 Target Classes)       │
                             ▼
                     [ Bilingual EasyOCR ]
                     (Persian / English)
                             │
                             ▼
                    [ Extracted Text & Digits ]
```

---

## 📊 Dataset Overview

This project utilizes the **Persian Traffic Sign Dataset (PTSD)** hosted on Kaggle:

- **Source**: [`saraparsaseresht/persian-traffic-sign-dataset-ptsd`](https://www.kaggle.com/datasets/saraparsaseresht/persian-traffic-sign-dataset-ptsd)
- **Classes**: 43 distinct traffic sign categories
- **Categories Included**:
  - **Speed Limits**: 30, 40, 50, 60, 70, 80, 90 km/h
  - **Prohibitory**: No Entry, No Stopping, No Horn, No Left/Right Turn, No Overtaking, etc.
  - **Warning / Hazard**: Speed Bumps, Sharp Curves, Pedestrian Crossing, School Zone, Road Work, etc.
  - **Mandatory & Priority**: STOP, Yield (Give Way), Roundabout, Directional Arrows, etc.

---

## 🛠 Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/iranian-traffic-sign-ocr.git
cd iranian-traffic-sign-ocr
pip install -r requirements.txt
```

### Key Dependencies

```text
ultralytics
easyocr
albumentations
kagglehub
opencv-python
matplotlib
numpy
torch
torchvision
```

---

## 🚀 Quickstart & Usage

### 1. Dataset Download & Organization

Run the automated ingestion script to download the dataset and partition images into training (85%) and validation (15%) subsets:

```python
import os
import shutil
import glob
import numpy as np
import kagglehub

# Download PTSD dataset
base_path = kagglehub.dataset_download("saraparsaseresht/persian-traffic-sign-dataset-ptsd")
ptsd_root = os.path.join(base_path, 'PTSD_Recognition', 'PTSD_Recognition')

print(f"Dataset downloaded to: {ptsd_root}")
```

### 2. Preprocessing with CLAHE

Enhance symbols and numerical details before model inference:

```python
import cv2
import numpy as np

def preprocess_for_digits(img: np.ndarray) -> np.ndarray:
    """Applies CLAHE enhancement to improve digit and symbol clarity."""
    img_uint8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
```

### 3. Running the Bilingual OCR Module

Extract Persian text or speed limit digits from detected signs:

```python
import cv2
import easyocr

# Initialize EasyOCR reader for Persian and English
reader = easyocr.Reader(['fa', 'en'], gpu=True)

def extract_sign_text(crop_img: np.ndarray, confidence_threshold: float = 0.30) -> str:
    """Extracts text and digits from a cropped traffic sign image."""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    
    extracted_tokens = [text for (_, text, prob) in results if prob > confidence_threshold]
    return " ".join(extracted_tokens) if extracted_tokens else "No text detected"
```

---

## 📁 Repository Structure

```text
├── data/
│   ├── train/                  # Class-sorted training images
│   ├── val/                    # Class-sorted validation images
│   └── test/                   # Benchmark test images
├── src/
│   ├── preprocessing.py        # CLAHE and image transformation utilities
│   ├── ocr_engine.py           # EasyOCR text and digit recognition module
│   ├── train.py                # Model training and hyperparameter setup
│   └── evaluate.py             # Evaluation metrics and confusion matrices
├── notebooks/
│   └── Traffic_Sign_PTSD.ipynb # End-to-end experimental notebook
├── requirements.txt
└── README.md
```

---

## 📌 Roadmap & Future Enhancements

- [ ] Deploy an interactive web interface using **Streamlit** or **Gradio**.
- [ ] Export trained checkpoints to **ONNX** and **TensorRT** for real-time edge execution (e.g., NVIDIA Jetson).
- [ ] Integrate multi-object tracking (**ByteTrack / DeepSORT**) for live video stream inferencing.
- [ ] Expand the OCR post-processing dictionary for Iranian street and highway names.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.