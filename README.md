# Saffron Flower Detection & Plucking Point Prediction

This repository implements a **two-stage AI pipeline** for detecting saffron flowers and predicting the optimal plucking point using **YOLOv8** (object detection) and **HRNet** (keypoint regression).

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Pipeline](#pipeline)
* [Setup](#setup)
* [Training](#training)
* [Inference](#inference)
* [Results](#results)
* [License](#license)

---

## Overview

This project is designed for **automated saffron flower harvesting**. The workflow includes:

1. Detecting flowers using **YOLOv8**.
2. Cropping the detected flower bounding box.
3. Predicting the **plucking point** inside the bounding box using **HRNet-W18** regression.

This approach ensures accurate plucking point prediction, even with varying flower orientations.

---

## Features

* Two-stage AI system: **YOLOv8 + HRNet**.
* Handles **top-view and side-view flowers** (plucking point only for side-view).
* HRNet regression outputs **normalized (x, y) keypoints** in the bounding box.
* Supports **batch inference on multiple images**.
* Preserves **aspect ratio** to avoid distortion during HRNet training.
* **Data augmentation** for robust HRNet training (rotation, horizontal flip, color jitter).
* Mixed precision training for **faster training and lower GPU memory usage**.

---

## Dataset

* Dataset contains **two classes**: `top` (no plucking point) and `side` (plucking point visible).
* Annotation format for YOLOv8: `.txt` files containing:

  ```
  class x_center y_center width height
  ```
* Annotation format for HRNet regression (after cropping):

  ```
  class x_center y_center width height pluck_x pluck_y
  ```

  where `pluck_x` and `pluck_y` are normalized \[0,1] within the cropped flower.

---

## Pipeline

### 1. YOLOv8 Detection

* Detects flowers in images.
* Confidence threshold applied (default: 0.3).
* Outputs bounding boxes for each detected flower.

### 2. HRNet Keypoint Regression

* Crops the detected flower bounding box.
* HRNet-W18 predicts **plucking point** in normalized coordinates.
* Can process images in **original aspect ratio** using padding.

### 3. Combined Inference

* Loop through input images.
* Apply YOLOv8 detection → crop → HRNet prediction.
* Draw bounding boxes + predicted plucking points on output images.
* Save results to a specified folder.

---

## Setup

1. Clone the repository:

```bash
git clone <repo_url>
cd saffron_pluck_detection
```

2. Install required packages:

```bash
pip install torch torchvision timm ultralytics opencv-python tqdm pillow
```

3. Make sure your dataset structure is:

```
Dataset/
├── images/
│   ├── img1.jpg
│   └── ...
└── labels/
    ├── img1.txt
    └── ...
```

---

## Training

### 1. YOLOv8 Training

```python
from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # Pretrained weights
model.train(data='data.yaml', epochs=400, imgsz=640, device=0)
model.export(format='onnx', dynamic=True)
```

* `data.yaml` should include train/val paths and class names.
* Export weights to `best.pt`.

### 2. HRNet Training

```bash
python train_hrnet_pluck.py \
--data_dir final_dataset \
--epochs 50 \
--batch_size 16 \
--img_size 640 \
--out hrnet_pluck_best.pth
```

* Uses HRNet-W18 backbone (`timm`) + small regression head.
* Mixed precision training enabled.
* Best model saved as `_best.pth`.

**Important:** HRNet input images are cropped from YOLO bounding boxes and normalized keypoints are used.

---

## Inference

### Single Image

```python
from train_hrnet_pluck import predict_on_crop
px, py = predict_on_crop("hrnet_pluck_best.pth", "crop_image.jpg", img_size=640, device="cpu")
print("Normalized pluck point:", px, py)
```

### Batch Inference

```python
# Run YOLO + HRNet pipeline on folder
python run_yolo_hrnet_inference.py --input_folder test_images --output_folder results
```

* `confidence_thresh` defaults to 0.3 (can be adjusted).
* Output images contain bounding boxes and plucking point markers.

---

## Results

* YOLOv8 detects flowers with high precision.
* HRNet predicts plucking points with pixel distance metric (`avg ~few pixels`).
* Works robustly on various flower orientations and lighting conditions.

---

## Notes / Tips

* **Image size for HRNet:** 640 recommended for best accuracy.
* **Aspect ratio preservation** is important for HRNet regression.
* **Data augmentation** improves generalization for unseen flowers.
* Combine YOLO + HRNet for **real-time plucking automation**.

---

## License

This project is released under the **MIT License**.
