# KITTI to YOLO Training & Evaluation Pipeline

## Overview

This repository provides a complete, modular pipeline for training YOLO models on the KITTI Object Detection Dataset. It includes:

- Automated dataset download  
- KITTI → YOLO label conversion  
- Train/val split  
- Model training on Google Colab GPU  
- Evaluation (mAP50 & mAP50–95)  
- Visualization of predictions vs. ground truth  
- Sample video inference results  

The entire workflow is orchestrated through a Colab notebook for reproducibility and GPU acceleration.

---

## Repository Structure

├── download_kittidataset.py
├── convert_kitti_to_yolo.py
├── evaluate_yolo.py
├── visualize_predictions.py
├── google_collab_trigger_training.ipynb
├── kitti.yaml
└── README.md

## Requirements

### Python Version
- Python 3.8 or higher

### Python Packages

Install all required packages using:

pip install -r requirements.txt


Or install manually:

- ultralytics  
- opencv-python  
- matplotlib  
- numpy  
- tqdm  
- requests  
- zipfile36 (if using Python < 3.10)  
- PyYAML  
- glob2  

### Google Colab Requirements

- GPU runtime enabled (Runtime → Change runtime type → GPU)  
- Google Drive mounted for persistent storage  

### Dataset

The pipeline automatically downloads:

- KITTI Object Detection Dataset (images + labels)

### Hardware

- Recommended: Google Colab GPU (T4, L4, or A100)  
- Not recommended: CPU-only training (extremely slow)

---

## Script Descriptions

### 1. `download_kittidataset.py`
- Downloads the KITTI object detection dataset (images + labels)  
- Unzips and organizes the dataset  
- Optionally visualizes a sample image with its KITTI label  
- Ensures the dataset is ready for conversion  

### 2. `convert_kitti_to_yolo.py`
- Converts KITTI label format → YOLO label format  
- Normalizes bounding boxes  
- Creates the YOLO folder structure:
kitti_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/


- Performs a train/val split  
- Generates the `kitti.yaml` configuration file  

### 3. `evaluate_yolo.py`
- Loads a trained YOLO model  
- Evaluates on the validation set  
- Outputs mAP50, mAP50–95, precision, and recall  

### 4. `visualize_predictions.py`
- Loads validation images  
- Draws ground truth bounding boxes  
- Runs YOLO inference and draws predicted bounding boxes  
- Displays side‑by‑side comparison for qualitative inspection  

---

## Training Workflow (Google Colab)

All scripts are orchestrated inside:

### `google_collab_trigger_training.ipynb`

This notebook:

1. Mounts Google Drive  
2. Downloads KITTI dataset  
3. Converts KITTI → YOLO  
4. Triggers YOLO training on GPU  
5. Evaluates the trained model  
6. Visualizes predictions  
7. Saves results back to Drive  

---

## `kitti.yaml` — Dataset Configuration File

This YAML file defines:

- Train image paths  
- Validation image paths  
- Class names  

Example:

path: kitti_yolo
train: images/train
val: images/val
names:
0: Car
1: Pedestrian
2: Cyclist

---

## Sample Video Inference

After training, inference results are saved to: runs/detect/predict/


This folder contains:

- Annotated video  
- Annotated frames  
- Prediction logs  

---

## How to Run the Pipeline

### Option A — Google Colab (recommended)

Open:

google_collab_trigger_training.ipynb


Run all cells sequentially.

### Option B — Local Machine (CPU only)

Scripts will run, but full training is slow without a GPU.

---

## Final Notes

- Each script is modular and can run independently  
- The Colab notebook ties everything together  
- Supports YOLOv8, YOLOv9, YOLOv10, YOLOv11  
- Results are reproducible and easy to extend  