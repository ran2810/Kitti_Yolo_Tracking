# KITTI to YOLO Training & Evaluation Pipeline

## Overview

This repository provides a complete, modular pipeline for training YOLO models on the KITTI Object Detection Dataset. It includes:

- Automated dataset download  
- KITTI -> YOLO label conversion  
- Train/val split  
- Model training on Google Colab GPU  
- Evaluation (mAP50 & mAP50-95) 
- **Model benchmarking across FP32, FP32-GPU, FP16, and INT8**
- Visualization of predictions vs. ground truth  
- Sample video inference results  

The entire workflow is orchestrated through a Colab notebook for reproducibility and GPU acceleration.

---

## Repository Structure

'-- download_kittidataset.py
'-- convert_kitti_to_yolo.py
'-- evaluate_yolo.py
'-- benchmark_model.py          <-- NEW: Precision & latency benchmarking
'-- visualize_predictions.py
'-- google_collab_trigger_training.ipynb
'-- kitti.yaml
'-- README.md

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
|-- images/
|   '-- train/
|   '-- val/
|-- labels/
    '-- train/
    '-- val/


- Performs a train/val split  
- Generates the `kitti.yaml` configuration file  

### 3. `evaluate_yolo.py`
- Loads a trained YOLO model  
- Evaluates on the validation set  
- Outputs mAP50, mAP50-95, precision, and recall  

### 4. `visualize_predictions.py`
- Loads validation images  
- Draws ground truth bounding boxes  
- Runs YOLO inference and draws predicted bounding boxes  
- Displays side-by-side comparison for qualitative inspection  


## 5: `benchmark_model.py`

This script benchmarks a trained YOLO model across multiple precisions:

- **FP32 (CPU)**
- **FP32 (GPU)**
- **FP16 (GPU)**
- **INT8 (CPU, PTQ)**

For each precision, it reports:

- `mAP50`  
- `mAP50-95`  
- `latency_ms` (average inference latency per image)

The script is automatically triggered inside:

### `google_collab_trigger_training.ipynb`

after training and evaluation are complete.

---

## Benchmark Results

Below are the results obtained from running `benchmark_model.py` on the trained YOLO model:

| Precision     | mAP50   | mAP50–95 | Latency (ms) |
|---------------|---------|----------|--------------|
| **FP32 (CPU)**     | 0.8648 | 0.5959   | 146.58       |
| **FP32 (GPU)**     | 0.8648 | 0.5959   | 14.35        |
| **FP16 (GPU)**     | 0.8647 | 0.5950   | 14.86        |
| **INT8 (CPU)**     | 0.8682 | 0.5976   | **723.84**   |

---

## Benchmark Summary

- **Accuracy:**  
  INT8 surprisingly achieved slightly higher mAP50/mAP95 than FP32.  
  This can happen due to quantization acting as a regularizer on structured datasets like KITTI.

- **Latency:**  
  - GPU inference (FP32/FP16) is ~10× faster than CPU FP32.  
  - FP16 did **not** provide additional speedup over FP32 on this backend.  
  - INT8 latency was **significantly slower** because the current environment lacks optimized INT8 kernels (TensorRT, OpenVINO, or ONNX Runtime EP).  
    As a result, INT8 falls back to slower CPU execution.

- **Conclusion:**  
  The model is robust across precisions, but **hardware-aware deployment** is required to unlock real INT8/FP16 speedups.

---

## Future Work

To achieve true acceleration from quantization and mixed precision, the next steps are:

### 🔹 **1. TensorRT FP16 & INT8 Deployment**
- Export YOLO -> ONNX -> TensorRT engine  
- Use TensorRT calibrators for INT8  
- Expect 3-6x speedup with minimal accuracy drop  

### 🔹 **2. OpenVINO INT8 Optimization (Intel CPUs)**
- Use Post-Training Quantization (PTQ)  
- Expect 2-4x CPU speedup  
- Very stable accuracy on KITTI  

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
6. **Runs precision benchmarking (`benchmark_model.py`)**  
7. Visualizes predictions  
8. Saves results back to Drive  

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
---

## Final Notes

- Each script is modular and can run independently  
- The Colab notebook ties everything together  
- Supports YOLOv8, YOLOv9, YOLOv10, YOLOv11 
- Results are reproducible and easy to extend  
- Benchmarking module provides deeper insight into deployment performance  
- Future work will focus on TensorRT and OpenVINO acceleration   