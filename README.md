# KITTI to YOLO Training & Evaluation Pipeline

## Overview

This repository provides a complete, modular pipeline for training YOLO models on the KITTI Object Detection Dataset & RAG‑Powered LLM Query Engine (Llama3 + FAISS). It includes:

- Automated dataset download  
- KITTI <-> YOLO label conversion  
- Train/val split  
- Model training on Google Colab GPU  
- Evaluation (mAP50 & mAP50-95) 
- Model benchmarking across FP32, FP32-GPU, FP16, and INT8
- FAISS vector index and document store used by the RAG‑powered query application
- RAG‑Powered LLM Query Engine (Llama3 + FAISS)
- Visualization of predictions vs. ground truth  
- Sample video inference results  

The entire workflow is orchestrated through a Colab notebook for reproducibility and GPU acceleration.

---

## Repository Structure

'-- download_kittidataset.py\
'-- label_convertor.py\
'-- evaluate_yolo.py\
'-- benchmark_model.py\
'-- visualize_predictions.py\
'-- generate_faiss_doc.py\
'-- llmquery_app.py\
'-- data/  <-- generated files & label format.txt\
'-- google_collab_trigger_training.ipynb\
'-- kitti.yaml\
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
- onnxruntime-tools
- glob2
- faiss-cpu
- streamlit
- sentence-transformers


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

### 2. `label_convertor.py`
- Converts KITTI label format <-> YOLO label format based on input arg as `convert_typ`
- Normalizes bounding boxes  
- Creates the YOLO folder structure:
kitti_yolo/\
|-- images/\
&emsp;'-- train/\
&emsp;'-- val/\
|-- labels/\
&emsp;'-- train/\
&emsp;'-- val/

The yolo to kitti converted labels are available by default in `runs\detect\predict\kitti_labels`


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


### 5: `benchmark_model.py`

This script benchmarks a trained YOLO model across multiple precisions:

- **FP32 (CPU)**
- **FP32 (GPU)**
- **FP16 (GPU)**
- **INT8 (CPU, PTQ)**

For each precision, it reports:

- `mAP50`  
- `mAP50-95`  
- `latency_ms` (average inference latency per image)

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

### 6: `generate_faiss_doc.py`

This script now builds **two FAISS indexes**:

# A. Scene‑Level Index

Used for natural‑language queries about scene content:

- number of cars / pedestrians / cyclists

- maximum occlusion

- maximum truncation

- fuzzy terms (crowded, busy, heavy occlusion, etc.)

Each frame produces a document containing:

- metadata (counts, occlusion, truncation)

- summary text

- image path

- FAISS embedding

# B. Error‑Level Index (FP/FN Analysis) (NEW)

The script now:

- Loads KITTI ground‑truth labels

- Loads YOLO predictions converted to KITTI format

- Computes IoU for each GT–prediction pair

Identifies:

# False Positives (FP)

# False Negatives (FN)

Stores detailed error documents containing:

- error_type (FP or FN)

- class

- iou

- bbox

- occlusion_level (from GT)

- truncation_value (from GT)

- summary_text

- image_path

Stores:

`error_docs.json` → list of detailed error documents with error type and class

`error_index.faiss` → FAISS index for fast similarity search

`kitti_docs.json` → list of documents with metadata + summary

`kitti_index.faiss` → FAISS index for fast similarity search

`embedding_model.txt` → name of the embedding model used

These are generated into `data/`

---

### 7: `llmquery_app.py`

This Streamlit application lets you query in 2 modes the KITTI dataset using natural language, combining:

# 1. Scene Search Mode

**Numeric filtering** (e.g., “more than 5 pedestrians”)

**Fuzzy interpretation** (e.g., “crowded”, “busy”, “heavy occlusion”)

**Synonym expansion** (e.g., “packed”, “dense”, “traffic heavy”)

**Semantic search** using FAISS + SentenceTransformer

Local LLM reasoning using Ollama Llama3

**Capabilities**

Understands multi‑condition queries

Applies fuzzy rules from fuzzy_rules.json

Falls back to semantic search when numeric filters match nothing

Displays matching KITTI frames with images and metadata

# 2. Error Analysis Mode (NEW)

**FP/FN filtering** (e.g. "false positives for cyclists")

**IoU filtering** (e.g. "IoU < 0.4 errors" or "IoU > 0.7")

**Occlusion/truncation filtering** (e.g. "missed pedestrians with occlusion 3")

**Semantic search** using FAISS + SentenceTransformer

# Side‑by‑side visualization: GT (left) vs Predictions (right)

# Bounding‑box overlays:

- GT → green

- Predictions → yellow

- FP → red

- FN → blue

Only GT classes relevant to YOLO (Car, Pedestrian, Cyclist) are drawn.

**Running the LLM Query App**

1. **Start Ollama with Llama3**

Before launching the app, you must start the local LLM server:

`ollama run llama3`

This downloads the model (first time only) and starts the inference engine. 
Leave this terminal open.

2. **Run the Streamlit App**

In a new terminal:

`streamlit run llmquery_app.py`

The app will automatically:

Load FAISS index\
Load SentenceTransformer embeddings\
Load fuzzy rules\
Connect to Ollama\
Interpret your query\
Display matching KITTI frames or side by side depending on query mode

**Example Queries:**

# A. Scene Search 

"more than 5 pedestrians"\
"busy intersection with few cyclists"\
"heavy occlusion and more than 2 cars"\
"atleast 2 cars and more than 3 pedestrians"\
"crowded and heavy occlusion"\
"crowded and busy with rare cyclists"\
"few cyclists and rare pedestrians"

# B. Error Anaylsis

“false positives for cyclists”\
“missed pedestrians with occlusion 3”\
“IoU < 0.4 errors for cars”\
“FN for truncation > 0.5”

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
7. Runs precision benchmarking
8. Generate prediction label files (YOLO format)
9. Convert prediction label files: YOLO -> KITTI
10. generate faiss vector idnex and document store 

## `kitti.yaml` — Dataset Configuration File

This YAML file defines:

- Train image paths  
- Validation image paths  
- Class names  

Example: (available in `data/`)

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
- builds the FAISS vector index,  document store for GT & Error document generation (FP/FN/IoU/occ/trunc)
- Streamlit application lets you RAG‑powered LLM query engine for KITTI dataset using natural language
- Query engine for error analysis shows Side‑by‑side GT vs prediction visualization
- Future work will focus on TensorRT and OpenVINO acceleration   
