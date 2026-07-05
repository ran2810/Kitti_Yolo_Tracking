"""
KITTI Caption Generation — Google Colab Script
===============================================
Run this in Google Colab (T4 GPU runtime).

Steps before running:
  1. Zip your KITTI images folder locally:
        zip -r kitti_image_2.zip H:/GitHub/Kitti_Yolo_Tracking/data/training/image_2/
     Or on Windows PowerShell:
        Compress-Archive -Path "H:\GitHub\Kitti_Yolo_Tracking\data\training\image_2\*" `
                         -DestinationPath "H:\kitti_image_2.zip"

  2. Upload kitti_image_2.zip to Google Drive (any folder)

  3. Copy this script into a Colab notebook and run cell by cell.

Output:
  kitti_captions.json saved to your Google Drive → download to:
  H:/GitHub/Kitti_Yolo_Tracking/data/kitti_captions.json
"""


#  CELL 1 — Install dependencies 
# Paste into Colab cell and run

import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "accelerate", "pillow", "tqdm"])


#  CELL 2 — Mount Google Drive 

from google.colab import drive
drive.mount("/content/drive")


#  CELL 3 — Upload check: find your zip in Drive 
# Update DRIVE_ZIP_PATH to match where you uploaded the zip

import os

EXTRACT_PATH    = "/content/kitti_images/"
CAPTION_OUTPUT  = "/content/drive/MyDrive/kitti_captions.json" # saved back to Drive


image_files = sorted([
    f for f in os.listdir(EXTRACT_PATH)
    if f.lower().endswith(".png")
])
print(f"Extracted {len(image_files)} images")


#  CELL 5 — Load BLIP-2 model 
# Salesforce/blip2-opt-2.7b: best quality/speed on T4
# Runs in float16 → fits in 16GB VRAM with headroom

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# BLIP-2 with OPT supports conditional (prompted) generation via VQA mode.
# Passing a text prompt directs the LM to describe specific objects instead
# of defaulting to the most visually dominant element (usually background trees).
VQA_PROMPT = "Question: Describe all vehicles, pedestrians, signs, workers, and objects visible on or near the road. Answer:"

MODEL_ID = "Salesforce/blip2-opt-2.7b"
print(f"Loading {MODEL_ID} ...")

processor = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model loaded.")


#  CELL 6 — Generate captions 
# ~0.3-0.5s per image on T4 → ~45-60 min for 7,481 frames
# Saves a checkpoint every 500 frames in case of Colab timeout

from PIL import Image
from tqdm import tqdm
import json

CHECKPOINT_PATH = "/content/captions_checkpoint.json"
BATCH_SIZE = 8   # process in batches for speed

# Resume from checkpoint if exists
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        captions = json.load(f)
    print(f"Resuming from checkpoint: {len(captions)} frames done")
else:
    captions = {}

remaining = [f for f in image_files if f.rsplit(".", 1)[0] not in captions]
print(f"Remaining: {len(remaining)} frames")


def generate_batch(image_paths):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    # Pass the VQA prompt to each image — this conditions the LM on the question
    # so it describes objects on the road instead of background trees/sky.
    prompts = [VQA_PROMPT] * len(images)
    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
    ).to(device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,   # longer answers for detailed object lists
            num_beams=3,
        )
    # batch_decode returns "Question: ... Answer: <answer>" — strip the prompt prefix
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    clean = []
    for t in texts:
        # Remove the prompt prefix if present
        if "Answer:" in t:
            t = t.split("Answer:", 1)[-1]
        clean.append(t.strip())
    return clean


# Process in batches
for i in tqdm(range(0, len(remaining), BATCH_SIZE)):
    batch_files = remaining[i : i + BATCH_SIZE]
    batch_paths = [os.path.join(EXTRACT_PATH, f) for f in batch_files]

    try:
        batch_captions = generate_batch(batch_paths)
        for fname, caption in zip(batch_files, batch_captions):
            frame_id = fname.rsplit(".", 1)[0]
            captions[frame_id] = caption
    except Exception as e:
        # Fallback: process one by one
        for fname in batch_files:
            frame_id = fname.rsplit(".", 1)[0]
            try:
                img = Image.open(os.path.join(EXTRACT_PATH, fname)).convert("RGB")
                inputs = processor(images=img, text=VQA_PROMPT, return_tensors="pt").to(device, torch.float16)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=80)
                raw = processor.decode(out[0], skip_special_tokens=True).strip()
                if "Answer:" in raw:
                    raw = raw.split("Answer:", 1)[-1].strip()
                captions[frame_id] = raw
            except Exception as e2:
                print(f"  Skipping {fname}: {e2}")
                captions[frame_id] = ""

    # Checkpoint every 500 frames
    if (i // BATCH_SIZE) % (500 // BATCH_SIZE) == 0:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(captions, f)

print(f"\nDone. {len(captions)} captions generated.")


#  CELL 7 — Preview sample captions 

print("Sample captions:")
for fid, cap in list(captions.items())[:10]:
    print(f"  {fid}: {cap}")


#  CELL 8 — Save to Drive 

with open(CAPTION_OUTPUT, "w") as f:
    json.dump(captions, f, indent=2)

print(f"\nSaved to Drive: {CAPTION_OUTPUT}")
