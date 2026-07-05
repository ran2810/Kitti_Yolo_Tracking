"""
KITTI Caption Quality Test — InstructBLIP vs BLIP-2, first 10 images
======================================================================
Tests InstructBLIP-Flan-T5-XL against BLIP-2-OPT on 10 KITTI frames.
InstructBLIP is specifically trained for instruction following and gives
detailed object-level descriptions instead of generic scene labels.

Runtime: ~2 minutes on T4 GPU.

Steps:
  1. Mount Drive (images already unzipped from previous run)
  2. Run cells — compare outputs side by side
  3. If InstructBLIP looks good, update full caption script to use it
"""


# ── CELL 1 — Install & mount ──────────────────────────────────────────────────

import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "accelerate", "pillow"])

from google.colab import drive
drive.mount("/content/drive")


# ── CELL 2 — Point to images (already unzipped) ───────────────────────────────

import os, zipfile

DRIVE_ZIP_PATH = "/content/drive/MyDrive/kitti_image_2.zip"
EXTRACT_PATH   = "/content/kitti_images/"

os.makedirs(EXTRACT_PATH, exist_ok=True)
if not os.listdir(EXTRACT_PATH):
    print("Unzipping...")
    with zipfile.ZipFile(DRIVE_ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_PATH)

image_files = sorted([f for f in os.listdir(EXTRACT_PATH) if f.endswith(".png")])
TEST_FILES  = image_files[:10]
print(f"Testing on {len(TEST_FILES)} images: {TEST_FILES}")


# ── CELL 3 — Load InstructBLIP-Flan-T5-XL ────────────────────────────────────
# Flan-T5 is instruction-tuned so it follows "describe vehicles..." literally.
# Flan-T5-XL (3B) fits comfortably on T4 in float16.

import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

MODEL_ID = "Salesforce/instructblip-flan-t5-xl"
print(f"Loading {MODEL_ID} ...")

processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model ready.")


# ── CELL 4 — Define prompt & caption function ─────────────────────────────────

from PIL import Image

# Instruction tells the model exactly what to enumerate.
# Flan-T5 follows this reliably unlike OPT which ignores it.
PROMPT = (
    "Describe in detail all vehicles, pedestrians, cyclists, road signs, "
    "workers, and any unusual objects visible in this driving scene. "
    "Be specific about counts, positions, and appearance."
)

def caption_one(img_path):
    img    = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, text=PROMPT, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=5,
            length_penalty=1.2,   # encourage longer, more complete answers
        )
    return processor.decode(out[0], skip_special_tokens=True).strip()


# ── CELL 5 — Run on 10 images and print ──────────────────────────────────────

print("\n" + "="*70)
print(f"InstructBLIP ({MODEL_ID})")
print(f"Prompt: {PROMPT[:80]}...")
print("="*70)

import json

captions = {}
for fname in TEST_FILES:
    path     = os.path.join(EXTRACT_PATH, fname)
    frame_id = fname.rsplit(".", 1)[0]
    cap      = caption_one(path)
    captions[frame_id] = cap
    print(f"\n[{fname}]")
    print(f"  {cap}")

# ── CELL 6 — Keyword coverage check ──────────────────────────────────────────

keywords = ["car", "vehicle", "person", "pedestrian", "cyclist", "bicycle",
            "sign", "truck", "bus", "train", "worker", "parked"]
print("\nKeyword hits across 10 captions:")
for kw in keywords:
    n = sum(1 for c in captions.values() if kw in c.lower())
    bar = "█" * n
    print(f"  {kw:12s}: {bar} {n}/10")

# ── CELL 7 — Save for inspection ─────────────────────────────────────────────

out_path = "/content/drive/MyDrive/kitti_captions_instructblip_10.json"
with open(out_path, "w") as f:
    json.dump(captions, f, indent=2)
print(f"\nSaved → {out_path}")
print("If descriptions look good, proceed with full 7,481-frame run.")
