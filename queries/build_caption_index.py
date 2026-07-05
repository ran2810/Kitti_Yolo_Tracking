"""
Build Caption FAISS Index
=========================
Run this locally AFTER downloading kitti_captions.json from Colab to:
    H:/GitHub/Kitti_Yolo_Tracking/data/kitti_captions.json

Usage:
    cd H:/GitHub/Kitti_Yolo_Tracking/queries
    python build_caption_index.py

Outputs (both required by llmquery_app.py):
    ../data/caption_index.faiss
    ../data/caption_docs.json
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CAPTIONS_PATH       = "../data/kitti_captions.json"
KITTI_DOCS_PATH     = "../data/kitti_docs.json"
IMAGE_DIR           = "../data/training/image_2"
CAPTION_INDEX_PATH  = "../data/caption_index.faiss"
CAPTION_DOCS_PATH   = "../data/caption_docs.json"

# Same model already loaded in llmquery_app.py — no extra download needed
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    #  Load captions 
    print(f"Loading {CAPTIONS_PATH} ...")
    with open(CAPTIONS_PATH) as f:
        captions = json.load(f)
    print(f"  {len(captions)} captions loaded")

    # ── Load scene docs for image_path lookup ────────────────────────────────
    with open(KITTI_DOCS_PATH) as f:
        scene_docs = json.load(f)
    id_to_doc = {d["id"]: d for d in scene_docs}

    # ── Build caption docs list ──────────────────────────────────────────────
    # Combine BLIP caption with KITTI GT summary so object counts (cars,
    # pedestrians, cyclists, occlusion) are always searchable even when the
    # visual caption misses them (e.g. describes background trees instead).
    caption_docs = []
    for frame_id, caption in captions.items():
        doc = id_to_doc.get(frame_id, {})
        blip_text    = (caption or "").strip()
        summary_text = doc.get("summary_text", "")

        # Combined text: GT object summary + BLIP visual description
        # e.g. "Frame 000007 contains 3 cars, 1 pedestrians, and 0 cyclists.
        #        Max occlusion level is 1 ... A street lined with trees."
        combined = f"{summary_text} {blip_text}".strip()
        if not combined:
            continue

        caption_docs.append({
            "frame_id":    frame_id,
            "caption":     blip_text,        # shown in UI as the visual description
            "combined":    combined,          # what gets encoded into the index
            "image_path":  doc.get("image_path", f"{IMAGE_DIR}/{frame_id}.png"),
            "summary_text": summary_text,
        })

    print(f"  {len(caption_docs)} non-empty captions → building index")

    # ── Encode captions ──────────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [d["combined"] for d in caption_docs]   # encode combined text
    print("Encoding combined texts (GT summary + BLIP caption) ...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=256,
    ).astype("float32")

    # ── Build FAISS IndexFlatL2 (matches scene_index for consistency) ────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # ── Save ─────────────────────────────────────────────────────────────────
    faiss.write_index(index, CAPTION_INDEX_PATH)
    with open(CAPTION_DOCS_PATH, "w") as f:
        json.dump(caption_docs, f, indent=2)

    print(f"\nSaved:")
    print(f"  {CAPTION_INDEX_PATH}  ({len(caption_docs)} vectors, dim={dim})")
    print(f"  {CAPTION_DOCS_PATH}")

    # ── Sanity checks ────────────────────────────────────────────────────────
    test_queries = [
        "construction worker in high visibility vest",
        "pedestrian crossing sign on road",
        "road tunnel entrance",
        "snow covered road winter",
    ]
    print("\nSanity check — top 2 caption matches:")
    for tq in test_queries:
        q_emb = model.encode([tq], convert_to_numpy=True).astype("float32")
        D, I = index.search(q_emb, 2)
        print(f"  '{tq}':")
        for dist, idx in zip(D[0], I[0]):
            d = caption_docs[idx]
            print(f"    [{d['frame_id']}] dist={dist:.3f}  \"{d['caption'][:80]}\"")


if __name__ == "__main__":
    main()
