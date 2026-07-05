"""
SearchAD CLIP Retrieval — KITTI Train Split
============================================
Generates a predictions JSON for all KITTI labels using the existing
CLIP image index (clip_index.faiss + clip_frame_ids.json).

For each label:
  1. Encode the label as a CLIP text query
  2. Search all 7,481 KITTI frames ranked by cosine similarity
  3. Write ranked list in SearchAD prediction format

Run:
    cd H:/GitHub/Kitti_Yolo_Tracking/searchad
    python generate_predictions.py

Then evaluate:
    python H:/GitHub/searchad_devkit/searchad/evaluate.py \
        --predictions-file predictions_kitti_clip.json \
        --split train \
        --searchad-dir H:/GitHub/Kitti_Yolo_Tracking/searchad \
        --scores-output-dir scores/
"""

import json
import numpy as np
import faiss
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CLIP_INDEX_PATH   = "../data/clip_index.faiss"
CLIP_IDS_PATH     = "../data/clip_frame_ids.json"
TRAIN_ANN_PATH    = "searchad_annotations_train.json"
PREDICTIONS_PATH  = "predictions_kitti_clip.json"

# ------------------------------------------------------------------
# Label → descriptive text query
# Splitting on hyphens works well for most labels.
# Override specific ones that need better phrasing.
# ------------------------------------------------------------------
LABEL_QUERY_MAP = {
    "Object-Beacon":                    "road beacon flashing amber light",
    "Object-Trash-Bin":                 "trash bin garbage bin on street",
    "Vehicle-Special-Train":            "tram or train on road",
    "Trailer-Car-Trailer":              "car with trailer attached",
    "Scene-Open-Trunk":                 "car with open trunk boot",
    "Scene-Open-Door":                  "car with open door",
    "Marking-Bicycle-Symbol":           "bicycle lane marking on road",
    "Vehicle-Special-Recreational":     "recreational vehicle camper van",
    "Object-Traffic-Cone":              "orange traffic cone on road",
    "Vehicle-Construction-Truck-Crane": "construction truck with crane",
    "Vehicle-Construction-Excavator":   "excavator construction vehicle",
    "Vehicle-Construction-Loader":      "front loader construction vehicle",
    "Human-Construction-Worker":        "construction worker in high visibility vest",
    "Human-Duty-Other":                 "person in uniform duty",
    "Marking-Yellow-Lane-Arrow":        "yellow arrow road marking lane",
    "Object-Hydrant":                   "fire hydrant on pavement",
    "Sign-Train-Sign":                  "train railway sign",
    "Trailer-Bicycle-Trailer":          "bicycle with cargo trailer",
    "Rideable-Stroller":                "baby stroller pram pushchair",
    "Human-On-Loading-Area":            "person on loading dock platform",
    "Trailer-Caravan-Trailer":          "caravan trailer camper",
    "Animal-Real-Dog":                  "dog on road or pavement",
    "Object-Rollator":                  "rollator walking frame elderly",
    "Vehicle-Special-Bicycle-On-Back":  "bicycle mounted on back of car",
    "Vehicle-Special-Bicycle-On-Roof":  "bicycle mounted on roof rack",
    "Human-With-Sticks-or-Crutches":   "person with crutches or walking sticks",
    "Marking-Bus-Text":                 "bus lane text road marking",
    "Object-Euro-Pallet":               "wooden euro pallet on road",
    "Sign-Road-Bumper-Sign":            "speed bump road sign",
    "Rideable-Three-Wheeler":           "three wheel vehicle or trike",
    "Vehicle-Duty-Fire":                "fire truck fire engine",
    "Vehicle-Duty-Garbage":             "garbage truck refuse collection",
    "Marking-Temporarily-Invalidated":  "crossed out invalidated road marking",
    "Scene-Active-Amber-Lights":        "vehicle with flashing amber warning lights",
    "Animal-Real-Cat":                  "cat on road or pavement",
    "Vehicle-Duty-Police":              "police car vehicle",
    "Object-Platform-Truck":            "platform flatbed truck",
    "Scene-Tunnel":                     "road tunnel entrance",
    "Object-Wheelbarrow":               "wheelbarrow on road or construction site",
    "Vehicle-Construction-Steamroller": "road steamroller compactor",
    "Scene-Snow":                       "snow covered road winter driving",
    "Rideable-Wheelchair":              "wheelchair on pavement",
    "Object-Shopping-Trolley":          "shopping cart trolley on street",
    "Rideable-Toy-Car":                 "child riding toy car or ride-on toy",
    "Object-Hand-Dolly":                "hand truck dolly cart",
    "Object-Suitcase-Trolley":          "suitcase on wheels trolley",
    "Vehicle-Duty-Medical":             "ambulance medical vehicle",
    "Scene-Active-Emergency-Lights":    "vehicle with blue red flashing emergency lights",
    "Object-Movable-Other":             "movable object on road",
    "Vehicle-Construction-Other":       "construction vehicle machinery",
}


def label_to_text(label):
    """Use override map if available, else split label by hyphen."""
    if label in LABEL_QUERY_MAP:
        return LABEL_QUERY_MAP[label]
    parts = label.split("-")
    return " ".join(parts).lower()


def frame_id_to_searchad_path(frame_id):
    """e.g. '003420' → 'kitti/training/image_2/003420.png'"""
    return f"kitti/training/image_2/{frame_id}.png"


def main():
    #  Load CLIP index 
    print("Loading CLIP index...")
    # clip index file generated by doc script
    clip_index = faiss.read_index(CLIP_INDEX_PATH)
    # clip Ids file generated by doc script
    with open(CLIP_IDS_PATH) as f:
        clip_frame_ids = json.load(f)
    total_frames = len(clip_frame_ids)
    print(f"  {total_frames} frames, dim={clip_index.d}")

    #  Load CLIP model 
    print("Loading CLIP model...")
    clip_model = SentenceTransformer("clip-ViT-B-32")

    #  Load train annotations — KITTI only (downloaded from HF dataset - searchAD)
    print("Loading train annotations...")
    with open(TRAIN_ANN_PATH) as f:
        ann = json.load(f)
    kitti_ann = {k: v for k, v in ann.items() if k.startswith("kitti/")}
    print(f"  {len(kitti_ann)} KITTI frames in train annotations")

    # Collect unique labels + positive frame sets
    label_to_pos = defaultdict(set)
    for path, items in kitti_ann.items():
        for item in items:
            label_to_pos[item["label"]].add(path)
    print(f"  {len(label_to_pos)} unique labels")

    # Generate predictions
    predictions = {}

    print("\nGenerating ranked predictions per label...")
    for label in tqdm(sorted(label_to_pos.keys())):
        text_query = label_to_text(label)

        # Encode text with CLIP
        q_emb = clip_model.encode(text_query, convert_to_numpy=True).astype("float32")
        q_emb /= np.linalg.norm(q_emb)

        # Search all frames (full rank)
        scores, indices = clip_index.search(q_emb.reshape(1, -1), total_frames)

        # Convert to SearchAD paths
        ranked = [frame_id_to_searchad_path(clip_frame_ids[i]) for i in indices[0] if i >= 0]
        predictions[label] = ranked

    #  Save predictions 
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f)
    print(f"\nSaved {len(predictions)} label predictions → {PREDICTIONS_PATH}")

    #  Quick sanity check 
    print("\nQuick P@5 sanity check:")
    for label, pos_frames in sorted(label_to_pos.items(),
                                     key=lambda x: -len(x[1]))[:10]:
        ranked = predictions[label]
        hits = sum(1 for p in ranked[:5] if p in pos_frames)
        print(f"  {label:<45} positives={len(pos_frames):>4}  P@5={hits}/5")


if __name__ == "__main__":
    main()
