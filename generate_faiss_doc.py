import os, json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def parse_kitti_label_file(path):
    """
    Docstring for parse_kitti_label_file
    
    :param path: label files relative path
    """
    objects = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            obj = {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": list(map(float, parts[4:8])),
                "dimensions": list(map(float, parts[8:11])),
                "location": list(map(float, parts[11:14])),
                "rotation_y": float(parts[14])
            }
            objects.append(obj)
    return objects


def build_doc(frame_id, label_dir, image_dir):
    """
    generate doc for each image from label file
    
    :param frame_id: image frame id
    :param label_dir: label files relative path
    :param image_dir: image files relative path
    """
    label_path = os.path.join(label_dir, f"{frame_id}.txt")
    image_path = os.path.join(image_dir, f"{frame_id}.png")

    # parse label file
    objects = parse_kitti_label_file(label_path)

    num_cars = sum(1 for o in objects if o["type"] == "Car")
    num_peds = sum(1 for o in objects if o["type"] == "Pedestrian")
    num_cyc = sum(1 for o in objects if o["type"] == "Cyclist")
    max_occ = max((o["occluded"] for o in objects), default=0)
    max_trunc = max((o["truncated"] for o in objects), default=0.0)

    # create summary
    summary = (
        f"Frame {frame_id} contains {num_cars} cars, {num_peds} pedestrians, "
        f"and {num_cyc} cyclists. Max occlusion level is {max_occ}, "
        f"max truncation is {max_trunc}."
    )

    # return label file data as doc
    return {
        "id": frame_id,
        "image_path": image_path,
        "label_path": label_path,
        "objects": objects,
        "summary_text": summary,
        "num_cars": num_cars,
        "num_pedestrians": num_peds,
        "num_cyclists": num_cyc,
        "max_occlusion": max_occ,
        "max_truncation": max_trunc
    }

def build_kitti_index(label_dir, image_dir):
    """
    index into faiss db and embedded into sentence transformer model 
    
    :param label_dir: label files relative path
    :param image_dir: image files relative path
    """
    # generate label files data as doc
    frame_ids = [f.split(".")[0] for f in os.listdir(label_dir)]
    docs = [build_doc(fid, label_dir, image_dir) for fid in frame_ids]

    # encode summary of each label file into sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["summary_text"] for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # index generated the embeddings into faiss database
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return docs, index, model


# ------------------------------------------------------------
# Run directly from CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    docs, index, model = build_kitti_index("data/training/label_2", "data/training/image_2")

    # write data into files
    with open("data/kitti_docs.json", "w") as f: 
        json.dump(docs, f, indent=2)
    faiss.write_index(index, "data/kitti_index.faiss")
    with open("data/embedding_model.txt", "w") as f: 
        f.write("all-MiniLM-L6-v2")
