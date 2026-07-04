import argparse
from ultralytics import YOLO


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
def evaluate(model_path="../runs/detect/train/weights/best.pt", data="../data/kitti.yaml"):
    model = YOLO(model_path)
    metrics = model.val(data=data)

    print("\n--- KITTI Evaluation ---")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("\nPer-class AP50:")
    for cls, ap in enumerate(metrics.box.maps):
        name = metrics.names.get(cls, f"Class {cls}")
        print(f"  {name:<12}: {ap:.4f}")

    return metrics


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on KITTI dataset")
    parser.add_argument("--model", default="../runs/detect/train/weights/best.pt",
                        help="Path to model weights (default: runs/detect/train/weights/best.pt)")
    parser.add_argument("--data",  default="../data/kitti.yaml",
                        help="Path to dataset YAML (default: data/kitti.yaml)")
    args = parser.parse_args()

    evaluate(model_path=args.model, data=args.data)