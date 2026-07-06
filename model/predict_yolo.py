import argparse
from ultralytics import YOLO


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
def predict(
    model_path="runs/detect/train/weights/best.pt",
    source="data/training/image_2",     # path to images or video
    project="runs/detect",              # output root directory
    name="predict",                     # output sub-directory name
    conf=0.25,                          # confidence threshold
):
    """Run YOLOv8 inference and save YOLO-format prediction labels."""
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        save_txt=True,      # saves YOLO-format labels to <project>/<name>/labels/
        save_conf=True,     # include confidence scores in label files
        conf=conf,
        project=project,
        name=name,
    )

    print(f"\nPrediction complete.")
    print(f"Labels saved to : {project}/{name}/labels/")
    return results


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on KITTI images")
    parser.add_argument("--model",   default="runs/detect/train/weights/best.pt",
                        help="Path to model weights (default: runs/detect/train/weights/best.pt)")
    parser.add_argument("--source",  default="data/training/image_2",
                        help="Image source directory or file (default: data/training/image_2)")
    parser.add_argument("--project", default="runs/detect",
                        help="Output root directory (default: runs/detect)")
    parser.add_argument("--name",    default="predict",
                        help="Output sub-directory name (default: predict)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    predict(
        model_path=args.model,
        source=args.source,
        project=args.project,
        name=args.name,
        conf=args.conf,
    )
