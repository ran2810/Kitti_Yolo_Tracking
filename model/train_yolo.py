import argparse
from ultralytics import YOLO


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def train(
    model_path="model/yolov8n.pt",
    data="data/kitti.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="0",
    resume=False,
):
    """
    Train YOLOv8 on KITTI dataset.

    Args:
        model_path: pretrained weights to start from (yolov8n/s/m/l/x.pt)
        data:       path to dataset YAML
        epochs:     number of training epochs
        imgsz:      input image size
        batch:      batch size (-1 = auto)
        device:     "0" for GPU 0, "cpu" for CPU, "0,1" for multi-GPU
        resume:     resume from last checkpoint if True
    """
    model = YOLO(model_path)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        resume=resume,
    )

    best = results.save_dir / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")
    print(f"Results dir  : {results.save_dir}")
    return results


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on KITTI dataset")
    parser.add_argument("--model",  default="model/yolov8n.pt",
                        help="Pretrained weights (default: yolov8n.pt)")
    parser.add_argument("--data",   default="data/kitti.yaml",
                        help="Dataset YAML (default: data/kitti.yaml)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--batch",  type=int, default=16,
                        help="Batch size, -1 = auto (default: 16)")
    parser.add_argument("--device", default="0",
                        help="Device: 0=GPU, cpu, 0,1=multi-GPU (default: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    train(
        model_path=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
    )
