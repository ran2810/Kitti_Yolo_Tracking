import argparse
from ultralytics import YOLO


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def train(
    model_path="model/yolov8n.pt",
    data="data/kitti.yaml", # path to dataset # path to dataset yaml
    epochs=50,              # number of training epochs
    imgsz=640,              # input image size
    batch=16,               # batch size
    device="0",             # device as GPU or cpu
    resume=False,           # resume from last checkpoint if true
):
    """
    Train YOLOv8 on KITTI dataset.
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
