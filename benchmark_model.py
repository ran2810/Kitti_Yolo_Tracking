import torch
import time
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics.utils.metrics import ConfusionMatrix

# ------------------------------------------------------------
# FP32 Evaluation (CPU)
# ------------------------------------------------------------
def evaluate_fp32(model_path, data_yaml):
    model = YOLO(model_path)

    print("\n=== FP32 Evaluation ===")
    metrics = model.val(data=data_yaml, device="cpu")
    map50 = metrics.box.map50
    map95 = metrics.box.map

    # Warm-up 
    for _ in range(5): 
        dummy = torch.randn(1, 3, 640, 640)
        model.predict(dummy, device="cpu")
    
    def bench_fp32():
        start = time.time()
        for _ in range(20):
            dummy = torch.randn(1, 3, 640, 640)
            _ = model.predict(dummy, device="cpu")
        return (time.time() - start) / 20

    latency = bench_fp32()

    return {
        "mAP50": map50, "mAP95": map95, "latency_ms": latency * 1000
    }

# ------------------------------------------------------------
# FP32 Evaluation (GPU)
# ------------------------------------------------------------
def eval_fp32_gpu(model_path, data_yaml):
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    model = YOLO(model_path)

    print("\n=== FP32 Evaluation (GPU) ===")
    metrics = model.val(data=data_yaml, device=0)
    map50 = metrics.box.map50
    map95 = metrics.box.map

    dummy = torch.randn(1, 3, 640, 640).cuda()

    # Warm-up
    for _ in range(5):
        model.predict(dummy, device=0)

    def bench():
        start = time.time()
        for _ in range(20):
            model.predict(dummy, device=0)
        return (time.time() - start) / 20

    latency = bench()

    return {"mAP50": map50, "mAP95": map95, "latency_ms": latency * 1000}


# ------------------------------------------------------------
# FP16 Evaluation (GPU)
# ------------------------------------------------------------
def evaluate_fp16(model_path, data_yaml):
    model = YOLO(model_path)

    print("\n=== FP16 Evaluation (GPU) ===")
    metrics = model.val(data=data_yaml, device=0, half=True)
    map50 = metrics.box.map50
    map95 = metrics.box.map

    dummy = torch.randn(1, 3, 640, 640).cuda().half()

    # Warm-up
    for _ in range(5):
        model.predict(dummy, device=0, half=True)

    def bench_fp16():
        start = time.time()
        for _ in range(20):
            model.predict(dummy, device=0, half=True)
        return (time.time() - start) / 20

    latency = bench_fp16()

    return {
        "mAP50": map50,
        "mAP95": map95,
        "latency_ms": latency * 1000
    }


# ------------------------------------------------------------
# INT8 PTQ Evaluation (CPU)
# ------------------------------------------------------------
def evaluate_int8(model_path, data_yaml):
    model = YOLO(model_path)

    # print("\n=== INT8 Quantization (PTQ) ===")
    onnx_path = model.export(format="onnx", opset=12)
    int8_path = "best_int8.onnx"

    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)

    session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy = torch.randn(1, 3, 640, 640)
    dummy_np = dummy.numpy().astype(np.float32)

    def bench_int8():
        start = time.time()
        for _ in range(20):
            session.run(None, {input_name: dummy_np})
        return (time.time() - start) / 20

    latency = bench_int8()

    print("\n=== INT8 Evaluation (ONNX Runtime) ===")

    int8_model = YOLO("runs/detect/train/weights/best.onnx")

    int8_metrics = int8_model.val(
        data="kitti.yaml",
        imgsz=640,
        batch=1,
        device="cpu"
    )

    return {
        "mAP50": int8_metrics.box.map50,
        "mAP95": int8_metrics.box.map,
        "latency_ms": latency * 1000
    }
    
# def evaluate_int8(model_path, data_yaml):
#     model = YOLO(model_path)

#     print("\n=== INT8 Quantization (PTQ) ===")
#     onnx_path = model.export(format="onnx", opset=12)
#     int8_path = "best_int8.onnx"

#     quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)

#     session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
#     input_name = session.get_inputs()[0].name

#     dummy = torch.randn(1, 3, 640, 640)
#     dummy_np = dummy.numpy().astype(np.float32)

#     def bench_int8():
#         start = time.time()
#         for _ in range(20):
#             session.run(None, {input_name: dummy_np})
#         return (time.time() - start) / 20

#     latency = bench_int8()

#     print("\n=== INT8 Evaluation (ONNX Runtime) ===")
#     cm = ConfusionMatrix()

#     validator = model.val(
#         data=data_yaml,
#         device="cpu",
#         plots=False,
#         save_json=False
#     )
#     val_loader = validator.dataloader

#     for batch in val_loader:
#         imgs = batch[0].numpy().astype(np.float32)
#         preds = session.run(None, {input_name: imgs})[0]
#         preds = torch.tensor(preds)
#         preds = model.postprocess(preds)[0]
#         cm.process_batch(preds, batch[1])

#     map50, map95 = cm.ap50, cm.ap

#     return {
#         "mAP50": map50,
#         "mAP95": map95,
#         "latency_ms": latency * 1000
#     }


# ------------------------------------------------------------
# MASTER FUNCTION
# ------------------------------------------------------------
def trigger_all_benchmarks(model_path="best.pt", data_yaml="kitti.yaml"):

    results = {}

    results["FP32"] = evaluate_fp32(model_path, data_yaml)

    if torch.cuda.is_available():
        results["FP32_GPU"] = eval_fp32_gpu(model_path, data_yaml)
        results["FP16"] = evaluate_fp16(model_path, data_yaml)
    else:
        results["FP16"] = {"error": "CUDA not available"}

    results["INT8"] = evaluate_int8(model_path, data_yaml)

    print("\n==============================")
    print("     FINAL COMPARISON")
    print("==============================")
    for k, v in results.items():
        print(f"\n{k}: {v}")

    return results


# ------------------------------------------------------------
# Run directly from CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    trigger_all_benchmarks("best.pt", "kitti.yaml")