from ultralytics import YOLO
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="car_yolo_gpu")
    p.add_argument("--device", default="0")          # "0" = first GPU, "cpu" = CPU
    p.add_argument("--workers", type=int, default=0) # 0 on Windows avoids dataloader issues
    p.add_argument("--from_best", action="store_true",
                   help="Start from your previous best weights instead of a COCO pretrained model")
    args = p.parse_args()

    # choose starting weights
    start_weights = "yolov8n.pt"
    if args.from_best:
        start_weights = "runs/car_yolo_gpu/weights/best.pt"  # adjust if your best is elsewhere

    model = YOLO(start_weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project="runs",
        device=args.device,
        workers=args.workers,
        # useful stability/perf knobs:
        cache=True,          # cache images to RAM
        amp=True,            # mixed precision
        cos_lr=True,         # cosine LR schedule
        patience=20,         # early stop patience
        seed=42,
        deterministic=True,
        close_mosaic=10,     # last N epochs disable mosaic for stability
    )

    # Validate once at the end
    model.val(data=args.data, device=args.device, workers=args.workers)

if __name__ == "__main__":
    main()
