import os
import torch
from ultralytics import YOLO

# Set CUDA environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear CUDA cache
torch.cuda.empty_cache()

# Load and train model
model = YOLO("yolo11n.yaml").load("/media/williancaddd/CODES/fiotec/eggs-count-algorithms/Labels Eggs.v9i.yolov11/runs/detect/train3/weights/best.pt")
results = model.train(
    data='/media/williancaddd/CODES/fiotec/eggs-count-algorithms/Labels Eggs.v9i.yolov11/data.yaml',
    epochs=200,
    imgsz=512,          # Reduced image size
    batch=-1,            # Smaller batch size
    lr0=0.001,
    momentum=0.937,
    weight_decay=0.001,
    save=True,
    resume=True,
    workers=1,          # Reduced workers
    optimizer='AdamW',
    iou=0.3,
    conf=0.2,
    nms=True,
    augment=True,
    amp=True,           # Mixed precision training
)
