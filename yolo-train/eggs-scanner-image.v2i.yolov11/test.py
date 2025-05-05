import os
import torch
from ultralytics import YOLO


# Clear CUDA cache
torch.cuda.empty_cache()

# Load and train model
model = YOLO("yolo11n.yaml")
results = model.train(
    data='/media/williancaddd/CODES/WORKSPACE-FIOTEC/eggs-count-algorithms/yolo-train/eggs-scanner-image.v2i.yolov11/data.yaml',
    epochs=200,
    imgsz=512,
    batch=-1,
    save=True,
    resume=True
)
