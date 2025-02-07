import os
import torch
from ultralytics import YOLO


# Clear CUDA cache
torch.cuda.empty_cache()

# Load and train model
model = YOLO("yolo11n.yaml")
results = model.train(
    data='/media/williancaddd/CODES/fiotec/eggs-count-algorithms/eggs-scanner-image.v2i.yolov11/data.yaml',
    epochs=200,
    imgsz=512,          # Reduced image size
    batch=-1,        
    save=True,
    resume=True
)
