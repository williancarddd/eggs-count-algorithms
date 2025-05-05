from ultralytics import YOLO

# Load a model
model = YOLO("/media/williancaddd/CODES/WORKSPACE-FIOTEC/eggs-count-algorithms/yolo-train/eggs-scanner-image.v2i.yolov11/runs/detect/train2/weights/best.pt")  # load an official model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
