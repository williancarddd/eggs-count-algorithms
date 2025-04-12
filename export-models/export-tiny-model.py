from ultralytics import YOLO

# Caminho do seu modelo treinado (.pt)
model_path = "/media/williancaddd/CODES/WORKSPACE-FIOTEC/eggs-count-algorithms/yolo-train/eggs-scanner-image.v2i.yolov11/runs/detect/train2/weights/best.pt"

# Carrega o modelo
model = YOLO(model_path)

# Exporta para ONNX (mais compatível com vários backends)
model.export(format='onnx', dynamic=True)

# Exporta para TensorRT (requer suporte local instalado)
# model.export(format='engine')

# Exporta para OpenVINO
# model.export(format='openvino')

# Exporta para TorchScript
# model.export(format='torchscript')
