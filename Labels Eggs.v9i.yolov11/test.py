import os
from ultralytics import YOLO

model = YOLO("yolo11n.yaml", ch=1).load(
    "/media/williancaddd/CODES/fiotec/AETrampa/Labels Eggs.v9i.yolov11/runs/detect/train3/weights/best.pt")
results = model.train(
    data='/media/williancaddd/CODES/fiotec/AETrampa/Labels Eggs.v9i.yolov11/data.yaml',
    epochs=200,          # Número de épocas (mantido)
    imgsz=512,           # Resolução das imagens (mantido)
    batch=-1,            # Batch size ajustado para um valor razoável
    lr0=0.001,           # Taxa de aprendizado inicial
    momentum=0.937,      # Valor típico para YOLO
    weight_decay=0.001,  # Regularização para evitar overfitting
    save=True,           # Salvar os checkpoints
    resume=True,         # Retomar o treinamento
    workers=4,           # Número de threads de leitura de dados
    optimizer='AdamW',   # Melhor otimização para problemas complexos
    iou=0.3,            # Limiar de IoU para detecção
    conf=0.2,          # Limiar de confiança para detecção
    nms=True,            # Limiar de NMS para detecção,
    augment=True,        # Aumento de dados,
    
)
