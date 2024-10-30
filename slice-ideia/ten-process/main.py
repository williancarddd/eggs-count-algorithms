
import cv2
import matplotlib.pyplot as plt
from image_processor import ImageProcessor

# Definir caminho para a imagem
image_path = '/media/williancaddd/CODES/fiotec/AETrampa/Fotos/10__1ovos.jpg'

# Processar imagem e contar objetos
image_processor = ImageProcessor(image_path, show_stages=False)
result, img_height, img_width = image_processor.divide_and_process_image()
print(f"Resultado do processamento: {result}")

