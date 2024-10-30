import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import time

app = Flask(__name__)

# Preprocessamento e ajustes de exposição/contraste
def adjust_exposure(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def increase_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def remove_granular_noise(image):
    return cv2.bilateralFilter(image, 15, 75, 75)

def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

def smooth_background(image, kernel_size=7):
    return cv2.medianBlur(image, kernel_size)

def subtract_background(image, smoothed_image):
    return cv2.addWeighted(image, 1.5, smoothed_image, -0.5, 0)

# Função principal de processamento de um quadrado
def process_square(square):
    smoothed_square = smooth_background(square, kernel_size=7)
    subtracted_square = subtract_background(smoothed_square, smoothed_square)
    bright_contrast = increase_brightness_contrast(subtracted_square, alpha=1.5, beta=40)
    adjusted = adjust_exposure(bright_contrast)
    denoised = remove_noise(adjusted)
    denoised_granular = remove_granular_noise(denoised)
    return denoised_granular

# Função de detecção de objetos vermelhos
def detect_red_objects_hsv(square):
    hsv_image = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def count_objects_in_square(square):
    mask = detect_red_objects_hsv(square)
    cleaned_mask = clean_mask(mask)
    
    # Encontrar contornos e desenhá-los na imagem
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50   
    max_area = 2000  
    filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    # Desenhar contornos na imagem original
    contour_img = square.copy()
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)  # Desenhar em verde
    
    return len(filtered_contours), contour_img  # Retornar o número de ovos e a imagem com contornos

def divide_and_process_image(image, square_size=512):
    img_height, img_width, _ = image.shape
    total_objects = 0
    processed_squares = []
    reconstructed_squares = []

    for y in range(0, img_height, square_size):
        for x in range(0, img_width, square_size):
            square = image[y:y + square_size, x:x + square_size]
            processed_square = process_square(square)
            num_objects, contour_square = count_objects_in_square(processed_square)
            total_objects += num_objects
            
            # Adicionando o quadrado processado com contornos e suas informações
            processed_squares.append({
                "coordinates": {"x": x, "y": y},
                "objects_detected": num_objects,
                "square_base64": convert_image_to_base64(contour_square)  # Retornando a imagem processada com contornos
            })
            
            # Guardando a imagem processada para reconstrução final
            reconstructed_squares.append(contour_square)

    return {
        "total_objects": total_objects,
        "processed_squares": processed_squares,
        "image_dimensions": {"height": img_height, "width": img_width},
        "reconstructed_squares": reconstructed_squares  # Retorna todos os quadrados processados para reconstruir a imagem completa
    }

def reconstruct_image(squares, img_height, img_width, square_size=512):
    reconstructed_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    count = 0
    for y in range(0, img_height, square_size):
        for x in range(0, img_width, square_size):
            reconstructed_image[y:y + square_size, x:x + square_size] = squares[count]
            count += 1
    return reconstructed_image

def convert_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/process", methods=["POST"])
def process_image():
    try:
        start_time = time.time()

        # Lendo a imagem enviada pela requisição POST
        file = request.files['file']
        contents = file.read()

        # Carregar a imagem
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Formato de imagem inválido"}), 422

        # Processa a imagem e obtém as informações
        result = divide_and_process_image(image)
        final_time = time.time()

        # Reconstruir a imagem processada
        img_height = result["image_dimensions"]["height"]
        img_width = result["image_dimensions"]["width"]
        reconstructed_image = reconstruct_image(result["reconstructed_squares"], img_height, img_width)

        # Converter a imagem completa processada para base64
        final_image_base64 = convert_image_to_base64(reconstructed_image)

        return jsonify({
            "total_eggs": result["total_objects"],
            "squares": result["processed_squares"],  # Vetor com informações de cada square
            "final_image": final_image_base64,  # A imagem completa processada em base64
            "initial_time": start_time,
            "final_time": final_time,
            "image_dimensions": result["image_dimensions"]
        })
    except Exception as e:
        return jsonify({"error": f"Falha no processamento da imagem: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)