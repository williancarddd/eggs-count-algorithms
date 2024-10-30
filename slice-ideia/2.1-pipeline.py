import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_exposure(image, gamma=1.0):
    invGamma = 21.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def increase_brightness_contrast(image, alpha=1.0, beta=0):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

def enhance_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image

def apply_fourier_transform(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 180 # radius of the circle in the frequency domain
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    return img_back

def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)
    clahe_applied = cv2.cvtColor(clahe_applied, cv2.COLOR_GRAY2BGR)
    return clahe_applied

def remove_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return denoised

def remove_noise_rbg(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)


def remove_granular_noise(image):
    denoised = cv2.bilateralFilter(image, 15, 75, 75) 
    return denoised

def highlight_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    highlighted = cv2.addWeighted(image, 1.2, edges_colored, 0.2, 0)
    return highlighted


def process_square(square):
    color_image = cv2.cvtColor(square, cv2.COLOR_RGBA2BGR)
    bright_contrast = increase_brightness_contrast(color_image, alpha=-12, beta=50)
    adjusted = adjust_exposure(bright_contrast, 30)
    denoised = remove_noise_rbg(adjusted)
    denoised_granular = remove_granular_noise(denoised)
    enhanced = enhance_color(denoised_granular)
    highlighted = highlight_edges(enhanced)
    return highlighted


def detect_red_objects(square):
    """Detecta objetos na faixa de cor vermelha em um quadrado da imagem no espaço RGB."""
    # Definir intervalos para a cor vermelha no espaço RGB
    lower_red1 = np.array([100, 0, 0])  # Vermelho mais escuro
    upper_red1 = np.array([255, 80, 80])  # Vermelho mais claro
    
    # Criar uma máscara para o intervalo de vermelho
    mask = cv2.inRange(square, lower_red1, upper_red1)
    
    # Aplicar a máscara na imagem original
    red_objects = cv2.bitwise_and(square, square, mask=mask)
    
    return red_objects, mask

def count_objects_in_red_range(square, plot=False):
    """Conta o número de objetos vermelhos em um quadrado da imagem e plota o processo."""
    red_objects, mask = detect_red_objects(square)
    
    # Encontrar contornos (objetos) na máscara binarizada
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_img = square.copy()  # Inicializa a variável contour_img

    if plot:
        # Desenhar os contornos na imagem
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Plotagem
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(square)
        ax[0].set_title("Imagem Original")
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Máscara Vermelha")
        ax[2].imshow(contour_img)
        ax[2].set_title("Contornos Detectados")
        plt.show()
    
    return len(contours), contour_img

def find_square_with_most_red(image, square_size=512):
    """Encontra o quadrado com a maior quantidade de cor vermelha."""
    img_height, img_width, _ = image.shape
    most_red_square = None
    max_red_count = -1
    best_coordinates = (0, 0)
    
    for y in range(0, img_height, square_size):
        for x in range(0, img_width, square_size):
            square = image[y:y + square_size, x:x + square_size]
            red_objects, _ = detect_red_objects(square)
            red_count = np.sum(cv2.cvtColor(red_objects, cv2.COLOR_RGB2GRAY) > 0)
            
            if red_count > max_red_count:
                max_red_count = red_count
                most_red_square = square
                best_coordinates = (x, y)
    
    return most_red_square, best_coordinates


def divide_and_process_image(image_path, square_size=512):
    # Carregar a imagem
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape
    total_objects = 0

    # Lista para armazenar os quadrados processados
    processed_squares = []
    no_processed_squares = []

    # Dividir a imagem em quadrados de 512x512 pixels
    for y in range(0, img_height, square_size):
        for x in range(0, img_width, square_size):
            # Extrair o quadrado da imagem
            square = image[y:y + square_size, x:x + square_size]
            # Processar o quadrado
            processed_square = process_square(square)
            
            no_processed_squares.append(square)
            # Contar objetos vermelhos no quadrado
            num_objects, contours_img = count_objects_in_red_range(processed_square)
            processed_squares.append(contours_img)
            total_objects += num_objects

    return processed_squares, no_processed_squares, img_height, img_width, total_objects


def reconstruct_image(squares, img_height, img_width, square_size=512):
    reconstructed_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    count = 0
    for y in range(0, img_height, square_size):
        for x in range(0, img_width, square_size):
            reconstructed_image[y:y + square_size, x:x + square_size] = squares[count]
            count += 1

    return reconstructed_image


image_path = '/media/williancarddd/NVME/fiotec/AETrampa/Fotos/28_ovos.jpg'

processed_squares, no_processed_squares, img_height, img_width, total_objects = divide_and_process_image(image_path)

# Reconstruir a imagem
reconstructed_image = reconstruct_image(processed_squares, img_height, img_width)

# Salvar a imagem reconstruída
cv2.imwrite('reconstructed_image.jpg', reconstructed_image)

print(f"Total de objetos vermelhos na imagem: {total_objects}")

