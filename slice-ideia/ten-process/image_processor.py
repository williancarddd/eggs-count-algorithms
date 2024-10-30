
import cv2
import numpy as np
from square_processor import SquareProcessor

SQUARE_SIZE = 512


class ImageProcessor:
    def __init__(self, image_path, square_size=SQUARE_SIZE, show_stages=False):
        self.image_path = image_path
        self.square_size = square_size
        self.show_stages = show_stages
        self.shape = []
        self.contoured_squares = []

    def get_original_image_shape(self):
        return self.shape
    
    def resize_to_fit_grid(self, image):
        """Redimensiona a imagem para que suas dimensões sejam múltiplos exatos do tamanho dos quadrados."""
        img_height, img_width, _ = image.shape
        self.shape = [img_height, img_width]
        new_height = (img_height // self.square_size) * self.square_size
        new_width = (img_width // self.square_size) * self.square_size
        return cv2.resize(image, (new_width, new_height))

    def divide_and_process_image(self):
        """Divide a imagem redimensionada em quadrados e processa cada um."""
        image = cv2.imread(self.image_path)
        resized_image = self.resize_to_fit_grid(image)
        img_height, img_width, _ = resized_image.shape

        # Dividir a imagem em quadrados de tamanho `square_size`
        total_objects = 0
        squares_info = []

        for y in range(0, img_height, self.square_size):
            for x in range(0, img_width, self.square_size):
                # Extrair o quadrado da imagem redimensionada
                square = resized_image[y:y + self.square_size, x:x + self.square_size]

                # Processar o quadrado
                square_processor = SquareProcessor(square, show_stages=self.show_stages)
                objects_in_square = square_processor.process()

                # Adicionar informações do quadrado
                index = (y // self.square_size) * (img_width // self.square_size) + (x // self.square_size)
                squares_info.append({'index': index, 'count': objects_in_square})

                # Acumular total de objetos
                total_objects += objects_in_square

                # Armazenar quadrado com contornos desenhados
                self.contoured_squares.append(square_processor.stages['final_contours'])

                # Exibir as etapas do processamento para cada quadrado
                square_processor.plot_stages()

        return {
            'count_total': total_objects,
            'squares': squares_info
        }, img_height, img_width

    def reconstruct_image(self, img_height, img_width):
        """Reconstrói a imagem a partir dos quadrados processados."""
        reconstructed_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        count = 0
        for y in range(0, img_height, self.square_size):
            for x in range(0, img_width, self.square_size):
                reconstructed_image[y:y + self.square_size, x:x + self.square_size] = self.contoured_squares[count]
                count += 1
        return reconstructed_image
