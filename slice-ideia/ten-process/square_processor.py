# square_processor.py

import cv2
import numpy as np
import matplotlib.pyplot as plt


class SquareProcessor:
    def __init__(self, square, show_stages=False):
        self.square = square
        self.show_stages = show_stages
        self.stages = {'original': square.copy()}  # Dicionário para armazenar as etapas de processamento

    def adjust_exposure(self):
        """Ajuste de exposição usando CLAHE no canal de luminância da imagem."""
        hsv = cv2.cvtColor(self.square, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        self.square = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.stages['adjust_exposure'] = self.square.copy()

    def increase_brightness_contrast(self, alpha=1.5, beta=40):
        """Ajusta o brilho e contraste da imagem."""
        self.square = cv2.convertScaleAbs(self.square, alpha=alpha, beta=beta)
        self.stages['brightness_contrast'] = self.square.copy()

    def remove_noise(self):
        """Remove ruído da imagem usando a técnica Non-Local Means."""
        self.square = cv2.fastNlMeansDenoisingColored(self.square, None, 10, 10, 7, 21)
        self.stages['remove_noise'] = self.square.copy()

    def laplacian_filter(self):
        """Aplica filtro Laplaciano para realçar bordas."""
        self.square = cv2.Laplacian(self.square, cv2.CV_8U, ksize=3)
        self.stages['laplacian_filter'] = self.square.copy()

    def remove_granular_noise(self):
        """Remove ruído granular usando filtro bilateral."""
        self.square = cv2.bilateralFilter(self.square, 15, 75, 75)
        self.stages['remove_granular_noise'] = self.square.copy()

    def smooth_background(self, kernel_size=7):
        """Suaviza a imagem para remover texturas usando blur mediano."""
        self.square = cv2.medianBlur(self.square, kernel_size)
        self.stages['smooth_background'] = self.square.copy()

    def subtract_background(self, original_image):
        """Subtrai a imagem suavizada da original para destacar os objetos."""
        self.square = cv2.addWeighted(original_image, 1.5, self.square, -0.5, 0)
        self.stages['subtract_background'] = self.square.copy()

    def detect_red_objects_hsv(self):
        """Detecta objetos vermelhos na imagem utilizando o espaço HSV."""
        hsv_image = cv2.cvtColor(self.square, cv2.COLOR_BGR2HSV)

        # Intervalos de cor vermelha ajustados
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        self.mask = cv2.bitwise_or(mask1, mask2)
        self.stages['red_mask'] = self.mask.copy()

    def clean_mask(self):
        """Aplica morfologia para limpar a máscara."""
        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.stages['cleaned_mask'] = self.mask.copy()

    def count_objects(self):
        """Conta os objetos vermelhos no quadrado usando a máscara limpa."""
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por área para evitar ruído
        min_area = 50  # Tamanho mínimo para ser considerado um ovo
        max_area = 2000  # Tamanho máximo para eliminar contornos irrelevantes
        filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

        # Desenhar os contornos detectados na imagem original
        contour_img = self.square.copy()
        cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)
        self.stages['final_contours'] = contour_img

        return len(filtered_contours)

    def process(self):
        """Executa todo o pipeline de processamento de imagem para o quadrado."""
        original_image = self.square.copy()

        # Pipeline de processamento
        self.laplacian_filter()
        self.smooth_background()
        self.subtract_background(original_image)
        self.increase_brightness_contrast()
        self.adjust_exposure()
        self.remove_noise()
        self.remove_granular_noise()
        self.detect_red_objects_hsv()
        self.clean_mask()

        # Contar objetos detectados
        return self.count_objects()

    def plot_stages(self):
        """Exibe todas as etapas do processamento de imagem."""
        if not self.show_stages:
            return
        stages = self.stages
        n_stages = len(stages)

        fig, axes = plt.subplots(1, n_stages, figsize=(20, 5))
        for idx, (stage_name, image) in enumerate(stages.items()):
            if len(image.shape) == 3:  # Se for uma imagem colorida
                axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:  # Se for uma imagem de máscara (grayscale)
                axes[idx].imshow(image, cmap='gray')
            axes[idx].set_title(stage_name)
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
