import cv2
import numpy as np
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

# Constante para o tamanho da janela deslizante
WINDOW_SIZE = 254


class ImageProcessor:
    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        """
        Inicializa o processador de imagem.

        :param window_size: Tamanho da janela para o processamento.
        """
        self.window_size: int = window_size
        self.image: Optional[np.ndarray] = None

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Carrega uma imagem do disco e converte de BGR para RGB.

        :param image_path: Caminho para o arquivo de imagem.
        :return: Imagem carregada.
        :raises ValueError: Se a imagem não for carregada.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(
                f"Não foi possível carregar a imagem: {image_path}")
        return image

    def save_processed_window(self, processed_window: np.ndarray, output_folder: str, index: int) -> None:
        """
        Salva uma janela processada com um nome único gerado via hash MD5.

        :param processed_window: Janela processada a ser salva.
        :param output_folder: Pasta onde a janela será salva.
        :param index: Índice da janela para nomeação.
        """
        os.makedirs(output_folder, exist_ok=True)
        contiguous_array = np.ascontiguousarray(processed_window)
        hs = hashlib.md5(contiguous_array.tobytes()).hexdigest()
        file_name = f"{index}_{hs}.png"
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, processed_window)

    def generate_squares(self, image_path: str, output_folder: str) -> None:
        """
        Gera quadrados processados a partir de uma imagem e salva em uma pasta.

        :param image_path: Caminho da imagem.
        :param output_folder: Pasta onde os quadrados serão salvos.
        """
        self.image = self.load_image(image_path)
        img_height, img_width, _ = self.image.shape

        tasks = []
        index = 0

        with ThreadPoolExecutor() as executor:
            for y in range(0, img_height, self.window_size):
                for x in range(0, img_width, self.window_size):
                    window = self.image[y:y +
                                        self.window_size, x:x + self.window_size]
                    future = executor.submit(
                        self.save_processed_window, window, output_folder, index)
                    tasks.append(future)
                    index += 1

        for task in tasks:
            task.result()


def main() -> None:
    folder_path = '/media/williancaddd/CODES/WORKSPACE-FIOTEC/eggs-count-algorithms/base-5'
    output_folder = os.path.join(folder_path, "generated-squares")
    processor = ImageProcessor()

    image_files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in image_files:
        full_path = os.path.join(folder_path, filename)
        try:
            processor.generate_squares(full_path, output_folder)
            print(f"Quadrados processados e salvos para {filename}.")
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()
