import os
import cv2
import csv
import time
import re
from image_processor import ImageProcessor

# Nome do algoritmo (pode ser personalizado para cada execução)
ALGORITHM_NAME = "Egg Counting Algorithm v1.0"

# Caminho para as pastas de testes
TESTS_FOLDER = 'tests'
RESULTS_FOLDER = 'results'

# Regex para extrair informações do nome da imagem
IMAGE_PATTERN = re.compile(r"(\d+)_(\d+)(?:_(\d))?\.(jpg|png)$")
RESULTS_PATTERN = re.compile(r"test_results_(\d+)\.csv$")

def create_results_folder():
    """Cria a pasta de resultados se ela não existir."""
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

def get_next_results_file():
    """Obtém o próximo nome de arquivo sequencial para salvar os resultados."""
    existing_files = [f for f in os.listdir(RESULTS_FOLDER) if RESULTS_PATTERN.match(f)]
    if not existing_files:
        return os.path.join(RESULTS_FOLDER, "test_results_1.csv")

    # Encontrar o maior número sequencial existente
    max_index = max(int(RESULTS_PATTERN.match(f).group(1)) for f in existing_files)
    next_index = max_index + 1
    return os.path.join(RESULTS_FOLDER, f"test_results_{next_index}.csv")

def parse_image_filename(filename):
    """Extrai informações do nome do arquivo de imagem."""
    match = IMAGE_PATTERN.match(filename)
    if match:
        sequence = int(match.group(1))
        real_eggs = int(match.group(2))
        flash = int(match.group(3)) if match.group(3) is not None else None
        return sequence, real_eggs, flash
    return None, None, None

def process_image(image_path, real_eggs, flash):
    """Processa uma imagem e retorna as informações de contagem."""
    # Inicializar o processador de imagem
    image_processor = ImageProcessor(image_path, show_stages=False)

    # Medir o tempo de processamento
    start_time = time.time()
    result, img_height, img_width = image_processor.divide_and_process_image()
    end_time = time.time()

    # Tempo total de processamento
    processing_time = end_time - start_time

    # Informações de contagem
    counted_eggs = result['count_total']
    shape = image_processor.get_original_image_shape()

    return counted_eggs, processing_time, shape[0], shape[1]

def run_tests():
    """Executa os testes em todas as imagens nas pastas de teste."""
    create_results_folder()

    # Obter o próximo nome de arquivo sequencial para salvar os resultados
    results_file = get_next_results_file()

    # Criar o arquivo CSV com cabeçalhos
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "id", "test_name", "image_name", "real_eggs", "counted_eggs", 
            "algorithm_name", "time_processment", "image_height", "image_width", "flash"
        ])

        # ID do teste
        test_id = 1

        # Percorrer cada pasta de teste
        for test_folder in sorted(os.listdir(TESTS_FOLDER)):
            folder_path = os.path.join(TESTS_FOLDER, test_folder)

            # Verificar se é uma pasta
            if not os.path.isdir(folder_path):
                continue

            # Nome da pasta de teste (e.g., "test1", "test2")
            test_name = test_folder

            # Processar cada imagem na pasta de teste
            for image_name in sorted(os.listdir(folder_path)):
                # Obter caminho completo da imagem
                image_path = os.path.join(folder_path, image_name)

                # Extrair informações da imagem pelo nome do arquivo
                sequence, real_eggs, flash = parse_image_filename(image_name)
                if sequence is None or real_eggs is None:
                    print(f"Ignorando arquivo {image_name}: formato de nome inválido.")
                    continue

                # Processar a imagem e obter os resultados
                try:
                    counted_eggs, processing_time, img_height, img_width = process_image(
                        image_path, real_eggs, flash
                    )

                    # Escrever os resultados no CSV
                    writer.writerow([
                        test_id, test_name, image_name, real_eggs, counted_eggs, 
                        ALGORITHM_NAME, processing_time, img_height, img_width, flash
                    ])
                    print(f"Teste {test_id} - {image_name}: Real {real_eggs}, Detectado {counted_eggs}")

                    # Incrementar o ID do teste
                    test_id += 1

                except Exception as e:
                    print(f"Erro ao processar {image_name}: {e}")

if __name__ == "__main__":
    run_tests()
