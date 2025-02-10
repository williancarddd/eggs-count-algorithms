import os
import cv2
import albumentations as A
from albumentations.core.composition import OneOf
from tqdm import tqdm

# Diretórios
base_dir = '/media/williancaddd/CODES/fiotec/eggs-count-algorithms/eggs-scanner-image.v2i.yolov11'  # Substitua pelo caminho base do seu projeto
folders = ['train', 'valid', 'test']

# Augmentations usando Albumentations
augmentation = A.Compose([
    A.RandomBrightnessContrast(p=0.6),  # Ajuste de brilho e contraste
    OneOf([
        A.GaussNoise(p=0.4),  # Adição de ruído gaussiano
        A.MotionBlur(p=0.4),  # Efeito de desfoque de movimento
        A.MedianBlur(blur_limit=3, p=0.45),  # Efeito de desfoque mediano
    ], p=0.5),
    A.HueSaturationValue(p=0.4)  # Alteração de matiz, saturação e valor
])

# Função para aplicar augmentations
def augment_and_save(image_path, output_path):
    # Carrega a imagem
    image = cv2.imread(image_path)
    # Converte para RGB se necessário
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplica as transformações
    augmented = augmentation(image=image)
    augmented_image = augmented['image']

    # Salva a imagem processada
    cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

# Processa as pastas
for folder in folders:
    images_dir = os.path.join(base_dir, folder, 'images')
    labels_dir = os.path.join(base_dir, folder, 'labels')  # Supondo que os rótulos estejam aqui
    output_images_dir = os.path.join(images_dir, 'augmented')
    output_labels_dir = os.path.join(labels_dir, 'augmented')

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Itera pelas imagens
    for image_name in tqdm(os.listdir(images_dir), desc=f'Processing {folder} folder'):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

            # Caminho de saída
            output_image_path = os.path.join(output_images_dir, f'aug_{image_name}')
            output_label_path = os.path.join(output_labels_dir, f'aug_{image_name.replace(".jpg", ".txt").replace(".png", ".txt")}')

            # Aplica augmentations e copia o rótulo
            augment_and_save(image_path, output_image_path)
            if os.path.exists(label_path):
                os.system(f'cp {label_path} {output_label_path}')
