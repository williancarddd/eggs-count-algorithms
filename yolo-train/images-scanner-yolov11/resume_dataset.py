import os
import glob

# Caminho do dataset
DATASET_PATH = "/media/williancaddd/CODES/WORKSPACE-FIOTEC/eggs-count-algorithms/yolo-train/eggs-scanner-image.v2i.yolov11"
SUBSETS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

# Dicionários
image_counts = {}
object_counts = {}
images_with_eggs = {}  # Novo dicionário para contar imagens com ovos

# Coletar dados
for subset in SUBSETS:
    images_folder = os.path.join(DATASET_PATH, subset, "images")
    labels_folder = os.path.join(DATASET_PATH, subset, "labels")

    # Contar imagens
    image_count = sum(len(glob.glob(os.path.join(images_folder, f"*.{ext}"))) for ext in IMAGE_EXTENSIONS)
    image_counts[subset] = image_count

    # Contar objetos e imagens com ovos
    total_objects = 0
    images_with_eggs[subset] = 0  # Inicializa contador para este subset

    for txt_file in glob.glob(os.path.join(labels_folder, "*.txt")):
        with open(txt_file, "r") as f:
            lines = f.readlines()
            total_objects += len(lines)
            if len(lines) > 0:  # Se houver pelo menos uma linha (um ovo)
                images_with_eggs[subset] += 1

    object_counts[subset] = total_objects

# Totais
total_images = sum(image_counts.values())
total_objects = sum(object_counts.values())
total_images_with_eggs = sum(images_with_eggs.values())

# Tabela
print("\n📊 Resumo Geral do Dataset YOLO:")
print("=" * 90)
print(f"{'CONJUNTO':<10} | {'IMAGENS':^10} | {'IMGS C/OVO':^10} | {'% IMGS':^8} | {'OBJETOS':^10} | {'% OBJ':^8}")
print("-" * 90)

for subset in SUBSETS:
    img = image_counts[subset]
    obj = object_counts[subset]
    img_with_eggs = images_with_eggs[subset]
    img_pct = (img / total_images) * 100 if total_images > 0 else 0
    obj_pct = (obj / total_objects) * 100 if total_objects > 0 else 0

    print(f"{subset.upper():<10} | {img:^10} | {img_with_eggs:^10} | {img_pct:>7.2f}% | {obj:^10} | {obj_pct:>7.2f}%")

print("=" * 90)
print(f"{'TOTAL':<10} | {total_images:^10} | {total_images_with_eggs:^10} | {100:>7.2f}% | {total_objects:^10} | {100:>7.2f}%")
print("=" * 90)
