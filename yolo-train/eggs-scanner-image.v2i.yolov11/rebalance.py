import os
import shutil
import random

# Caminhos de origem
base_path = ""
splits_orig = ["train", "valid", "test"]

# Caminhos de destino (recriados limpos)
output_base = os.path.join(base_path, "balanced")
splits_new = ["train", "valid", "test"]

# Criar destino limpo
for split in splits_new:
    img_path = os.path.join(output_base, split, "images")
    label_path = os.path.join(output_base, split, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

# Coletar todas as imagens e contar objetos
image_data = []

for split in splits_orig:
    img_dir = os.path.join(base_path, split, "images")
    label_dir = os.path.join(base_path, split, "labels")

    for file in os.listdir(img_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(file)[0]
            label_path = os.path.join(label_dir, base + ".txt")
            label_count = 0
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    label_count = len(f.readlines())
            image_data.append({
                "filename": file,
                "split": split,
                "obj_count": label_count,
                "img_path": os.path.join(img_dir, file),
                "label_path": label_path
            })

# Embaralhar
random.seed(42)
random.shuffle(image_data)

# Calcular totais
total_objects = sum(item["obj_count"] for item in image_data)
target_train = round(0.7 * total_objects)
target_valid = round(0.2 * total_objects)
target_test = total_objects - target_train - target_valid

# Inicialização
buckets = {"train": [], "valid": [], "test": []}
obj_counters = {"train": 0, "valid": 0, "test": 0}

# Alocação
for item in image_data:
    count = item["obj_count"]
    if obj_counters["train"] + count <= target_train:
        buckets["train"].append(item)
        obj_counters["train"] += count
    elif obj_counters["valid"] + count <= target_valid:
        buckets["valid"].append(item)
        obj_counters["valid"] += count
    else:
        buckets["test"].append(item)
        obj_counters["test"] += count

# Mover arquivos para nova pasta
for split in splits_new:
    for item in buckets[split]:
        base = os.path.splitext(item["filename"])[0]

        # Imagem
        dest_img = os.path.join(output_base, split, "images", item["filename"])
        shutil.copy(item["img_path"], dest_img)

        # Label
        if os.path.exists(item["label_path"]):
            dest_label = os.path.join(output_base, split, "labels", base + ".txt")
            shutil.copy(item["label_path"], dest_label)

# Resumo final
print("✅ Redistribuição concluída!")
print(f"Objetos totais: {total_objects}")
for split in splits_new:
    print(f"{split.upper()}: {obj_counters[split]} objetos em {len(buckets[split])} imagens")
print(f"\nNovo dataset salvo em: {output_base}/")
