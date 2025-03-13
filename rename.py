import os
import csv
import re

# Definir o diretório base
directory_base = "."  # Altere para o caminho correto se necessário
bases = [f"base-{i}" for i in range(4, 10)]

# Expressão regular para o formato paleta-x-y png or jpg
pattern_paleta = re.compile(r"paleta-(\d+)-(\d+)\.(png|jpg)")

for base in bases:
    base_path = os.path.join(directory_base, base)
    csv_file = os.path.join(base_path, f"{base}_metadata.csv")

    if not os.path.exists(base_path):
        continue

    data = []

    for file_name in os.listdir(base_path):
        file_path = os.path.join(base_path, file_name)

        
        # Ignorar diretórios dentro das bases
        if os.path.isdir(file_path):
            continue

        match_paleta = pattern_paleta.match(file_name)

        if match_paleta:
            paleta_number = match_paleta.group(1)
            quantity_ovos = match_paleta.group(2)
            data.append([file_path, quantity_ovos, paleta_number])
        else:
            continue  # Ignorar arquivos que não seguem o formato reconhecido

    # Escrever CSV com os dados apenas se houver informações
    if data:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["file_path", "quantity_ovos", "paleta_number"])
            writer.writerows(data)
        print(f"Processado {base}, CSV salvo em: {csv_file}")
