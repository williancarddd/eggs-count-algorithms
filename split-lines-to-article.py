import csv
import os
from datetime import datetime

def format_input_date(date_str: str) -> str:
    """Converte data do formato dd-mm/aaaa para a data padrão yyyy-mm-dd."""
    return datetime.strptime(date_str, "%d-%m/%Y").strftime("%Y-%m-%d")

def extract_dataset_name(full_path: str) -> str:
    """Extrai o último diretório de um caminho completo."""
    return os.path.basename(full_path.strip())

def filter_csv_by_date(input_csv_path: str, user_date: str):
    output_folder = "article"
    os.makedirs(output_folder, exist_ok=True)

    formatted_date = format_input_date(user_date)
    output_path = os.path.join(output_folder, "filtered_output.csv")

    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_path, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Cabeçalho de saída
        writer.writerow([
            "image_name", "dataset_name", "real_eggs_images",
            "counted_eggs_image", "model_name"
        ])

        for row in reader:
            timestamp = row.get("timestamp", "")
            if timestamp.startswith(formatted_date):
                dataset_name = extract_dataset_name(row.get("name_file", ""))
                writer.writerow([
                    row.get("image_name", ""),
                    dataset_name,
                    row.get("real_eggs_images", ""),
                    row.get("counted_eggs_image", ""),
                    row.get("model_name", "")
                ])

    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    path_csv = "./model_comparison.csv"
    data_usuario = "13-05/2025"
    filter_csv_by_date(path_csv, data_usuario)
