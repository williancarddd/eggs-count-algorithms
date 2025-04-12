import pandas as pd
import zipfile
import os
from io import StringIO

# Caminho do CSV
csv_path = "model_comparison.csv"

# Nome do modelo alvo
target_model = "best-train2.pt"

# Caminho de saída do zip
zip_output_path = "imagens_diferenca_maior_50_best_train2.zip"

# Carregar o CSV
df = pd.read_csv(csv_path)

# Filtrar pelo modelo
filtered_df = df[df["model_name"] == target_model].copy()

# Substituir zeros para evitar divisão por zero
real_eggs = filtered_df["real_eggs_images"].replace(0, pd.NA)

# Calcular diferença percentual
filtered_df["abs_diff_percentage"] = (
    abs(filtered_df["real_eggs_images"] - filtered_df["counted_eggs_image"]) / real_eggs
) * 100

# Filtrar as imagens com diferença > 50%
filtered_df = filtered_df[filtered_df["abs_diff_percentage"] > 50]

# Criar conteúdo do TXT em memória
txt_buffer = StringIO()
txt_buffer.write("image_name, counted_eggs_image, real_eggs_images, abs_diff_percentage\n")
for _, row in filtered_df.iterrows():
    txt_buffer.write(f"{row['image_name']}, {row['counted_eggs_image']}, {row['real_eggs_images']}, {row['abs_diff_percentage']:.2f}%\n")

# Criar o ZIP com imagens e o TXT interno
with zipfile.ZipFile(zip_output_path, "w") as zipf:
    # Adiciona as imagens
    for _, row in filtered_df.iterrows():
        full_image_path = os.path.join(row["name_file"], row["image_name"])
        if os.path.exists(full_image_path):
            zipf.write(full_image_path, arcname=row["image_name"])
        else:
            print(f"⚠ Imagem não encontrada: {full_image_path}")

    # Adiciona o TXT gerado em memória
    zipf.writestr("resumo_diferencas_best_train2.txt", txt_buffer.getvalue())

print("✔ ZIP gerado com imagens e resumo interno:")
print(f"📦 {zip_output_path}")
