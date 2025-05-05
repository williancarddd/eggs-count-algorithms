import pandas as pd

# Load the uploaded CSV file
file_path = '/mnt/data/modelo_1_com_tipo.csv'
df = pd.read_csv(file_path)

# Apply the requested transformations
df = df[df['dataset_name'] != 'base-5']
df = df.rename(columns={
    'image_name': 'Palheta',
    'dataset_name': 'dataset_name',  # keeping for reference, will drop later
    'real_eggs_images': 'Ovos Esperados (E)',
    'counted_eggs_image': 'Ovos observados (O)'
})
df['Diferença (O-E)'] = df['Ovos observados (O)'] - df['Ovos Esperados (E)']
df = df.drop(columns=['dataset_name'])

import ace_tools as tools; tools.display_dataframe_to_user(name="Palheta Ovos Processado", dataframe=df)
