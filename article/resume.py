import pandas as pd

# Load CSV file
file_path = '/media/williancaddd/CODES/WORKSPACE-FIOTEC/eggs-count-algorithms/article/filtered_output.csv'
df = pd.read_csv(file_path)

# Filter out 'base-4'
df = df[df['dataset_name'] != 'base-4']

# Prepare old name and base for mapping
mapping_df = df[['image_name', 'dataset_name']].copy()
mapping_df = mapping_df.rename(columns={'image_name': 'old_name', 'dataset_name': 'base'})

# Apply transformations
df = df.rename(columns={
    'dataset_name': 'dataset_name',
    'real_eggs_images': 'Ovos Esperados (E)',
    'counted_eggs_image': 'Ovos observados (O)',
    '[(O-E) / E] *100': 'Diferença (%)',
})

# Add Diferença (O-E) column in absolute value
df['Diferença (O-E)'] = abs(df['Ovos observados (O)'] - df['Ovos Esperados (E)'])

# Diferença (%) as number
df['Diferença_num'] = (df['Diferença (O-E)'] / df['Ovos Esperados (E)']) * 100

# Diferença (%) with 00.00 precision (as string)
df['Diferença (%)'] = df['Diferença_num'].apply(lambda x: f'{x:.2f}')

# Add the >= 50% column and mark with 'x' if >= 50%
df['>= 50%'] = df['Diferença_num'].apply(lambda x: 'x' if x >= 50 else '')

# Drop unnecessary columns
df = df.drop(columns=['dataset_name',  'model_name', 'Diferença_num'])

# Save outputs
df.to_csv('resume.csv', index=False)
mapping_df.to_csv('palheta_mapping.csv', index=False)
