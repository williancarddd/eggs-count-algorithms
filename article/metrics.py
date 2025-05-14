import pandas as pd
import numpy as np
from scipy import stats

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Mean Absolute Error (MAE)
    mae = stats.tmean(np.abs(y_true - y_pred))

    # Filtra casos onde y_true != 0 para MRE
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mre = stats.tmean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
    else:
        mre = np.nan  # ou 0, ou lançar uma exceção, dependendo do uso

    # Pearson Correlation
    pearson_corr, _ = stats.pearsonr(y_true, y_pred)

    # R-squared
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    r_squared = r_value ** 2

    return {
        'MAE': mae,
        'MRE': mre,
        'Pearson Correlation': pearson_corr,
        'R²': r_squared
    }

# Leitura do arquivo resume.csv
df = pd.read_csv('resume.csv')

# Extração das colunas relevantes
y_true = df['Ovos Esperados (E)']
y_pred = df['Ovos observados (O)']

# Cálculo das métricas
results = calculate_metrics(y_true, y_pred)

# Impressão dos resultados formatados
print("==== Métricas Calculadas ====")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}" if not np.isnan(value) else f"{metric}: NA (valores esperados iguais a zero)")
