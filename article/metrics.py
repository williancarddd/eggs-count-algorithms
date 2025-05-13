import pandas as pd
import numpy as np
from scipy import stats

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Mean Absolute Error (MAE) → usando scipy
    mae = stats.tmean(np.abs(y_true - y_pred))  # tmean == trimmed mean, no trimming → same as mean

    # Mean Relative Error (MRE) → usando scipy
    mre = stats.tmean(np.abs((y_true - y_pred) / y_true))

    # Pearson Correlation Coefficient → scipy
    pearson_corr, _ = stats.pearsonr(y_true, y_pred)

    # R-squared → usando linregress de scipy
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
    print(f"{metric}: {value:.4f}")
