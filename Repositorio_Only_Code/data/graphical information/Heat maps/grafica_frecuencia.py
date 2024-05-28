import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(input_file):
    # Leer el archivo CSV, usando la primera columna (seq_id) como índice
    data = pd.read_csv(input_file)
    
    # Extraer y limpiar los identificadores de la columna 'seq_id'
    labels = data['seq_id'].str.split('_').str[1]
    
    # Establecer los identificadores limpios como índice del DataFrame
    data.set_index(labels, inplace=True)
    
    # Eliminar la columna original 'seq_id' ya que ahora es el índice
    data.drop('seq_id', axis=1, inplace=True)
    
    # Convertir todas las columnas restantes a numérico, gestionando errores por si acaso
    data = data.apply(pd.to_numeric, errors='coerce')

    return labels, data

def normalize (input_data):
    maximum = np.max(input_data)
    data_normalize = input_data/maximum
    return(data_normalize)

input_file = r'C:\Users\JoseLuisLopezCarmona\Documents\MCD\TFM\Codigo\datos\AMPk3.csv'
labels, data = load_data(input_file)
data = normalize(data)

plt.figure(figsize=(18, 12))
sns.heatmap(data.iloc[:, 1:], cmap='viridis', yticklabels=False)
plt.title('Mapa Calor k-mers AMP, k=3')
plt.xlabel('Subconjuntos')
plt.ylabel('Secuencias')
plt.show()