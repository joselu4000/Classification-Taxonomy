# Python 3.11.5
# FILE: Heat_maps.py
# AUTHOR: José Luis López Carmona
# CREATE DATE: 29/05/2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(input_file):
    # Load CSV
    data = pd.read_csv(input_file)
    
    # Clean
    labels = data['seq_id'].str.split('_').str[1]
    
    # Reindex
    data.set_index(labels, inplace=True)
    data.drop('seq_id', axis=1, inplace=True)
    
    # Convert rest colmns to numericall
    data = data.apply(pd.to_numeric, errors='coerce')

    return labels, data

def normalize (input_data):
    # Standard data
    maximum = np.max(input_data)
    data_normalize = input_data/maximum
    return(data_normalize)

# My own path
input_file = r'C:\Users\JoseLuisLopezCarmona\Documents\MCD\TFM\Codigo\datos\AMPk3.csv'

# Labels and data
labels, data = load_data(input_file)
data = normalize(data)

# Heat-map
plt.figure(figsize=(18, 12))
sns.heatmap(data.iloc[:, 1:], cmap='viridis', yticklabels=False)
plt.title('Mapa Calor k-mers AMP, k=3')
plt.xlabel('Subconjuntos')
plt.ylabel('Secuencias')
plt.show()
