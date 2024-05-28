import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

###############################################################################
###############################################################################
import time

# Guarda el tiempo de inicio
start_time = time.time()

###############################################################################
###############################################################################

# Función para cargar y preparar los datos
def load_data(input_file):
    data = pd.read_csv(input_file)
    labels = data['seq_id'].str.split('_').str[1]
    data.set_index(labels, inplace=True)
    data.drop('seq_id', axis=1, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    return labels, data

# Función para normalizar los datos
def normalize(input_data):
    return input_data / np.max(input_data)

# Función para preparar los datos según la taxonomía y clase seleccionada
def prepare_data(clase, taxonomy_file, labels, data):
    taxonomy = pd.read_csv(taxonomy_file)
    Taxon_L = taxonomy.shape[1]
    tax_label = taxonomy.iloc[:, [0, Taxon_L-1]] if clase == 'Genus' else taxonomy.iloc[:, [0, 2 if clase == 'Class' else 3 if clase == 'Order' else 4]]
    taxon = taxonomy.iloc[:, -1 if clase == 'Genus' else 2 if clase == 'Class' else 3 if clase == 'Order' else 4]
    classes = 100 if clase == 'Genus' else 3 if clase == 'Class' else 20 if clase == 'Order' else 39
    tax_label.set_index('Sequence', inplace=True)
    tax_label_sorted = tax_label.reindex(labels)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(tax_label_sorted.values.ravel())
    y_encoded = to_categorical(encoded_labels, num_classes=classes)
    data_n = normalize(data)
    return data_n, y_encoded

# Cargar y preparar los datos
input_file = r'C:\Users\JoseLuisLopezCarmona\Documents\MCD\TFM\Codigo\datos\SGk7.csv'
input_micro = r'C:\Users\JoseLuisLopezCarmona\Documents\MCD\TFM\Codigo\datos\taxonomy.csv'
labels, data = load_data(input_file)
data_n, y_encoded = prepare_data('Genus', input_micro, labels, data)
X_train, X_val, y_train, y_val = train_test_split(data_n, y_encoded, test_size=0.1, stratify=np.argmax(y_encoded, axis=1), random_state=42)

# Aplicar SMOTE para equilibrar el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Definir la cuadrícula de parámetros para GridSearchCV 
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Configurar y ejecutar GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, verbose=1, n_jobs=-1)
grid_search.fit(X_train, np.argmax(y_train, axis=1))  # Fit using categorical labels

# Obtener el mejor modelo y realizar predicciones sobre el conjunto de validación
best_rf = grid_search.best_estimator_
y_val_pred = best_rf.predict(X_val)  # Predict using the model

# Convertir predicciones y verdaderas etiquetas de one-hot a categóricas
y_val_dense = np.argmax(y_val, axis=1)  # Convertir verdaderas etiquetas
y_val_pred_dense = y_val_pred  # Las predicciones ya están en formato categórico

# Generar y mostrar el reporte de clasificación
print(best_rf)
print("Mejores parámetros:", grid_search.best_params_)
print("Reporte de clasificación en el conjunto de validación:\n",
      classification_report(y_val_dense, y_val_pred_dense, zero_division=0))

# Calcula el tiempo de ejecución
elapsed_time = time.time() - start_time

# Imprime el tiempo de ejecución
print(f"El script tardó {elapsed_time:.2f} segundos en completarse.")
