# Python 3.11.5.
# FILE: CNN_k_reducida.py
# AUTHOR: José Luis López Carmona
# CREATE DATE: 17/04/2024

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

###############################################################################
###############################################################################

import time

# Saving initial time
start_time = time.time()

################################################################################
################################################################################
def load_data(input_file):
    # Read CSV
    data = pd.read_csv(input_file)
    
    # Clean
    labels = data['seq_id'].str.split('_').str[1]
    
    # Reindex
    data.set_index(labels, inplace=True)
    
    # Deleta last index
    data.drop('seq_id', axis=1, inplace=True)
    
    # Convert the rest of column to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    return labels, data

def normalize (input_data):
    maximum = np.max(input_data)
    data_normalize = input_data/maximum
    return(data_normalize)

def apply_pca(data, variance_threshold=0.95):
    # PCA application
    pca = PCA(n_components=variance_threshold)  
    principal_components = pca.fit_transform(data)
    print(f"Se conservaron {pca.n_components_} componentes principales que explican al menos {variance_threshold * 100}% de la varianza total.")
    return principal_components

def create_model(nb_classes, input_length, input_features):
    # Build network and compile
    model = Sequential([
        # 
        Conv1D(16, 3, padding='same', activation='relu', input_shape=(input_length, input_features)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        #
        Conv1D(32, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
       #
        Conv1D(64, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        #
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        # Capa de salida
        Dense(nb_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.metrics.Precision(name = 'precision'),
            tf.metrics.Recall(name = 'recall'),
            tf.metrics.F1Score(average='macro')
        ]
    )
    return model

def split (n,data_n, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(data_n, y_encoded, test_size=n, random_state=42)
    return [X_train, y_train, X_test, y_test]

def train_and_plot_metrics(model, X_train, X_test, y_train, y_test, 
                           epochs=100, batch_size=20):
    # Train and history
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Saving metrics
    metrics = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'precision': history.history['precision'],
        'val_precision': history.history['val_precision'],
        'recall': history.history['recall'],
        'val_recall': history.history['val_recall'],
        'f1_score': history.history['f1_score'],
        'val_f1_score': history.history['val_f1_score']
    }

    return metrics

##################################################################################
##################################################################################

def plottings(metrics):
    plt.figure(figsize=(20, 6))

    # Loss train and test
    plt.subplot(1, 5, 1)
    plt.plot(metrics['loss'], label='Pérdida de Entrenamiento')
    plt.plot(metrics['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.legend()

    # Accuracy train and test
    plt.subplot(1, 5, 2)
    plt.plot(metrics['accuracy'], label='Accuracy de Entrenamiento')
    plt.plot(metrics['val_accuracy'], label='Accuracy de Validación')
    plt.title('Accuracy durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision train and test
    plt.subplot(1, 5, 3)
    plt.plot(metrics['precision'], label='Precisión de Entrenamiento')
    plt.plot(metrics['val_precision'], label='Precisión de Validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.legend()

    # Recall train and test
    plt.subplot(1, 5, 4)
    plt.plot(metrics['recall'], label='Recall de Entrenamiento')
    plt.plot(metrics['val_recall'], label='Recall de Validación')
    plt.title('Recall durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # F1-score train and test
    plt.subplot(1, 5, 5)
    plt.plot(metrics['f1_score'], label='F1-Score de Entrenamiento', linestyle='--')
    plt.plot(metrics['val_f1_score'], label='F1-Score de Validación', linestyle='--')
    plt.title('F1-Score durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def resumen(history):
    # Extracting metrics
    accuracy = history['accuracy']
    precision = history['precision']
    recall = history['recall']
    f1_score = history['f1_score']
    loss = history['loss']

    val_accuracy = history['val_accuracy']
    val_precision = history['val_precision']
    val_recall = history['val_recall']
    val_f1_score = history['val_f1_score']
    val_loss = history['val_loss']

    # Statistics
    metrics_mean = {
        'accuracy': np.mean(accuracy),
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1_score),
        'loss': np.mean(loss),
        'val_accuracy': np.mean(val_accuracy),
        'val_precision': np.mean(val_precision),
        'val_recall': np.mean(val_recall),
        'val_f1_score': np.mean(val_f1_score),
        'val_loss': np.mean(val_loss)
    }

    metrics_std = {
        'accuracy': np.std(accuracy),
        'precision': np.std(precision),
        'recall': np.std(recall),
        'f1_score': np.std(f1_score),
        'loss': np.std(loss),
        'val_accuracy': np.std(val_accuracy),
        'val_precision': np.std(val_precision),
        'val_recall': np.std(val_recall),
        'val_f1_score': np.std(val_f1_score),
        'val_loss': np.std(val_loss)
    }

    # DataFrame 
    df_mean = pd.DataFrame(metrics_mean, index=['Mean'])
    df_std = pd.DataFrame(metrics_std, index=['Std'])

    # Combine
    df_summary = pd.concat([df_mean, df_std])

    # Transpose cause by structure
    df_summary = df_summary.transpose()

    # Print
    print("Resumen de métricas:")
    print(df_summary)

##################################################################################################
##################################################################################################

# My own path, change
input_file = r'C:\Users\JoseLuisLopezCarmona\Documents\MCD\TFM\Codigo\datos\AMPk7.csv'
input_micro = r'C:\Users\JoseLuisLopezCarmona\Documents\MCD\TFM\Codigo\datos\taxonomy.csv'

# Labels and data
labels, data = load_data(input_file)
data_m = normalize(data)
data_n = apply_pca(data_m)

# Division data
taxonomy = pd.read_csv(input_micro)
Taxon_L = taxonomy.shape[1]
clase = 'Genus'
if clase == 'Genus':
    tax_label = taxonomy.iloc[:,[0,Taxon_L-1]]
    taxon = taxonomy.iloc[0:,-1]
    classes = 100
if clase == 'Class':
    tax_label = taxonomy.iloc[:,[0,2]]
    taxon = taxonomy.iloc[0:,2]
    classes = 3

# Reindex
tax_label.set_index('Sequence', inplace=True)
tax_label_sorted = tax_label.reindex(labels)

# Encoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(tax_label_sorted.values.ravel())

# One-hot encoding
y_encoded = to_categorical(encoded_labels, num_classes=classes)

# train/test y validacion
n = 0.1
X_tT, y_tT, X_val, y_val = split(n,data_n,y_encoded)

##################################################################################################
##################################################################################################

# Cross-validation by hand
fold = 5
for i in range(0,fold):
    X_train, y_train, X_test, y_test = split(0.2,X_tT,y_tT)

    # Model
    length = X_train.shape[0]
    input_features = X_train.shape[1]
    model = create_model(nb_classes=classes,input_length=input_features,input_features=1)

    # Train and summary
    history = train_and_plot_metrics(model, X_train, X_test, y_train, y_test, 
                                     epochs=30, batch_size=20)

    if i > 0:
        H1 = pd.DataFrame(history)
        H2 = pd.concat([H2,H1],axis=0)
    else:
        H2 = pd.DataFrame(history)
        
# Summary
resumen(H2)

# Final time for train
elapsed_time = time.time() - start_time
print(f"El script tardó {elapsed_time:.2f} segundos en completarse.")

# Evaluation model by validation data
evaluation = model.evaluate(X_val, y_val)
evaluation_df = pd.DataFrame({
    "Metric": ["Loss", "Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [evaluation[0], evaluation[1], evaluation[2], evaluation[3], evaluation[4]]
})
print(evaluation_df)

# Summary and plots
model.summary()
plottings(history)
