# -*- coding: utf-8 -*-
"""
================================================================================
 RED N° 1 — MODELO SECUENCIAL
 TFM: "Comparación de redes neuronales para la detección de Alzheimer
       mediante análisis de imágenes médicas" (UNIR, 2026)
 Autora: Maria Paula Guaca Campo
================================================================================
"""

import os
import random

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers

# ==============================================================================
# 1. PARÁMETROS GLOBALES
# ==============================================================================
SEED        = 123
IMG_SIZE    = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE  = 32
EPOCHS      = 15
NUM_CLASES  = 4
VAL_SPLIT   = 0.2
LR_INICIAL  = 1e-3

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# 2. DESCARGA Y CARGA DE DATOS
# ==============================================================================
path = kagglehub.dataset_download(
    "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"
)
print("Dataset descargado en:", path)


def encontrar_dir_clases(root):
    """Localiza la carpeta que contiene directamente las subcarpetas de clase."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    mejor_dir, mejor_conteo = root, -1
    for dirpath, dirnames, _ in os.walk(root):
        subdirs = [d for d in sorted(dirnames) if not d.startswith(".")]
        if len(subdirs) < 2:
            continue
        total, con_imagenes = 0, 0
        for sd in subdirs:
            p = os.path.join(dirpath, sd)
            imgs = [f for f in os.listdir(p) if f.lower().endswith(exts)]
            if imgs:
                con_imagenes += 1
                total += len(imgs)
        if con_imagenes >= 2 and total > mejor_conteo:
            mejor_dir, mejor_conteo = dirpath, total
    return mejor_dir


DATA_DIR = encontrar_dir_clases(path)
print("Carpeta de clases detectada:", DATA_DIR)
print("Subcarpetas:", sorted(os.listdir(DATA_DIR)))

# shuffle=True baraja con la semilla antes de partir,
# dejando las 4 clases estratificadas en train y val.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
)

class_names = train_ds.class_names
print("Clases detectadas:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)


# ==============================================================================
# 2.1 ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
conteos = {}
ejemplos = {}
for clase in sorted(os.listdir(DATA_DIR)):
    carpeta = os.path.join(DATA_DIR, clase)
    if not os.path.isdir(carpeta):
        continue
    archivos = [f for f in os.listdir(carpeta)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    conteos[clase] = len(archivos)
    if archivos:
        ejemplos[clase] = os.path.join(carpeta, random.choice(archivos))

print("\n--- CONTEOS REALES POR CLASE (dataset completo, antes del split) ---")
total = sum(conteos.values())
for c, n in conteos.items():
    print(f"  {c}: {n}  ({100*n/total:.1f} %)")
print(f"  TOTAL: {total}")

# Figura 1: grilla de muestras (1 imagen real por clase)
fig, axes = plt.subplots(1, len(ejemplos), figsize=(4 * len(ejemplos), 4))
for ax, (clase, ruta) in zip(axes, ejemplos.items()):
    ax.imshow(Image.open(ruta), cmap="gray")
    ax.set_title(clase, fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig("figura_muestra_clases.png", dpi=200, bbox_inches="tight")
plt.show()

# Figura 2: distribución de clases (conteo real, dataset completo)
plt.figure(figsize=(7, 5))
plt.bar(conteos.keys(), conteos.values(), color="#4C72B0")
plt.ylabel("Número de imágenes")
plt.title("Distribución de clases en el conjunto de datos completo")
plt.xticks(rotation=20)
for i, (c, n) in enumerate(conteos.items()):
    plt.text(i, n, str(n), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("figura_distribucion.png", dpi=200, bbox_inches="tight")
plt.show()


# ==============================================================================
# 3. ARQUITECTURA SECUENCIAL (Red N° 1)
# ==============================================================================
# Bloques convolucionales apilados LINEALMENTE (sin conexiones residuales),
# filtros 32 -> 64 -> 128 -> 256, con MaxPooling2D entre bloques.
# Sin Rescaling ni aumento (igual que la Red N° 2). La BatchNormalization tras
# cada convolución estabiliza los valores de entrada crudos.

def crear_modelo_secuencial(input_shape, num_clases):
    modelo = models.Sequential(name="Red_N1_Secuencial")
    modelo.add(layers.Input(shape=input_shape, name="Entrada_RM"))

    for filtros, drop in [(32, 0.25), (64, 0.25), (128, 0.30), (256, 0.30)]:
        modelo.add(layers.Conv2D(filtros, (3, 3), padding="same",
                                 kernel_initializer="he_normal"))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.Activation("relu"))
        modelo.add(layers.MaxPooling2D((2, 2)))
        modelo.add(layers.Dropout(drop))

    modelo.add(layers.GlobalAveragePooling2D())
    modelo.add(layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    modelo.add(layers.Dropout(0.5))
    modelo.add(layers.Dense(num_clases, activation="softmax", name="Output_Softmax"))
    return modelo


# ==============================================================================
# 4. BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":

    modelo = crear_modelo_secuencial(INPUT_SHAPE, NUM_CLASES)
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=LR_INICIAL),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.summary()

    print("\n--- Iniciando entrenamiento desde cero ---")
    history = modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Curvas de aprendizaje
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["accuracy"], label="Training Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Accuracy over epochs (Red N° 1 Secuencial)")
    ax1.set_xlabel("Epochs"); ax1.set_ylabel("Accuracy"); ax1.legend()
    ax2.plot(history.history["loss"], label="Training Error")
    ax2.plot(history.history["val_loss"], label="Validation Error")
    ax2.set_title("Loss over epochs (Red N° 1 Secuencial)")
    ax2.set_xlabel("Epochs"); ax2.set_ylabel("Error"); ax2.legend()
    plt.tight_layout()
    plt.savefig("curvas_aprendizaje_net1.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Inferencia sobre validación
    y_true, y_pred = [], []
    for imagenes, etiquetas in val_ds:
        preds = modelo.predict(imagenes, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(etiquetas.numpy(), axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Métricas (labels fijo a las 4 clases -> evita el error de tamaños)
    etiquetas_idx = list(range(NUM_CLASES))
    print("\nAccuracy global:", accuracy_score(y_true, y_pred))
    print("\nTABLA DE MÉTRICAS GLOBALES Y POR CLASE (Classification Report):")
    print(classification_report(y_true, y_pred, labels=etiquetas_idx,
                                target_names=class_names, digits=2))

    matriz = confusion_matrix(y_true, y_pred, labels=etiquetas_idx)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title("Matriz de Confusión Real - Red Neuronal N° 1 (Secuencial)",
              fontsize=13, pad=20)
    plt.xlabel("Predicciones del Modelo", fontsize=11, labelpad=10)
    plt.ylabel("Valores Reales (Ground Truth)", fontsize=11, labelpad=10)
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig("matriz_confusion_net1.png", dpi=300)
    plt.show()
