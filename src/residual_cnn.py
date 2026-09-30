# -*- coding: utf-8 -*-
"""
================================================================================
 RED N° 2 — MODELO RESIDUAL
 TFM: "Comparación de redes neuronales para la detección de Alzheimer
       mediante análisis de imágenes médicas" (UNIR, 2026)
 Autor de la arquitectura: Mateo Arteaga
 Repositorio original: https://github.com/mateoart1014/master_tfm
================================================================================
"""

import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import layers, models

# ==============================================================================
# 1. PARÁMETROS GLOBALES (idénticos a la Red N° 1)
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
# 2. ARQUITECTURA RESIDUAL (Red N° 2)
# ==============================================================================
def bloque_residual_custom(input_tensor, num_filtros):
    """Bloque con conexión residual para evitar el desvanecimiento del gradiente."""
    x = layers.Conv2D(num_filtros, (3, 3), padding="same",
                      kernel_initializer="he_normal")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filtros, (3, 3), padding="same",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)

    # Proyección 1x1 cuando el número de canales cambia
    if input_tensor.shape[-1] != num_filtros:
        proyeccion = layers.Conv2D(num_filtros, (1, 1), padding="same",
                                   kernel_initializer="he_normal")(input_tensor)
        proyeccion = layers.BatchNormalization()(proyeccion)
    else:
        proyeccion = input_tensor

    x = layers.add([x, proyeccion])
    x = layers.Activation("relu")(x)
    return x


def crear_modelo_residual(input_shape=INPUT_SHAPE, num_clases=NUM_CLASES):
    """Construcción de la topología de la Red N° 2."""
    inputs = layers.Input(shape=input_shape, name="Entrada_RM")

    x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding="same",
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = bloque_residual_custom(x, num_filtros=64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = bloque_residual_custom(x, num_filtros=128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_clases, activation="softmax", name="Output_Softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_ResNet_Net2")


# ==============================================================================
# 3. CARGA DE DATOS (mismo pipeline que la Red N° 1)
# ==============================================================================
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


# ==============================================================================
# 4. BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":

    path = kagglehub.dataset_download(
        "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"
    )
    DATA_DIR = encontrar_dir_clases(path)
    print("Carpeta de clases detectada:", DATA_DIR)

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

    # Compilación
    modelo = crear_modelo_residual()
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INICIAL),
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
    ax1.set_title("Accuracy over epochs (Red N° 2 Residual)")
    ax1.set_xlabel("Epochs"); ax1.set_ylabel("Accuracy"); ax1.legend()
    ax2.plot(history.history["loss"], label="Training Error")
    ax2.plot(history.history["val_loss"], label="Validation Error")
    ax2.set_title("Loss over epochs (Red N° 2 Residual)")
    ax2.set_xlabel("Epochs"); ax2.set_ylabel("Error"); ax2.legend()
    plt.tight_layout()
    plt.savefig("curvas_aprendizaje_net2.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Inferencia sobre validación
    y_true, y_pred = [], []
    for imagenes, etiquetas in val_ds:
        preds = modelo.predict(imagenes, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(etiquetas.numpy(), axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

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
    plt.title("Matriz de Confusión Real - Red Neuronal N° 2 (Residual)",
              fontsize=13, pad=20)
    plt.xlabel("Predicciones del Modelo", fontsize=11, labelpad=10)
    plt.ylabel("Valores Reales (Ground Truth)", fontsize=11, labelpad=10)
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig("matriz_confusion_net2.png", dpi=300)
    plt.show()
