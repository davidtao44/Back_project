# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 17:00:31 2023

UNIVERSIDAD PEDAGOGICA Y TECNOLOGICA DE COLOMBIA
Ingeniería Electrónica
GRUPO DE INVESTIGACION EN ROBOTICA Y AUTOMATIZACION           
Autor: Jaime Andres Moya Africano

Descripcion: Este codigo permite recorrer capa por capa y multiplicar por un factor
cada peso y tomar solo la parte entera de esta operacion.

"""
import numpy as np
import tensorflow as tf
import os

def modify_and_save_weights(model_path, save_path, multiplication_factor=100):
    # Cargar el modelo
    model = tf.keras.models.load_model(model_path)

    # Iterar a través de todas las capas del modelo
    for layer in model.layers:
        # Obtener los pesos de la capa
        weights = layer.get_weights()

        if not weights:
            print(f"La capa {layer.name} no tiene pesos.")
            continue

        # Multiplicar los pesos por el factor y convertirlos en enteros
        modified_weights = []
        for w in weights:
            if isinstance(w, np.ndarray):
                modified_weights.append((w * multiplication_factor).astype(int))
            else:
                modified_weights.append(int(w * multiplication_factor))

        # Asignar los nuevos pesos a la capa
        layer.set_weights(modified_weights)

    # Guardar el modelo modificado en un archivo .h5
    model.save(save_path)
    print(f"Modelo modificado y guardado en: {save_path}")
    return save_path


# model_path = r'C:\Users\PC\Documents\UNIVERSIDAD\Proyecto_g\Scripts_pruebas\lenet_model.h5'
# save_path = r'C:\Users\PC\Documents\UNIVERSIDAD\Proyecto_g\Scripts_pruebas\lenet_model_cuantizado.h5'

# modify_and_save_weights(model_path, save_path)

