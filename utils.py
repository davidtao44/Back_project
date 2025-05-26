import tensorflow as tf
from keras import layers, models
import os
import time
from models import CNNConfig

# Crear directorio para guardar modelos si no existe
os.makedirs("models", exist_ok=True)

def create_cnn(config: CNNConfig):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=tuple(config.input_shape)))
    
    for layer in config.conv_layers:
        model.add(layers.Conv2D(layer.filters, (layer.kernel_size, layer.kernel_size), activation=layer.activation))
        if layer.pooling:
            if layer.pooling == "max":
                model.add(layers.MaxPooling2D((2, 2)))
            elif layer.pooling == "average":
                model.add(layers.AveragePooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    for layer in config.dense_layers:
        model.add(layers.Dense(layer.units, activation=layer.activation))
    
    model.add(layers.Dense(config.output_units, activation=config.output_activation))
    return model

def generate_model_filename(model_name: str) -> str:
    # Formatear la fecha como DD_MM_YYYY
    current_date = time.strftime('%d_%m_%Y', time.localtime())
    
    # Usar el nombre proporcionado por el usuario o el predeterminado "model"
    user_name = model_name.replace(" ", "_")  # Reemplazar espacios con guiones bajos
    
    # Crear nombre de archivo con el formato DD_MM_YYYY_nombreUsuario
    return f"models/architecture_{current_date}_{user_name}.h5"