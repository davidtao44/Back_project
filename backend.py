import numpy as np
import tensorflow as tf
import keras
import os

#! Recordar que el orden de las matrices de los pesos que se obtienen de netron y tensorflow, 
#! es diferente a la forma convencional de la matrices de la arquitectura de LENET.
#Por lo tanto se le realiza el tratamiento para que entregue las matrices en el orden por canal. #! la matriz para cada canal 

def to_binary_c2(value, bits=8): #cantidad de bits pesos
    """Convierte un valor a binario en complemento a 2."""
    if value < 0:
        value = (1 << bits) + value
    return format(value, f'0{bits}b')

def extract_filters(model, output_dir="output_filters"):
    """
    Extrae los filtros y sesgos de cada capa convolucional y los guarda en archivos .txt.
    Cada capa convolucional tendrá su propio archivo.
    """
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            filters, biases = layer.get_weights()
            num_filters = filters.shape[-1]  # Número de filtros
            num_channels = filters.shape[-2]  # Número de canales de entrada

            # Crear un archivo de texto para la capa actual
            layer_filename = os.path.join(output_dir, f"layer_{i+1}_filters.txt")
            with open(layer_filename, "w") as file:
                # Escribir los filtros en el archivo
                file.write(f"Filtros de la capa convolucional {i+1}:\n")
                for j in range(num_filters):  # Recorrer cada filtro
                    for c in range(num_channels):  # Recorrer cada canal
                        filter_matrix = filters[:, :, c, j]  # Obtener el filtro para el canal c
                        file.write(f"constant FMAP_{c+1}_{j+1}: FILTER_TYPE:= (\n")
                        for k in range(filter_matrix.shape[0]):  # Filas del filtro
                            row = ["\"" + to_binary_c2(int(filter_matrix[k, l])) + "\"" 
                                   for l in range(filter_matrix.shape[1])]
                            file.write("    (" + ",".join(row) + ")" + ("," if k < filter_matrix.shape[0] - 1 else "") + "\n")
                        file.write(");\n\n")

                # Escribir los sesgos en el archivo
                file.write(f"Sesgos de la capa convolucional {i+1}:\n")
                for idx, bias in enumerate(biases):
                    bias_bin = to_binary_c2(int(bias), bits=24)  # Cantidad de bits Sesgos
                    file.write(f"constant BIAS_VAL_{idx+1}: signed (BIASES_SIZE-1 downto 0) := \"{bias_bin}\";\n")
                file.write("\n")

            print(f"Archivo generado: {layer_filename}")



# h5_file = "C:/Users/PC/Documents/UNIVERSIDAD/Proyecto_g/Proyecto/Obtención_VHDL/Back/lenet_model_cuantizado.h5"
h5_file = os.path.join(os.path.dirname(__file__), 'models', 'lenet_model_cuantizado.h5')


# Cargar el modelo
modelo = keras.models.load_model(h5_file)

# Extraer filtros y sesgos, y guardarlos en archivos .txt
extract_filters(modelo, output_dir="output_filters")


#Obtencion de los pesos de la fully connected 

# Crear directorio para guardar los archivos
os.makedirs("pesos_biases", exist_ok=True)


for layer in modelo.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        print(f"Procesando capa: {layer.name}")

        pesos = layer.get_weights()[0]  # Pesos
        biases = layer.get_weights()[1]  # Biases

        num_filas, num_columnas = pesos.shape  # (Entradas, Neuronas)

        # Guardar pesos en formato VHDL (recorriendo por columnas)
        with open(f"pesos_biases/{layer.name}_pesos.txt", "w") as f_pesos:
            for j in range(num_columnas):  # Recorrer por columnas primero
                for i in range(num_filas):  # Luego recorrer filas
                    bin_value = to_binary_c2(int(pesos[i, j]))  # Convertir a binario C2
                    f_pesos.write(f'constant FMAP_{j+1}_{i+1}: signed(WEIGHT_SIZE-1 downto 0) := "{bin_value}";\n')

        # Guardar biases en formato VHDL
        with open(f"pesos_biases/{layer.name}_biases.txt", "w") as f_biases:
            for i, bias in enumerate(biases, start=1):  # Iniciar en 1
                bin_value = to_binary_c2(int(bias))
                f_biases.write(f'constant BIAS_VAL_{i}: signed(BIASES_SIZE-1 downto 0) := "{bin_value}";\n')

print("Pesos y biases guardados en la carpeta 'pesos_biases' en formato VHDL (binario complemento a 2).")

