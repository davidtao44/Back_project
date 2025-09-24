import numpy as np
import cv2
from scipy.signal import correlate
import pandas as pd
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import tensorflow as tf
import uuid
import time
from .bitflip_injector import BitflipFaultInjector
from .weight_fault_injector import WeightFaultInjector

class ManualInference:
    """
    Clase para realizar inferencia manual capa por capa en LeNet-5
    y generar archivos Excel e imágenes de cada capa.
    """
    
    def __init__(self, model_path: str, output_dir: str = "layer_outputs", session_id: str = None, fault_config: Dict[str, Any] = None):
        self.model = tf.keras.models.load_model(model_path)
        
        # Generar ID único de sesión si no se proporciona
        if session_id is None:
            session_id = f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Crear directorio único para esta sesión
        self.session_id = session_id
        self.base_output_dir = output_dir
        self.output_dir = os.path.join(output_dir, session_id)
        
        # Inicializar inyectores de fallos
        self.fault_injector = BitflipFaultInjector()  # Para activaciones
        self.weight_fault_injector = WeightFaultInjector()  # Para pesos
        self.fault_enabled = False
        self.weight_fault_enabled = False
        self.fault_results = []
        
        # Configurar fallos si se proporcionan
        if fault_config:
            print(f"🔧 DEBUG ManualInference: Configurando fallos con: {fault_config}")
            self.configure_fault_injection(fault_config)
        else:
            print("ℹ️ DEBUG ManualInference: No se proporcionó configuración de fallos")
             
        # Inicializar variables de estado
        self.layer_outputs = {}
        self.create_output_directory()
    
    def create_output_directory(self):
        """Crear directorio de salida si no existe"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def configure_fault_injection(self, fault_config: Dict[str, Any]):
        """
        Configurar inyección de fallos.
        
        Args:
            fault_config: Diccionario con configuración de fallos. Puede ser:
                - Estructura legacy: {enabled, layers} para fallos en activaciones
                - Estructura nueva: {activation_faults: {enabled, layers}, weight_faults: {enabled, layers}}
        """
        print(f"🔧 DEBUG configure_fault_injection: Configuración recibida: {fault_config}")
        
        # Detectar tipo de configuración
        if 'activation_faults' in fault_config or 'weight_faults' in fault_config:
            # Nueva estructura combinada
            print("🔧 DEBUG configure_fault_injection: Procesando configuración combinada")
            
            # Configurar fallos en activaciones
            activation_config = fault_config.get('activation_faults')
            if activation_config:
                self.fault_enabled = activation_config.get('enabled', False)
                print(f"🔧 DEBUG configure_fault_injection: fault_enabled = {self.fault_enabled}")
                
                if self.fault_enabled:
                    self.fault_injector.clear_faults()
                    layers_config = activation_config.get('layers', {})
                    print(f"🔧 DEBUG configure_fault_injection: layers_config = {layers_config}")
                    for layer_name, layer_config in layers_config.items():
                        print(f"🔧 DEBUG configure_fault_injection: Configurando capa {layer_name} con {layer_config}")
                        self.fault_injector.configure_fault(layer_name, layer_config)
                else:
                    print("ℹ️ DEBUG configure_fault_injection: Inyección de fallos en activaciones deshabilitada")
            
            # Configurar fallos en pesos
            weight_config = fault_config.get('weight_faults')
            if weight_config:
                self.weight_fault_enabled = weight_config.get('enabled', False)
                print(f"🔧 DEBUG configure_fault_injection: weight_fault_enabled = {self.weight_fault_enabled}")
                
                if self.weight_fault_enabled:
                    self.weight_fault_injector.clear_faults()
                    self.weight_fault_injector.backup_original_weights(self.model)
                    
                    weight_layers_config = weight_config.get('layers', {})
                    print(f"🔧 DEBUG configure_fault_injection: weight_layers_config = {weight_layers_config}")
                    
                    for layer_name, layer_config in weight_layers_config.items():
                        print(f"🔧 DEBUG configure_fault_injection: Configurando fallos en pesos para capa {layer_name}")
                        self.weight_fault_injector.configure_fault(layer_name, layer_config)
                    
                    weight_faults = self.weight_fault_injector.inject_faults_in_weights(self.model)
                    print(f"✅ Inyectados {len(weight_faults)} fallos en pesos del modelo")
                else:
                    print("ℹ️ DEBUG configure_fault_injection: Inyección de fallos en pesos deshabilitada")
        else:
            # Estructura legacy (solo activaciones)
            print("🔧 DEBUG configure_fault_injection: Procesando configuración legacy")
            self.fault_enabled = fault_config.get('enabled', False)
            print(f"🔧 DEBUG configure_fault_injection: fault_enabled = {self.fault_enabled}")
            
            if self.fault_enabled:
                self.fault_injector.clear_faults()
                layers_config = fault_config.get('layers', {})
                print(f"🔧 DEBUG configure_fault_injection: layers_config = {layers_config}")
                for layer_name, layer_config in layers_config.items():
                    print(f"🔧 DEBUG configure_fault_injection: Configurando capa {layer_name} con {layer_config}")
                    self.fault_injector.configure_fault(layer_name, layer_config)
            else:
                print("ℹ️ DEBUG configure_fault_injection: Inyección de fallos deshabilitada")
    
    def apply_fault_injection(self, activations: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Aplicar inyección de fallos a las activaciones de una capa.
        
        Args:
            activations: Activaciones de la capa
            layer_name: Nombre de la capa
            
        Returns:
            Activaciones modificadas (con o sin fallos)
        """
        print(f"🔧 DEBUG apply_fault_injection: Procesando capa {layer_name}, fault_enabled = {self.fault_enabled}")
        
        if not self.fault_enabled:
            print(f"ℹ️ DEBUG apply_fault_injection: Fallos deshabilitados para capa {layer_name}")
            return activations
            
        print(f"🔧 DEBUG apply_fault_injection: Intentando inyectar fallos en capa {layer_name}")
        modified_activations, injected_faults = self.fault_injector.inject_faults_in_activations(
            activations, layer_name
        )
        
        # Registrar fallos inyectados
        if injected_faults:
            self.fault_results.extend(injected_faults)
            print(f"Inyectados {len(injected_faults)} fallos en capa {layer_name}")
            for i, fault in enumerate(injected_faults):
                diff = abs(fault['modified_value'] - fault['original_value'])
                print(f"  Fallo {i+1}: pos={fault['position']}, bit={fault['bit_position']}, "
                      f"original={fault['original_value']:.6f}, "
                      f"modificado={fault['modified_value']:.6f}, "
                      f"diferencia={diff:.6f}")
            
        return modified_activations
    
    def get_fault_summary(self) -> Dict[str, Any]:
        """Obtener resumen de fallos inyectados en esta inferencia."""
        summary = {
            'activation_fault_injection_enabled': self.fault_enabled,
            'weight_fault_injection_enabled': self.weight_fault_enabled,
            'session_id': self.session_id
        }
        
        # Resumen de fallos en activaciones
        if self.fault_enabled:
            activation_summary = self.fault_injector.get_fault_summary()
            summary['activation_faults'] = activation_summary
        else:
            summary['activation_faults'] = {'total_faults': 0, 'faults_by_layer': {}, 'fault_details': []}
            
        # Resumen de fallos en pesos
        if self.weight_fault_enabled:
            weight_summary = self.weight_fault_injector.get_fault_summary()
            summary['weight_faults'] = weight_summary
        else:
            summary['weight_faults'] = {'total_faults': 0, 'faults_by_layer': {}, 'faults_by_type': {}, 'fault_details': []}
            
        # Totales combinados
        total_activation_faults = summary['activation_faults']['total_faults']
        total_weight_faults = summary['weight_faults']['total_faults']
        summary['total_faults'] = total_activation_faults + total_weight_faults
        
        return summary
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocesar imagen para LeNet-5"""
        # Convertir bytes a imagen PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convertir a escala de grises
        if image.mode != 'L':
            image = image.convert('L')
        
        # Redimensionar a 32x32
        image = image.resize((32, 32))
        
        # Convertir a numpy array
        imagen = np.array(image)
        imagen = np.expand_dims(imagen, axis=-1)  # Agregar dimensión del canal
        imagen = np.expand_dims(imagen, axis=0)   # Agregar dimensión del batch
        
        return imagen
    
    def save_feature_maps_to_excel(self, feature_maps: np.ndarray, layer_name: str, max_channels: int = None):
        """Guardar mapas de características en archivo Excel"""
        try:
            if len(feature_maps.shape) == 3:  # (height, width, channels)
                num_channels = feature_maps.shape[-1]
                if max_channels:
                    num_channels = min(num_channels, max_channels)
                
                archivo_salida = os.path.join(self.output_dir, f"{layer_name}.xlsx")
                data_dict = {}
                
                for i in range(num_channels):
                    df = pd.DataFrame(feature_maps[:, :, i])
                    data_dict[f"Canal {i+1}"] = df
                
                # Exportar a Excel
                with pd.ExcelWriter(archivo_salida, engine="openpyxl") as writer:
                    for canal, df in data_dict.items():
                        df.to_excel(writer, sheet_name=canal, index=False, header=False)
                
                #print(f"Archivo Excel creado: {archivo_salida}")
                return archivo_salida
            
            elif len(feature_maps.shape) == 1:  # Vector 1D
                archivo_salida = os.path.join(self.output_dir, f"{layer_name}.xlsx")
                df = pd.DataFrame(feature_maps.reshape(-1, 1), columns=["Valores"])
                df.to_excel(archivo_salida, index=False, header=True)
                #print(f"Archivo Excel creado: {archivo_salida}")
                return archivo_salida
            
            else:
                print(f"Forma no soportada para {layer_name}: {feature_maps.shape}")
                return None
                
        except Exception as e:
            print(f"Error al crear archivo Excel para {layer_name}: {str(e)}")
            return None
    
    def save_feature_maps_as_images(self, feature_maps: np.ndarray, layer_name: str, max_channels: int = None):
        """Guardar mapas de características como imágenes"""
        try:
            if len(feature_maps.shape) != 3:
                print(f"No se pueden crear imágenes para {layer_name}: forma {feature_maps.shape}")
                return []
            
            num_channels = feature_maps.shape[-1] if max_channels is None else min(feature_maps.shape[-1], max_channels)
            image_paths = []
            
            for i in range(num_channels):
                # Normalizar valores para visualización
                feature_map = feature_maps[:, :, i]
                normalized = ((feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8) * 255).astype(np.uint8)
                
                # Guardar como imagen
                image_path = os.path.join(self.output_dir, f"{layer_name}_canal_{i+1}.png")
                success = cv2.imwrite(image_path, normalized)
                
                if success:
                    image_paths.append(image_path)
                   # print(f"Imagen creada: {image_path}")
                else:
                    print(f"Error al crear imagen: {image_path}")
            
            return image_paths
            
        except Exception as e:
            print(f"Error al crear imágenes para {layer_name}: {str(e)}")
            return []
    
    def conv2d_manual(self, input_data: np.ndarray, filters: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """Aplicar convolución 2D manualmente"""
        if len(input_data.shape) == 2:  # Primera capa convolucional
            num_filters = filters.shape[-1]
            kernel_size = filters.shape[0]
            
            feature_maps = np.zeros((
                input_data.shape[0] - kernel_size + 1,
                input_data.shape[1] - kernel_size + 1,
                num_filters
            ))
            
            for i in range(num_filters):
                filtro = filters[:, :, 0, i]
                feature_maps[:, :, i] = correlate(input_data, filtro, mode="valid") + biases[i]
        
        else:  # Capas convolucionales posteriores
            num_filters = filters.shape[-1]
            input_channels = filters.shape[2]
            kernel_size = filters.shape[0]
            
            feature_maps = np.zeros((
                input_data.shape[0] - kernel_size + 1,
                input_data.shape[1] - kernel_size + 1,
                num_filters
            ))
            
            for i in range(num_filters):
                feature_map_sum = np.zeros((feature_maps.shape[0], feature_maps.shape[1]))
                for j in range(input_channels):
                    filtro = filters[:, :, j, i]
                    feature_map_sum += correlate(input_data[:, :, j], filtro, mode="valid")
                feature_maps[:, :, i] = feature_map_sum + biases[i]
        
        return feature_maps
    
    def maxpool2d_manual(self, input_data: np.ndarray, pool_size: Tuple[int, int], strides: Tuple[int, int]) -> np.ndarray:
        """Aplicar MaxPooling 2D manualmente"""
        pooled_height = (input_data.shape[0] - pool_size[0]) // strides[0] + 1
        pooled_width = (input_data.shape[1] - pool_size[1]) // strides[1] + 1
        num_channels = input_data.shape[2]
        
        pooled_output = np.zeros((pooled_height, pooled_width, num_channels))
        
        for c in range(num_channels):
            for h in range(pooled_height):
                for w in range(pooled_width):
                    h_start = h * strides[0]
                    w_start = w * strides[1]
                    h_end = h_start + pool_size[0]
                    w_end = w_start + pool_size[1]
                    pooled_output[h, w, c] = np.max(input_data[h_start:h_end, w_start:w_end, c])
        
        return pooled_output
    
    def dense_manual(self, input_data: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """Aplicar capa densa manualmente"""
        return np.dot(input_data, weights) + biases
    
    def relu_activation(self, x: np.ndarray) -> np.ndarray:
        """Aplicar activación ReLU"""
        return np.maximum(0, x)
    
    def softmax_activation(self, x: np.ndarray) -> np.ndarray:
        """Aplicar activación Softmax"""
        exp_values = np.exp(x - np.max(x))
        return exp_values / np.sum(exp_values)
    
    def perform_manual_inference(self, image_data: bytes) -> Dict[str, Any]:
        """Realizar inferencia manual completa"""
        print(f"Iniciando inferencia manual. Directorio de salida: {self.output_dir}")
        
        # Preprocesar imagen
        imagen = self.preprocess_image(image_data)
        imagen_procesada = imagen[0, :, :, 0]  # (32, 32)
        
        results = {
            "layer_outputs": {},
            "excel_files": [],
            "image_files": [],
            "final_prediction": {}
        }
        
        # ==================== PRIMERA CAPA CONVOLUCIONAL ====================
        conv_layer1 = self.model.layers[0]
        filters1, biases1 = conv_layer1.get_weights()
        
        feature_maps1 = self.conv2d_manual(imagen_procesada, filters1, biases1)
        feature_maps1 = self.relu_activation(feature_maps1)
        
        # Aplicar inyección de fallos si está habilitada
        feature_maps1 = self.apply_fault_injection(feature_maps1, "conv2d_1")
        
        # Guardar resultados
        excel_file1 = self.save_feature_maps_to_excel(feature_maps1, "conv2d_1")
        image_files1 = self.save_feature_maps_as_images(feature_maps1, "conv2d_1")
        
        results["layer_outputs"]["conv2d_1"] = tuple(int(x) for x in feature_maps1.shape)
        results["excel_files"].append(excel_file1)
        results["image_files"].extend(image_files1)
        
        # ==================== PRIMER MAXPOOLING ====================
        pool_layer1 = self.model.layers[1]
        pool_size1 = pool_layer1.pool_size
        strides1 = pool_layer1.strides
        
        pooled_maps1 = self.maxpool2d_manual(feature_maps1, pool_size1, strides1)
        
        # Aplicar inyección de fallos si está habilitada
        pooled_maps1 = self.apply_fault_injection(pooled_maps1, "maxpooling2d_1")
        
        # Guardar resultados
        excel_file_pool1 = self.save_feature_maps_to_excel(pooled_maps1, "maxpooling2d_1")
        image_files_pool1 = self.save_feature_maps_as_images(pooled_maps1, "maxpooling2d_1")
        
        results["layer_outputs"]["maxpooling2d_1"] = tuple(int(x) for x in pooled_maps1.shape)
        results["excel_files"].append(excel_file_pool1)
        results["image_files"].extend(image_files_pool1)
        
        # ==================== SEGUNDA CAPA CONVOLUCIONAL ====================
        conv_layer2 = self.model.layers[2]
        filters2, biases2 = conv_layer2.get_weights()
        
        feature_maps2 = self.conv2d_manual(pooled_maps1, filters2, biases2)
        feature_maps2 = self.relu_activation(feature_maps2)
        
        # Aplicar inyección de fallos si está habilitada
        feature_maps2 = self.apply_fault_injection(feature_maps2, "conv2d_2")
        
        # Guardar resultados
        excel_file2 = self.save_feature_maps_to_excel(feature_maps2, "conv2d_2")
        image_files2 = self.save_feature_maps_as_images(feature_maps2, "conv2d_2")
        
        results["layer_outputs"]["conv2d_2"] = tuple(int(x) for x in feature_maps2.shape)
        results["excel_files"].append(excel_file2)
        results["image_files"].extend(image_files2)
        
        # ==================== SEGUNDO MAXPOOLING ====================
        pool_layer2 = self.model.layers[3]
        pool_size2 = pool_layer2.pool_size
        strides2 = pool_layer2.strides
        
        pooled_maps2 = self.maxpool2d_manual(feature_maps2, pool_size2, strides2)
        
        # Aplicar inyección de fallos si está habilitada
        pooled_maps2 = self.apply_fault_injection(pooled_maps2, "maxpooling2d_2")
        
        # Guardar resultados
        excel_file_pool2 = self.save_feature_maps_to_excel(pooled_maps2, "maxpooling2d_2")
        image_files_pool2 = self.save_feature_maps_as_images(pooled_maps2, "maxpooling2d_2")
        
        results["layer_outputs"]["maxpooling2d_2"] = tuple(int(x) for x in pooled_maps2.shape)
        results["excel_files"].append(excel_file_pool2)
        results["image_files"].extend(image_files_pool2)
        
        # ==================== FLATTEN ====================
        flatten_output = pooled_maps2.flatten()
        
        # Aplicar inyección de fallos si está habilitada
        flatten_output = self.apply_fault_injection(flatten_output, "flatten")
        
        # Guardar resultados
        excel_file_flatten = self.save_feature_maps_to_excel(flatten_output, "flatten")
        results["layer_outputs"]["flatten"] = tuple(int(x) for x in flatten_output.shape)
        results["excel_files"].append(excel_file_flatten)
        
        # ==================== PRIMERA CAPA DENSA (120 neuronas) ====================
        dense_layer1 = self.model.layers[5]
        weights1, biases1_dense = dense_layer1.get_weights()
        
        dense_output1 = self.dense_manual(flatten_output, weights1, biases1_dense)
        dense_output1 = self.relu_activation(dense_output1)
        
        # Aplicar inyección de fallos si está habilitada
        dense_output1 = self.apply_fault_injection(dense_output1, "dense_1")
        
        # Guardar resultados
        excel_file_dense1 = self.save_feature_maps_to_excel(dense_output1, "dense_1")
        results["layer_outputs"]["dense_1"] = tuple(int(x) for x in dense_output1.shape)
        results["excel_files"].append(excel_file_dense1)
        
        # ==================== SEGUNDA CAPA DENSA (84 neuronas) ====================
        dense_layer2 = self.model.layers[6]
        weights2, biases2_dense = dense_layer2.get_weights()
        
        dense_output2 = self.dense_manual(dense_output1, weights2, biases2_dense)
        dense_output2 = self.relu_activation(dense_output2)
        
        # Aplicar inyección de fallos si está habilitada
        dense_output2 = self.apply_fault_injection(dense_output2, "dense_2")
        
        # Guardar resultados
        excel_file_dense2 = self.save_feature_maps_to_excel(dense_output2, "dense_2")
        results["layer_outputs"]["dense_2"] = tuple(int(x) for x in dense_output2.shape)
        results["excel_files"].append(excel_file_dense2)
        
        # ==================== CAPA FINAL (10 neuronas + Softmax) ====================
        dense_layer3 = self.model.layers[7]
        weights3, biases3_dense = dense_layer3.get_weights()
        
        dense_output3 = self.dense_manual(dense_output2, weights3, biases3_dense)
        
        # Aplicar inyección de fallos si está habilitada (antes del softmax)
        dense_output3 = self.apply_fault_injection(dense_output3, "dense_3")
        
        softmax_output = self.softmax_activation(dense_output3)
        
        # Guardar resultados
        excel_file_softmax = self.save_feature_maps_to_excel(softmax_output, "softmax")
        results["layer_outputs"]["softmax"] = tuple(int(x) for x in softmax_output.shape)
        results["excel_files"].append(excel_file_softmax)
        
        # ==================== PREDICCIÓN FINAL ====================
        predicted_class = int(np.argmax(softmax_output))
        confidence = float(np.max(softmax_output))
        all_probabilities = softmax_output.tolist()
        
        results["final_prediction"] = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }
        
        # Filtrar archivos None de las listas
        results["excel_files"] = [f for f in results["excel_files"] if f is not None]
        results["image_files"] = [f for f in results["image_files"] if f is not None]
        
        # Ordenar archivos por orden de capas
        def get_layer_order(filename):
            layer_order = {
                'conv2d_1': 1,
                'maxpooling2d_1': 2,
                'conv2d_2': 3,
                'maxpooling2d_2': 4,
                'flatten': 5,
                'dense_1': 6,
                'dense_2': 7,
                'softmax': 8
            }
            for layer_name, order in layer_order.items():
                if layer_name in filename:
                    return order
            return 999  # Para archivos no reconocidos
        
        results["excel_files"].sort(key=lambda x: get_layer_order(os.path.basename(x)))
        results["image_files"].sort(key=lambda x: (get_layer_order(os.path.basename(x)), os.path.basename(x)))
        
        # Agregar información de inyección de fallos
        results["fault_injection"] = self.get_fault_summary()
        results["session_id"] = self.session_id
        
        print(f"Inferencia completada. Archivos Excel: {len(results['excel_files'])}, Imágenes: {len(results['image_files'])}")
        
        return results
