import numpy as np
import cv2
from scipy.signal import correlate
import pandas as pd
import os
import io
import json
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import tensorflow as tf
import uuid
import time
from .bitflip_injector import BitflipFaultInjector
from .weight_fault_injector import WeightFaultInjector

def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Leer metadatos opcionales de un modelo desde un sidecar <model_path>.meta.json.

    Permite declarar, por modelo, si su entrada debe normalizarse a [0, 1]
    (campo "normalize"). Por defecto no se normaliza, conservando el
    comportamiento histórico.
    """
    meta = {"normalize": False}
    if model_path:
        meta_path = f"{model_path}.meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta.update(json.load(f))
            except Exception as e:
                print(f"⚠️ No se pudo leer metadatos del modelo ({meta_path}): {e}")
    return meta


class ManualInference:
    """
    Clase para realizar inferencia manual capa por capa en LeNet-5
    y generar archivos Excel e imágenes de cada capa.
    """
    
    def __init__(self, model_path: str = None, output_dir: str = "layer_outputs", session_id: str = None, fault_config: Dict[str, Any] = None, model_instance: tf.keras.Model = None, normalize: bool = False):
        # Usar instancia del modelo si se proporciona, sino cargar desde ruta
        if model_instance is not None:
            self.model = model_instance
            print("🔧 DEBUG ManualInference: Usando instancia de modelo proporcionada")
        elif model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
            print(f"🔧 DEBUG ManualInference: Cargando modelo desde {model_path}")
        else:
            raise ValueError("Debe proporcionar model_path o model_instance")
        
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
        self.fault_enabled = False  # Para compatibilidad con configuración legacy
        self.activation_fault_enabled = False  # Para nueva estructura de configuración
        self.weight_fault_enabled = False
        self.fault_results = []
        
        # Configurar fallos si se proporcionan
        if fault_config:
            print(f"🔧 DEBUG ManualInference: Configurando fallos con: {fault_config}")
            self.configure_fault_injection(fault_config)
        else:
            print("ℹ️ DEBUG ManualInference: No se proporcionó configuración de fallos")
             
        # Normalización de entrada (0-1 en vez de 0-255). Por defecto se respeta
        # el comportamiento histórico (sin normalizar) para no alterar LeNet-5.
        self.normalize = normalize

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
                self.activation_fault_enabled = activation_config.get('enabled', False)
                print(f"🔧 DEBUG configure_fault_injection: activation_fault_enabled = {self.activation_fault_enabled}")
                
                if self.activation_fault_enabled:
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
        # Verificar si se está usando la nueva estructura de configuración o la legacy
        fault_enabled = getattr(self, 'activation_fault_enabled', getattr(self, 'fault_enabled', False))
        print(f"🔧 DEBUG apply_fault_injection: Procesando capa {layer_name}, fault_enabled = {fault_enabled}")
        
        if not fault_enabled:
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
        # Considerar tanto la estructura nueva (activation_fault_enabled) como
        # la legacy (fault_enabled).
        activation_enabled = getattr(self, 'activation_fault_enabled', False) or getattr(self, 'fault_enabled', False)

        summary = {
            'activation_fault_injection_enabled': activation_enabled,
            'weight_fault_injection_enabled': self.weight_fault_enabled,
            'session_id': self.session_id
        }

        # Resumen de fallos en activaciones
        if activation_enabled:
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
    
    def diagnose_weight_changes(self) -> Dict[str, Any]:
        """
        Diagnosticar cambios en los pesos del modelo para verificar que los fallos se aplicaron.
        
        Returns:
            Diccionario con información detallada sobre los cambios en pesos
        """
        diagnosis = {
            'weight_fault_enabled': self.weight_fault_enabled,
            'layers_analyzed': [],
            'weight_differences_found': False,
            'total_modified_weights': 0,
            'layer_details': {}
        }
        
        if not self.weight_fault_enabled:
            diagnosis['message'] = "Inyección de fallos en pesos no está habilitada"
            return diagnosis
            
        # Comparar pesos actuales con originales
        for i, layer in enumerate(self.model.layers):
            layer_name = layer.name  # nombre real de Keras (fuente única de verdad)
            
            if not hasattr(layer, 'get_weights') or not layer.get_weights():
                continue
                
            if layer_name not in self.weight_fault_injector.original_weights:
                continue
                
            current_weights = layer.get_weights()
            original_weights = self.weight_fault_injector.original_weights[layer_name]
            
            layer_info = {
                'layer_name': layer_name,
                'layer_type': layer.__class__.__name__,
                'weight_tensors': [],
                'differences_found': False,
                'total_differences': 0
            }
            
            # Comparar cada tensor de pesos
            for idx, (current, original) in enumerate(zip(current_weights, original_weights)):
                tensor_type = 'kernel' if idx == 0 else 'bias'
                
                # Calcular diferencias
                diff_mask = ~np.isclose(current, original, rtol=1e-9, atol=1e-9)
                num_differences = np.sum(diff_mask)
                
                tensor_info = {
                    'tensor_index': idx,
                    'tensor_type': tensor_type,
                    'shape': current.shape,
                    'total_elements': current.size,
                    'modified_elements': int(num_differences),
                    'modification_percentage': (num_differences / current.size) * 100,
                    'max_absolute_difference': 0.0,
                    'modified_positions': []
                }
                
                if num_differences > 0:
                    layer_info['differences_found'] = True
                    layer_info['total_differences'] += num_differences
                    diagnosis['weight_differences_found'] = True
                    diagnosis['total_modified_weights'] += num_differences
                    
                    # Calcular diferencia máxima
                    abs_diff = np.abs(current - original)
                    tensor_info['max_absolute_difference'] = float(np.max(abs_diff))
                    
                    # Obtener posiciones modificadas (limitado a las primeras 5)
                    modified_positions = np.where(diff_mask)
                    for j in range(min(5, num_differences)):
                        pos = tuple(int(modified_positions[k][j]) for k in range(len(modified_positions)))
                        tensor_info['modified_positions'].append({
                            'position': pos,
                            'original_value': float(original[pos]),
                            'modified_value': float(current[pos]),
                            'absolute_difference': float(abs_diff[pos])
                        })
                
                layer_info['weight_tensors'].append(tensor_info)
            
            diagnosis['layers_analyzed'].append(layer_name)
            diagnosis['layer_details'][layer_name] = layer_info
        
        return diagnosis

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocesar una imagen adaptándose a la entrada del modelo.

        El tamaño y el número de canales se derivan de model.input_shape, por lo
        que funciona con cualquier CNN (32x32x1, 224x224x3, etc.).
        """
        # Derivar forma de entrada esperada: (None, H, W, C)
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        # Convertir bytes a imagen PIL
        image = Image.open(io.BytesIO(image_data))

        # Modo de color según los canales del modelo
        target_mode = 'L' if channels == 1 else 'RGB'
        if image.mode != target_mode:
            image = image.convert(target_mode)

        # Redimensionar (PIL usa (ancho, alto))
        image = image.resize((width, height))

        # Convertir a numpy array
        imagen = np.array(image).astype('float32')
        if imagen.ndim == 2:  # escala de grises -> agregar canal
            imagen = np.expand_dims(imagen, axis=-1)

        # Normalización opcional (metadato del modelo)
        if self.normalize:
            imagen = imagen / 255.0

        imagen = np.expand_dims(imagen, axis=0)  # Agregar dimensión del batch
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
    
    def detect_numerical_errors(self, x: np.ndarray, context: str = "") -> Dict[str, Any]:
        """Detectar y reportar errores numéricos específicos"""
        errors = {
            "has_nan": bool(np.any(np.isnan(x))),
            "has_inf": bool(np.any(np.isinf(x))),
            "has_positive_inf": bool(np.any(np.isposinf(x))),
            "has_negative_inf": bool(np.any(np.isneginf(x))),
            "overflow_count": int(np.sum(np.isposinf(x))),
            "underflow_count": int(np.sum(np.isneginf(x))),
            "nan_count": int(np.sum(np.isnan(x))),
            "max_finite_value": float(np.max(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0,
            "min_finite_value": float(np.min(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0,
            "context": context
        }
        
        if errors["has_nan"] or errors["has_inf"]:
            print(f"🚨 ERROR NUMÉRICO DETECTADO en {context}:")
            if errors["has_positive_inf"]:
                print(f"   ➤ OVERFLOW: {errors['overflow_count']} valores +∞ (desbordamiento positivo)")
            if errors["has_negative_inf"]:
                print(f"   ➤ UNDERFLOW: {errors['underflow_count']} valores -∞ (desbordamiento negativo)")
            if errors["has_nan"]:
                print(f"   ➤ NaN: {errors['nan_count']} valores no numéricos (operación inválida)")
            if errors["max_finite_value"] is not None:
                print(f"   ➤ Valor finito máximo: {errors['max_finite_value']}")
            if errors["min_finite_value"] is not None:
                print(f"   ➤ Valor finito mínimo: {errors['min_finite_value']}")
        
        return errors

    def softmax_activation(self, x: np.ndarray) -> np.ndarray:
        """Aplicar activación Softmax con detección de errores numéricos"""
        # Detectar errores en la entrada
        input_errors = self.detect_numerical_errors(x, "entrada de softmax")
        
        # Aplicar softmax estándar
        exp_values = np.exp(x - np.max(x))
        
        # Detectar errores en exponenciales
        exp_errors = self.detect_numerical_errors(exp_values, "exponenciales de softmax")
        
        result = exp_values / np.sum(exp_values)
        
        # Detectar errores en resultado final
        result_errors = self.detect_numerical_errors(result, "resultado de softmax")
        
        return result
    
    def _is_softmax_layer(self, layer) -> bool:
        """Detectar si una capa aplica activación softmax."""
        act = getattr(layer, 'activation', None)
        return act is not None and getattr(act, '__name__', '') == 'softmax'

    def _save_layer_output(self, output: np.ndarray, layer_name: str, results: Dict[str, Any]):
        """Guardar la salida de una capa como Excel/imágenes según su dimensionalidad."""
        results["layer_outputs"][layer_name] = tuple(int(s) for s in output.shape)
        if output.ndim == 3:
            excel_file = self.save_feature_maps_to_excel(output, layer_name)
            image_files = self.save_feature_maps_as_images(output, layer_name)
            if excel_file:
                results["excel_files"].append(excel_file)
            results["image_files"].extend(image_files)
        elif output.ndim == 1:
            excel_file = self.save_feature_maps_to_excel(output, layer_name)
            if excel_file:
                results["excel_files"].append(excel_file)

    def perform_manual_inference(self, image_data: bytes) -> Dict[str, Any]:
        """
        Realizar inferencia capa por capa sobre cualquier CNN secuencial de Keras.

        Recorre model.layers calculando cada capa con Keras y deteniéndose en cada
        una para inyectar fallos en sus activaciones. Los fallos en pesos ya fueron
        aplicados sobre el modelo antes de llamar a este método.
        """
        print(f"Iniciando inferencia manual. Directorio de salida: {self.output_dir}")

        # Preprocesar imagen y preparar tensor de entrada
        imagen = self.preprocess_image(image_data)
        x = tf.convert_to_tensor(imagen, dtype=tf.float32)

        results = {
            "layer_outputs": {},
            "excel_files": [],
            "image_files": [],
            "final_prediction": {},
        }

        logits = None  # salida pre-softmax (para análisis de propagación)

        for layer in self.model.layers:
            layer_name = layer.name

            if self._is_softmax_layer(layer):
                # Calcular la capa SIN softmax para inyectar fallos en los logits,
                # conservando el flujo original (fallos antes del softmax).
                saved_activation = layer.activation
                layer.activation = tf.keras.activations.linear
                try:
                    pre_softmax = layer(x).numpy()[0]
                finally:
                    layer.activation = saved_activation

                pre_softmax = self.apply_fault_injection(pre_softmax, layer_name)
                logits = np.array(pre_softmax, dtype=np.float64)
                output = self.softmax_activation(pre_softmax)
            else:
                output = layer(x).numpy()[0]
                output = self.apply_fault_injection(output, layer_name)

            # Re-empaquetar como tensor con dimensión de batch para la capa siguiente
            x = tf.convert_to_tensor(output[np.newaxis, ...], dtype=tf.float32)

            self._save_layer_output(output, layer_name, results)

        # Salida final de la red
        final_output = np.array(x.numpy()[0])
        if logits is None:
            logits = final_output.astype(np.float64)

        # dense_output3: alias histórico para los logits pre-softmax (lo usa el
        # servicio de campañas para detectar cambios de propagación).
        results["dense_output3"] = [float(v) for v in logits]

        # Construir la predicción final (argmax + detección de NaN/Inf)
        results["final_prediction"] = self._build_prediction(final_output)

        # Filtrar archivos None de las listas
        results["excel_files"] = [f for f in results["excel_files"] if f is not None]
        results["image_files"] = [f for f in results["image_files"] if f is not None]

        # Agregar información de inyección de fallos
        results["fault_injection"] = self.get_fault_summary()
        results["session_id"] = self.session_id

        print(f"Inferencia completada. Archivos Excel: {len(results['excel_files'])}, Imágenes: {len(results['image_files'])}")
        return results

    def _build_prediction(self, output: np.ndarray) -> Dict[str, Any]:
        """Construir el diccionario de predicción final con manejo de errores numéricos."""
        softmax_errors = self.detect_numerical_errors(output, "predicción final")
        original_probabilities = [float(p) for p in output]

        try:
            predicted_class = int(np.argmax(output)) if np.any(np.isfinite(output)) else -1
            confidence = float(np.max(output[np.isfinite(output)])) if np.any(np.isfinite(output)) else 0.0
            all_probabilities = []
            for prob in output:
                if np.isfinite(prob):
                    all_probabilities.append(float(prob))
                elif np.isnan(prob):
                    all_probabilities.append("NaN")
                elif np.isposinf(prob):
                    all_probabilities.append("Infinity")
                elif np.isneginf(prob):
                    all_probabilities.append("-Infinity")
                else:
                    all_probabilities.append(0.0)
        except Exception as e:
            print(f"❌ Error al calcular predicción: {str(e)}")
            predicted_class = -1
            confidence = 0.0
            all_probabilities = [0.0] * len(output)
            original_probabilities = [0.0] * len(output)

        has_critical_errors = softmax_errors["has_nan"] or softmax_errors["has_inf"]

        if has_critical_errors:
            error_info = {
                "error_type": "numerical_overflow_underflow",
                "error_details": {
                    "overflow_detected": softmax_errors["has_positive_inf"],
                    "underflow_detected": softmax_errors["has_negative_inf"],
                    "nan_detected": softmax_errors["has_nan"],
                    "overflow_count": softmax_errors["overflow_count"],
                    "underflow_count": softmax_errors["underflow_count"],
                    "nan_count": softmax_errors["nan_count"],
                    "description": "La inyección de fallos ha causado valores numéricos fuera del rango IEEE 754",
                },
                "attempted_prediction": {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities_with_errors": softmax_errors["nan_count"] + softmax_errors["overflow_count"] + softmax_errors["underflow_count"],
                },
                "original_probabilities": original_probabilities,
            }
            return {
                "success": False,
                "error": error_info,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
            }

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "original_probabilities": original_probabilities,
        }

    def save_difference_maps(self, feature_maps: np.ndarray, layer_name: str, golden_maps: np.ndarray = None) -> List[str]:
        """
        Generar y guardar mapas de diferencias entre salidas golden y con fallos.
        
        Args:
            feature_maps: Mapas de características actuales (con fallos)
            layer_name: Nombre de la capa
            golden_maps: Mapas de características golden (sin fallos)
            
        Returns:
            Lista de rutas de archivos de diferencias generados
        """
        if golden_maps is None:
            print(f"⚠️ No hay mapas golden para comparar en {layer_name}")
            return []
            
        if feature_maps.shape != golden_maps.shape:
            print(f"⚠️ Las formas no coinciden para {layer_name}: {feature_maps.shape} vs {golden_maps.shape}")
            return []
            
        difference_files = []
        
        try:
            if len(feature_maps.shape) == 3:  # Mapas 2D con canales
                num_channels = feature_maps.shape[-1]
                
                for i in range(min(num_channels, 10)):  # Limitar a 10 canales
                    # Calcular diferencia absoluta
                    diff_map = np.abs(feature_maps[:, :, i] - golden_maps[:, :, i])
                    
                    # Verificar si hay diferencias significativas
                    max_diff = np.max(diff_map)
                    mean_diff = np.mean(diff_map)
                    
                    if max_diff > 1e-6:  # Solo guardar si hay diferencias significativas
                        # Normalizar para visualización
                        if max_diff > 0:
                            normalized_diff = (diff_map / max_diff * 255).astype(np.uint8)
                        else:
                            normalized_diff = np.zeros_like(diff_map, dtype=np.uint8)
                        
                        # Guardar imagen de diferencia
                        diff_path = os.path.join(self.output_dir, f"{layer_name}_diferencia_canal_{i+1}.png")
                        success = cv2.imwrite(diff_path, normalized_diff)
                        
                        if success:
                            difference_files.append(diff_path)
                            print(f"📊 Mapa de diferencia guardado: {diff_path}")
                            print(f"   Diferencia máxima: {max_diff:.8f}, Diferencia promedio: {mean_diff:.8f}")
                        
                        # Guardar datos numéricos de diferencia
                        excel_diff_path = os.path.join(self.output_dir, f"{layer_name}_diferencia_canal_{i+1}.xlsx")
                        df_diff = pd.DataFrame(diff_map)
                        df_diff.to_excel(excel_diff_path, index=False, header=False)
                        difference_files.append(excel_diff_path)
                    else:
                        print(f"ℹ️ No hay diferencias significativas en {layer_name} canal {i+1}")
                        
            elif len(feature_maps.shape) == 1:  # Vector 1D
                diff_vector = np.abs(feature_maps - golden_maps)
                max_diff = np.max(diff_vector)
                mean_diff = np.mean(diff_vector)
                
                if max_diff > 1e-6:
                    # Guardar diferencias como Excel
                    diff_path = os.path.join(self.output_dir, f"{layer_name}_diferencias.xlsx")
                    df_diff = pd.DataFrame({
                        'Golden': golden_maps,
                        'Con_Fallos': feature_maps,
                        'Diferencia_Absoluta': diff_vector
                    })
                    df_diff.to_excel(diff_path, index=True)
                    difference_files.append(diff_path)
                    print(f"📊 Diferencias guardadas: {diff_path}")
                    print(f"   Diferencia máxima: {max_diff:.8f}, Diferencia promedio: {mean_diff:.8f}")
                else:
                    print(f"ℹ️ No hay diferencias significativas en {layer_name}")
                    
        except Exception as e:
            print(f"❌ Error al generar mapas de diferencia para {layer_name}: {str(e)}")
            
        return difference_files