#!/usr/bin/env python3
"""
Módulo para ejecutar campañas de fallos en redes neuronales.
Permite ejecutar inferencias golden y con fallos para comparar métricas.
"""

import numpy as np
import tensorflow as tf
import os
import time
import uuid
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import json

from fault_injection.manual_inference import ManualInference
from fault_injection.weight_fault_injector import WeightFaultInjector

class FaultCampaign:
    """
    Clase para ejecutar campañas de fallos en redes neuronales.
    Permite comparar el comportamiento del modelo con y sin fallos.
    """
    
    def __init__(self, model_path: str, image_dir: Optional[str] = None, session_id: Optional[str] = None):
        """
        Inicializar campaña de fallos.
        
        Args:
            model_path: Ruta al modelo de TensorFlow
            image_dir: Directorio con imágenes de prueba (opcional, usa MNIST si no se especifica)
            session_id: ID de sesión único
        """
        self.model_path = model_path
        self.image_dir = image_dir
        self.session_id = session_id or f"campaign_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Cargar modelo
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo cargado: {model_path}")
        
        # Cargar dataset
        self.images, self.labels = self._load_dataset()
        print(f"📊 Dataset cargado: {len(self.images)} imágenes")
        
        # Inicializar servicios
        self.manual_inference = ManualInference(model_instance=self.model)
        self.weight_fault_injector = WeightFaultInjector()
        
        # Resultados
        self.results = {
            'golden': {'predictions': [], 'labels': []},
            'fault': {'predictions': [], 'labels': []},
            'metrics': {}
        }
    
    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cargar dataset de imágenes.
        
        Returns:
            Tuple con imágenes y etiquetas
        """
        if self.image_dir and os.path.exists(self.image_dir):
            return self._load_images_from_directory()
        else:
            return self._load_mnist_dataset()
    
    def _load_images_from_directory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cargar imágenes desde directorio organizado por carpetas de dígitos.
        Estructura esperada: image_dir/0/, image_dir/1/, ..., image_dir/9/
        
        Returns:
            Tuple con imágenes y etiquetas
        """
        images = []
        labels = []
        
        # Buscar carpetas de dígitos (0-9)
        for digit_folder in os.listdir(self.image_dir):
            digit_path = os.path.join(self.image_dir, digit_folder)
            
            # Verificar que sea una carpeta y que el nombre sea un dígito
            if os.path.isdir(digit_path) and digit_folder.isdigit():
                label = int(digit_folder)
                print(f"📁 Cargando imágenes del dígito {label}...")
                
                # Buscar archivos de imagen en la carpeta del dígito
                for filename in os.listdir(digit_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            image_path = os.path.join(digit_path, filename)
                            
                            # Cargar y procesar imagen
                            from PIL import Image
                            img = Image.open(image_path).convert('L')  # Convertir a escala de grises
                            img = img.resize((28, 28))  # Redimensionar a 28x28
                            img_array = np.array(img) / 255.0  # Normalizar
                            
                            images.append(img_array)
                            labels.append(label)
                        except Exception as e:
                            print(f"⚠️ No se pudo procesar archivo: {filename} - {str(e)}")
                            continue
        
        if not images:
            print("⚠️ No se encontraron imágenes válidas, usando MNIST")
            return self._load_mnist_dataset()
        
        # Convertir a arrays numpy
        images_array = np.array(images)
        labels_array = np.array(labels)
        
        # Expandir dimensiones si es necesario (agregar canal)
        if len(images_array.shape) == 3:
            images_array = np.expand_dims(images_array, axis=-1)
        
        print(f"✅ Cargadas {len(images_array)} imágenes locales con forma: {images_array.shape}")
        return images_array, labels_array
    
    def _load_mnist_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cargar dataset MNIST.
        
        Returns:
            Tuple con imágenes y etiquetas de prueba de MNIST
        """
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalizar imágenes
        x_test = x_test.astype('float32') / 255.0
        
        # Expandir dimensiones si es necesario
        if len(x_test.shape) == 3:
            x_test = np.expand_dims(x_test, axis=-1)
        
        return x_test, y_test
    
    def _convert_image_to_bytes(self, image: np.ndarray) -> bytes:
        """
        Convertir imagen numpy a bytes para inferencia manual.
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Imagen convertida a bytes
        """
        from PIL import Image as PILImage
        import io
        
        # Asegurar que la imagen tenga la forma correcta
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Convertir numpy array a imagen PIL
        image_pil = PILImage.fromarray((image.squeeze() * 255).astype(np.uint8), mode='L')
        
        # Convertir a bytes
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def _perform_inference_on_samples(self, indices: np.ndarray, description: str) -> Tuple[List[int], List[int]]:
        """
        Realizar inferencia en un conjunto de muestras.
        
        Args:
            indices: Índices de las muestras a procesar
            description: Descripción para logging
            
        Returns:
            Tuple con predicciones y etiquetas
        """
        predictions = []
        labels = []
        
        # Verificar estado de los pesos al inicio de la inferencia
        if "con pesos modificados" in description:
            # Obtener algunos pesos para verificar que están modificados
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    weights = layer.kernel.numpy()
                    print(f"🔍 DEBUG: Pesos de {layer.name} durante inferencia con fallos - muestra: {weights.flat[:3]}")
                    break  # Solo mostrar la primera capa con pesos
        
        for i, idx in enumerate(indices):
            image = self.images[idx]
            label = self.labels[idx]
            
            # Convertir imagen a bytes
            image_bytes = self._convert_image_to_bytes(image)
            
            # Realizar inferencia manual
            result = self.manual_inference.perform_manual_inference(image_bytes)
            predicted_class = result['final_prediction']['predicted_class']
            
            predictions.append(predicted_class)
            labels.append(label)
            
            # Log detallado para las primeras 3 muestras
            if i < 3:
                print(f"  🔍 DEBUG {description} - Muestra {i+1}: Predicción={predicted_class}, Etiqueta={label}, Índice={idx}")
            
            if (i + 1) % 5 == 0:  # Reducir frecuencia para ver mejor los logs
                print(f"  Procesadas {i + 1}/{len(indices)} muestras {description} - Última predicción: {predicted_class}, Etiqueta real: {label}")
        
        return predictions, labels
    
    def _create_campaign_results(self, golden_predictions: List[int], golden_labels: List[int], 
                               fault_predictions: List[int], fault_labels: List[int],
                               num_samples: int, execution_time: float, config: Dict[str, Any],
                               config_key: str) -> Dict[str, Any]:
        """
        Crear estructura de resultados para una campaña.
        
        Args:
            golden_predictions: Predicciones golden
            golden_labels: Etiquetas golden
            fault_predictions: Predicciones con fallos
            fault_labels: Etiquetas con fallos
            num_samples: Número de muestras procesadas
            execution_time: Tiempo de ejecución
            config: Configuración de fallos
            config_key: Clave para la configuración en los resultados
            
        Returns:
            Diccionario con resultados estructurados
        """
        # Calcular métricas
        print(f"🔍 DEBUG: Calculando métricas golden con {len(golden_labels)} etiquetas y {len(golden_predictions)} predicciones")
        golden_metrics = self.calculate_metrics(golden_labels, golden_predictions)
        print(f"🔍 DEBUG: Métricas golden calculadas: {golden_metrics}")
        
        print(f"🔍 DEBUG: Calculando métricas con fallos con {len(fault_labels)} etiquetas y {len(fault_predictions)} predicciones")
        fault_metrics = self.calculate_metrics(fault_labels, fault_predictions)
        print(f"🔍 DEBUG: Métricas con fallos calculadas: {fault_metrics}")
        
        # Comparar resultados
        comparison = self._compare_predictions(golden_predictions, fault_predictions)
        print(f"🔍 DEBUG: Comparación calculada: {comparison}")
        
        results = {
            'golden_results': {
                'predictions': golden_predictions,
                'labels': golden_labels,
                'metrics': golden_metrics
            },
            'fault_results': {
                'predictions': fault_predictions,
                'labels': fault_labels,
                'metrics': fault_metrics
            },
            'comparison': comparison,
            'campaign_info': {
                'session_id': self.session_id,
                'model_path': self.model_path,
                'num_samples': num_samples,
                'execution_time_seconds': execution_time,
                config_key: config
            }
        }
        
        print(f"🔍 DEBUG: Estructura final de resultados:")
        print(f"  - Golden metrics keys: {list(golden_metrics.keys()) if golden_metrics else 'None'}")
        print(f"  - Fault metrics keys: {list(fault_metrics.keys()) if fault_metrics else 'None'}")
        print(f"  - Comparison keys: {list(comparison.keys()) if comparison else 'None'}")
        
        return results

    def run_golden_inference(self, num_samples: int) -> Tuple[List[int], List[int]]:
        """
        Ejecutar inferencia golden (sin fallos).
        
        Args:
            num_samples: Número de muestras a procesar
            
        Returns:
            Tuple con predicciones y etiquetas reales
        """
        print(f"🏆 Iniciando inferencia golden con {num_samples} muestras...")
        
        # Seleccionar muestras aleatorias y guardarlas para reutilizar
        self.selected_indices = np.random.choice(len(self.images), size=min(num_samples, len(self.images)), replace=False)
        print(f"🔍 DEBUG: Índices seleccionados para golden: {self.selected_indices[:5]}...")  # Mostrar primeros 5
        
        # Realizar inferencia en las muestras
        predictions, labels = self._perform_inference_on_samples(self.selected_indices, "golden")
        
        self.results['golden']['predictions'] = predictions
        self.results['golden']['labels'] = labels
        
        print(f"✅ Inferencia golden completada: {len(predictions)} predicciones")
        return predictions, labels
    
    def run_fault_inference(self, num_samples: int, fault_config: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """
        Ejecutar inferencia con fallos en activaciones.
        
        Args:
            num_samples: Número de muestras a procesar
            fault_config: Configuración de fallos
            
        Returns:
            Tuple con predicciones y etiquetas reales
        """
        print(f"⚡ Iniciando inferencia con fallos en activaciones...")
        
        # Configurar inyección de fallos
        self.manual_inference.configure_fault_injection(fault_config)
        
        predictions = []
        labels = []
        
        # Seleccionar las mismas muestras que en golden
        indices = np.random.choice(len(self.images), size=min(num_samples, len(self.images)), replace=False)
        
        for i, idx in enumerate(indices):
            image = self.images[idx]
            label = self.labels[idx]
            
            # Realizar predicción con fallos
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            result = self.manual_inference.perform_inference_with_faults(image)
            predicted_class = result['predicted_class']
            
            predictions.append(predicted_class)
            labels.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i + 1}/{len(indices)} muestras con fallos")
        
        self.results['fault']['predictions'] = predictions
        self.results['fault']['labels'] = labels
        
        print(f"✅ Inferencia con fallos completada: {len(predictions)} predicciones")
        return predictions, labels
    
    def run_weight_fault_campaign(self, num_samples: int, weight_fault_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar campaña de fallos en pesos.
        
        Args:
            num_samples: Número de muestras a procesar
            weight_fault_config: Configuración de fallos en pesos
            
        Returns:
            Diccionario con resultados de la campaña
        """
        print(f"🎯 Iniciando campaña de fallos en pesos...")
        start_time = time.time()
        
        # 1. Ejecutar inferencia golden
        golden_predictions, golden_labels = self.run_golden_inference(num_samples)
        
        # 2. Hacer backup de pesos originales
        print("💾 Haciendo backup de pesos originales...")
        self.weight_fault_injector.backup_original_weights(self.model)
        
        # 3. Configurar fallos en pesos
        print("🔧 Configurando inyección de fallos en pesos...")
        for layer_name, layer_config in weight_fault_config.get('layers', {}).items():
            self.weight_fault_injector.configure_fault(layer_name, layer_config)
        
        # 4. Aplicar fallos en pesos
        print("⚡ Aplicando fallos en pesos del modelo...")
        injected_weight_faults = self.weight_fault_injector.inject_faults_in_weights(self.model)
        print(f"✅ Inyectados {len(injected_weight_faults)} fallos en pesos")
        
        # 5. Ejecutar inferencia con pesos modificados
        print("🔍 Ejecutando inferencia con pesos modificados...")
        # Usar exactamente las mismas muestras que en golden
        if not hasattr(self, 'selected_indices'):
            raise ValueError("❌ ERROR: No se han seleccionado índices en la inferencia golden")
        print(f"🔍 DEBUG: Usando los mismos índices que golden: {self.selected_indices[:5]}...")  # Mostrar primeros 5
        fault_predictions, fault_labels = self._perform_inference_on_samples(self.selected_indices, "con pesos modificados")
        
        # 6. Restaurar pesos originales
        print("🔄 Restaurando pesos originales...")
        self.weight_fault_injector.restore_original_weights(self.model)
        
        # 6.5. Comparar predicciones antes de calcular métricas
        print("🔍 DEBUG: Comparando predicciones individuales:")
        print(f"🔍 DEBUG: Golden predictions: {golden_predictions}")
        print(f"🔍 DEBUG: Fault predictions:  {fault_predictions}")
        differences = [i for i in range(len(golden_predictions)) if golden_predictions[i] != fault_predictions[i]]
        print(f"🔍 DEBUG: Diferencias en índices: {differences}")
        print(f"🔍 DEBUG: Total de diferencias: {len(differences)}/{len(golden_predictions)}")
        
        # 7. Crear resultados usando función auxiliar
        execution_time = time.time() - start_time
        results = self._create_campaign_results(
            golden_predictions, golden_labels,
            fault_predictions, fault_labels,
            num_samples, execution_time,
            weight_fault_config, 'weight_fault_config'
        )
        
        print(f"✅ Campaña de fallos en pesos completada en {execution_time:.2f} segundos")
        return results
    
    def run_campaign(self, num_samples: int, fault_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar campaña completa de fallos en activaciones.
        
        Args:
            num_samples: Número de muestras a procesar
            fault_config: Configuración de fallos
            
        Returns:
            Diccionario con resultados de la campaña
        """
        print(f"🎯 Iniciando campaña de fallos en activaciones...")
        start_time = time.time()
        
        # 1. Ejecutar inferencia golden
        golden_predictions, golden_labels = self.run_golden_inference(num_samples)
        
        # 2. Ejecutar inferencia con fallos
        fault_predictions, fault_labels = self.run_fault_inference(num_samples, fault_config)
        
        # 3. Crear resultados usando función auxiliar
        execution_time = time.time() - start_time
        results = self._create_campaign_results(
            golden_predictions, golden_labels,
            fault_predictions, fault_labels,
            num_samples, execution_time,
            fault_config, 'fault_config'
        )
        
        print(f"✅ Campaña de fallos completada en {execution_time:.2f} segundos")
        return results
    
    def calculate_metrics(self, true_labels: List[int], predictions: List[int]) -> Dict[str, Any]:
        """
        Calcular métricas de evaluación.
        
        Args:
            true_labels: Etiquetas reales
            predictions: Predicciones del modelo
            
        Returns:
            Diccionario con métricas calculadas
        """
        # Convertir a arrays numpy
        y_true = np.array(true_labels)
        y_pred = np.array(predictions)
        
        # Calcular métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calcular precision (macro average para multiclase)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Reporte de clasificación
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'num_samples': len(true_labels),
            'correct_predictions': int(np.sum(y_true == y_pred)),
            'incorrect_predictions': int(np.sum(y_true != y_pred))
        }
        
        return metrics
    
    def _compare_predictions(self, golden_predictions: List[int], fault_predictions: List[int]) -> Dict[str, Any]:
        """
        Comparar predicciones golden vs con fallos.
        
        Args:
            golden_predictions: Predicciones sin fallos
            fault_predictions: Predicciones con fallos
            
        Returns:
            Diccionario con comparación
        """
        golden_array = np.array(golden_predictions)
        fault_array = np.array(fault_predictions)
        
        # Calcular diferencias
        same_predictions = np.sum(golden_array == fault_array)
        different_predictions = np.sum(golden_array != fault_array)
        
        # Encontrar muestras donde las predicciones difieren
        different_indices = np.where(golden_array != fault_array)[0].tolist()
        
        comparison = {
            'samples_with_same_predictions': int(same_predictions),
            'samples_with_different_predictions': int(different_predictions),
            'percentage_different': float(different_predictions / len(golden_predictions) * 100),
            'different_prediction_indices': different_indices
        }
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], output_file: str = None) -> str:
        """
        Guardar resultados en archivo JSON.
        
        Args:
            results: Resultados de la campaña
            output_file: Archivo de salida (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        if output_file is None:
            output_file = f"fault_campaign_results_{self.session_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Resultados guardados en: {output_file}")
        return output_file