import csv
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class CSVProcessor:
    """
    Clase para procesar archivos CSV de simulación de Vivado y reorganizar
    los datos en matrices 28x28 por canal de la primera capa convolucional.
    """
    
    def __init__(self):
        """Inicializa el procesador CSV."""
        self.output_size = 28  # Tamaño de salida 28x28
        self.num_channels = 6  # 6 canales de salida
        self.start_row = 142   # Fila donde comienzan las salidas (basado en 1)
        
    def read_csv_file(self, csv_path: str) -> List[List[str]]:
        """
        Lee el archivo CSV y retorna los datos como lista de listas.
        
        Args:
            csv_path: Ruta al archivo CSV
            
        Returns:
            Lista de listas con los datos del CSV
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            Exception: Si hay error al leer el archivo
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")
            
        try:
            data = []
            with open(csv_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    data.append(row)
            
            logger.info(f"CSV leído exitosamente: {len(data)} filas")
            return data
            
        except Exception as e:
            logger.error(f"Error al leer CSV {csv_path}: {str(e)}")
            raise Exception(f"Error al leer archivo CSV: {str(e)}")
    
    def extract_channel_data(self, csv_data: List[List[str]]) -> Dict[int, List[float]]:
        """
        Extrae los datos de cada canal desde el CSV.
        
        Args:
            csv_data: Datos del CSV como lista de listas
            
        Returns:
            Diccionario con los datos de cada canal {canal_id: [valores]}
        """
        channels_data = {i: [] for i in range(self.num_channels)}
        
        # Comenzar desde la fila start_row (ajustar por índice 0)
        start_idx = self.start_row - 1
        
        for row_idx in range(start_idx, len(csv_data)):
            row = csv_data[row_idx]
            
            # Verificar que la fila tenga suficientes columnas
            if len(row) < 7:  # Columna 0 (ciclo) + 6 canales
                continue
                
            # Extraer valores de cada canal (columnas 1-6)
            for channel_idx in range(self.num_channels):
                try:
                    value = float(row[channel_idx + 1])  # +1 porque columna 0 es el ciclo
                    channels_data[channel_idx].append(value)
                except (ValueError, IndexError):
                    # Si hay error, agregar 0
                    channels_data[channel_idx].append(0.0)
        
        logger.info(f"Datos extraídos por canal: {[len(channels_data[i]) for i in range(self.num_channels)]}")
        return channels_data
    
    def reshape_to_matrices(self, channels_data: Dict[int, List[float]]) -> Dict[int, np.ndarray]:
        """
        Reorganiza los datos de cada canal en matrices 28x28.
        Considera que después de cada fila de 28 datos hay 4 ciclos de reloj adicionales que deben saltarse.
        
        Args:
            channels_data: Datos de cada canal
            
        Returns:
            Diccionario con matrices 28x28 por canal {canal_id: matriz_28x28}
        """
        matrices = {}
        
        for channel_idx in range(self.num_channels):
            channel_data = channels_data[channel_idx]
            matrix = np.zeros((self.output_size, self.output_size))
            
            data_idx = 0  # Índice en los datos originales
            
            # Construir matriz fila por fila
            for row in range(self.output_size):
                for col in range(self.output_size):
                    if data_idx < len(channel_data):
                        matrix[row, col] = channel_data[data_idx]
                        data_idx += 1
                    else:
                        matrix[row, col] = 0.0
                
                # Después de completar cada fila de 28 datos, saltar 4 ciclos de reloj
                # (excepto en la última fila para evitar saltar datos innecesarios)
                if row < self.output_size - 1:
                    data_idx += 4  # Saltar 4 ciclos de reloj
            
            matrices[channel_idx] = matrix
            
            logger.info(f"Canal {channel_idx}: Matriz {matrix.shape} creada con salto de 4 ciclos por fila")
            logger.debug(f"Canal {channel_idx}: Datos utilizados hasta índice {data_idx} de {len(channel_data)} disponibles")
        
        return matrices
    
    def matrices_to_dict(self, matrices: Dict[int, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Convierte las matrices numpy a diccionarios serializables.
        
        Args:
            matrices: Diccionario con matrices numpy
            
        Returns:
            Diccionario con matrices como listas de listas
        """
        result = {}
        
        for channel_idx, matrix in matrices.items():
            # Convertir a lista de listas para serialización JSON
            matrix_list = matrix.tolist()
            result[f"channel_{channel_idx}"] = matrix_list
            
        return result
    
    def get_channel_statistics(self, matrices: Dict[int, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calcula estadísticas básicas para cada canal.
        
        Args:
            matrices: Diccionario con matrices por canal
            
        Returns:
            Diccionario con estadísticas por canal
        """
        stats = {}
        
        for channel_idx, matrix in matrices.items():
            channel_stats = {
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
                "mean": float(np.mean(matrix)),
                "std": float(np.std(matrix)),
                "sum": float(np.sum(matrix)),
                "non_zero_count": int(np.count_nonzero(matrix))
            }
            stats[f"channel_{channel_idx}"] = channel_stats
            
        return stats
    
    def process_simulation_csv(self, csv_path: str) -> Dict:
        """
        Procesa completamente un archivo CSV de simulación.
        
        Args:
            csv_path: Ruta al archivo CSV
            
        Returns:
            Diccionario con matrices organizadas y estadísticas
        """
        try:
            # Leer CSV
            csv_data = self.read_csv_file(csv_path)
            
            # Extraer datos por canal
            channels_data = self.extract_channel_data(csv_data)
            
            # Crear matrices 28x28
            matrices = self.reshape_to_matrices(channels_data)
            
            # Convertir a formato serializable
            matrices_dict = self.matrices_to_dict(matrices)
            
            # Calcular estadísticas
            statistics = self.get_channel_statistics(matrices)
            
            result = {
                "status": "success",
                "message": "CSV procesado exitosamente",
                "matrices": matrices_dict,
                "statistics": statistics,
                "metadata": {
                    "output_size": f"{self.output_size}x{self.output_size}",
                    "num_channels": self.num_channels,
                    "start_row": self.start_row,
                    "csv_file": os.path.basename(csv_path)
                }
            }
            
            logger.info(f"CSV procesado exitosamente: {csv_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando CSV {csv_path}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error procesando CSV: {str(e)}",
                "matrices": {},
                "statistics": {},
                "metadata": {}
            }