import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import struct

class BitflipFaultInjector:
    """
    Clase para inyectar fallos de tipo bitflip en activaciones de redes neuronales.
    Soporta inyección en diferentes capas y posiciones específicas.
    """
    
    def __init__(self):
        self.fault_config = {}
        self.injected_faults = []
        
    def configure_fault(self, layer_name: str, fault_params: Dict[str, Any]):
        """
        Configurar parámetros de inyección de fallos para una capa específica.
        
        Args:
            layer_name: Nombre de la capa (ej: 'conv2d_1', 'dense_1')
            fault_params: Diccionario con parámetros de fallo:
                - enabled: bool - Si está habilitada la inyección
                - num_faults: int - Número de fallos a inyectar
                - fault_type: str - Tipo de fallo ('random', 'specific')
                - positions: List[Tuple] - Posiciones específicas (opcional)
                - bit_positions: List[int] - Posiciones de bits específicas (opcional)
        """
        self.fault_config[layer_name] = fault_params
        
    def clear_faults(self):
        """Limpiar configuración de fallos y registro de fallos inyectados."""
        self.fault_config.clear()
        self.injected_faults.clear()
        
    def float32_to_bits(self, value: float) -> str:
        """Convertir un float32 a su representación binaria."""
        packed = struct.pack('!f', value)
        return ''.join(format(byte, '08b') for byte in packed)
    
    def bits_to_float32(self, bits: str) -> float:
        """Convertir representación binaria a float32."""
        bytes_data = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
        return struct.unpack('!f', bytes_data)[0]
    
    def inject_bitflip(self, value: float, bit_position: int) -> float:
        """
        Inyectar un bitflip en una posición específica de un valor float32.
        
        Args:
            value: Valor original
            bit_position: Posición del bit a voltear (0-31, donde 0 es LSB)
            
        Returns:
            Valor con el bit volteado
        """
        if not (0 <= bit_position <= 31):
            raise ValueError("La posición del bit debe estar entre 0 y 31")
            
        # Convertir a bits
        bits = self.float32_to_bits(value)
        
        # Voltear el bit en la posición especificada
        bit_list = list(bits)
        bit_index = 31 - bit_position  # Invertir porque el MSB está a la izquierda
        bit_list[bit_index] = '1' if bit_list[bit_index] == '0' else '0'
        
        # Convertir de vuelta a float
        modified_bits = ''.join(bit_list)
        return self.bits_to_float32(modified_bits)
    
    def inject_faults_in_activations(self, activations: np.ndarray, layer_name: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Inyectar fallos en las activaciones de una capa específica.
        
        Args:
            activations: Array de activaciones de la capa
            layer_name: Nombre de la capa
            
        Returns:
            Tuple con (activaciones_modificadas, lista_de_fallos_inyectados)
        """
        if layer_name not in self.fault_config:
            return activations.copy(), []
            
        config = self.fault_config[layer_name]
        if not config.get('enabled', False):
            return activations.copy(), []
            
        # Crear copia para modificar
        modified_activations = activations.copy()
        injected_faults = []
        
        num_faults = config.get('num_faults', 1)
        fault_type = config.get('fault_type', 'random')
        
        if fault_type == 'random':
            # Inyección aleatoria
            for _ in range(num_faults):
                fault_info = self._inject_random_fault(modified_activations, layer_name)
                if fault_info:
                    injected_faults.append(fault_info)
                    
        elif fault_type == 'specific':
            # Inyección en posiciones específicas
            positions = config.get('positions', [])
            bit_positions = config.get('bit_positions', [15])  # Bit por defecto
            
            for pos_idx, position in enumerate(positions[:num_faults]):
                bit_pos = bit_positions[pos_idx % len(bit_positions)]
                fault_info = self._inject_specific_fault(
                    modified_activations, layer_name, position, bit_pos
                )
                if fault_info:
                    injected_faults.append(fault_info)
        
        # Registrar fallos inyectados
        self.injected_faults.extend(injected_faults)
        
        return modified_activations, injected_faults
    
    def _inject_random_fault(self, activations: np.ndarray, layer_name: str) -> Optional[Dict]:
        """Inyectar un fallo aleatorio en las activaciones."""
        try:
            # Obtener forma de las activaciones
            shape = activations.shape
            
            # Generar posición aleatoria
            if len(shape) == 1:  # Vector 1D (capas densas)
                pos = (random.randint(0, shape[0] - 1),)
            elif len(shape) == 3:  # 3D (mapas de características)
                pos = (
                    random.randint(0, shape[0] - 1),
                    random.randint(0, shape[1] - 1),
                    random.randint(0, shape[2] - 1)
                )
            else:
                return None
                
            # Generar posición de bit aleatoria con sesgo hacia bits más significativos
            # 70% probabilidad de bits significativos (15-30), 30% de bits menos significativos (0-14)
            if random.random() < 0.7:
                bit_position = random.randint(15, 30)  # Bits más significativos
            else:
                bit_position = random.randint(0, 14)   # Bits menos significativos
            
            # Obtener valor original
            original_value = activations[pos]
            
            # Inyectar bitflip
            modified_value = self.inject_bitflip(original_value, bit_position)
            
            # Aplicar modificación
            activations[pos] = modified_value
            
            return {
                'layer_name': layer_name,
                'position': pos,
                'bit_position': bit_position,
                'original_value': float(original_value),
                'modified_value': float(modified_value),
                'fault_type': 'random'
            }
            
        except Exception as e:
            print(f"Error al inyectar fallo aleatorio en {layer_name}: {str(e)}")
            return None
    
    def _inject_specific_fault(self, activations: np.ndarray, layer_name: str, 
                              position: Tuple, bit_position: int) -> Optional[Dict]:
        """Inyectar un fallo en una posición específica."""
        try:
            # Verificar que la posición sea válida
            if not self._is_valid_position(activations.shape, position):
                return None
                
            # Obtener valor original
            original_value = activations[position]
            
            # Inyectar bitflip
            modified_value = self.inject_bitflip(original_value, bit_position)
            
            # Aplicar modificación
            activations[position] = modified_value
            
            return {
                'layer_name': layer_name,
                'position': position,
                'bit_position': bit_position,
                'original_value': float(original_value),
                'modified_value': float(modified_value),
                'fault_type': 'specific'
            }
            
        except Exception as e:
            print(f"Error al inyectar fallo específico en {layer_name}: {str(e)}")
            return None
    
    def _is_valid_position(self, shape: Tuple, position: Tuple) -> bool:
        """Verificar si una posición es válida para la forma dada."""
        if len(position) != len(shape):
            return False
            
        for i, (pos, dim) in enumerate(zip(position, shape)):
            if not (0 <= pos < dim):
                return False
                
        return True
    
    def get_fault_summary(self) -> Dict[str, Any]:
        """Obtener resumen de fallos inyectados."""
        summary = {
            'total_faults': len(self.injected_faults),
            'faults_by_layer': {},
            'fault_details': self.injected_faults.copy()
        }
        
        # Agrupar por capa
        for fault in self.injected_faults:
            layer = fault['layer_name']
            if layer not in summary['faults_by_layer']:
                summary['faults_by_layer'][layer] = 0
            summary['faults_by_layer'][layer] += 1
            
        return summary
    
    def export_fault_report(self, output_dir: str) -> str:
        """Exportar reporte de fallos a archivo JSON."""
        import json
        import os
        
        summary = self.get_fault_summary()
        
        report_file = os.path.join(output_dir, "fault_injection_report.json")
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return report_file