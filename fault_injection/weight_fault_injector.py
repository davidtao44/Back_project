import numpy as np
import struct
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf

class WeightFaultInjector:
    """
    Inyector de fallos específico para pesos del modelo (kernels, bias, weights).
    Permite seleccionar exactamente qué pesos modificar.
    """
    
    def __init__(self):
        self.injected_faults = []
        self.original_weights = {}  # Backup de pesos originales
        self.fault_config = {}
        
    def configure_fault(self, layer_name: str, fault_params: Dict[str, Any]):
        """
        Configurar inyección de fallos para una capa específica.
        
        Args:
            layer_name: Nombre de la capa
            fault_params: Parámetros de configuración:
                - target_type: 'kernel', 'bias', 'weights'
                - positions: Lista de posiciones específicas [(i, j, k, ...)]
                - bit_positions: Lista de posiciones de bits a afectar
                - fault_type: Tipo de fallo a inyectar:
                    * 'bitflip': Invierte el bit (0→1, 1→0) [por defecto]
                    * 'stuck_at_0': Fuerza el bit a 0
                    * 'stuck_at_1': Fuerza el bit a 1
        """
        print(f"🔧 DEBUG WeightFaultInjector: Configurando fallos para capa {layer_name}")
        print(f"🔧 DEBUG WeightFaultInjector: Parámetros: {fault_params}")
        
        self.fault_config[layer_name] = fault_params.copy()
        
    def clear_faults(self):
        """Limpiar todos los fallos configurados."""
        self.fault_config.clear()
        self.injected_faults.clear()
        
    def backup_original_weights(self, model):
        """Hacer backup de los pesos originales del modelo."""
        print("🔧 DEBUG WeightFaultInjector: Haciendo backup de pesos originales")
        
        for i, layer in enumerate(model.layers):
            layer_name = f"{layer.__class__.__name__.lower()}_{i+1}"
            
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()
                self.original_weights[layer_name] = [w.copy() for w in weights]
                print(f"✅ Backup guardado para {layer_name}: {[w.shape for w in weights]}")
                
    def restore_original_weights(self, model):
        """Restaurar los pesos originales del modelo."""
        print("🔧 DEBUG WeightFaultInjector: Restaurando pesos originales")
        
        for i, layer in enumerate(model.layers):
            layer_name = f"{layer.__class__.__name__.lower()}_{i+1}"
            
            if layer_name in self.original_weights:
                layer.set_weights(self.original_weights[layer_name])
                print(f"✅ Pesos restaurados para {layer_name}")
                
    def float32_to_bits(self, value: float) -> str:
        """Convertir float32 a representación binaria."""
        return format(struct.unpack('I', struct.pack('f', value))[0], '032b')
    
    def bits_to_float32(self, bits: str) -> float:
        """Convertir representación binaria a float32."""
        return struct.unpack('f', struct.pack('I', int(bits, 2)))[0]
    
    def inject_bitflip(self, value: float, bit_position: int) -> float:
        """Inyectar un bitflip en una posición específica."""
        if not (0 <= bit_position <= 31):
            return value
            
        bits = self.float32_to_bits(value)
        bit_list = list(bits)
        # Usar la misma convención que BitflipFaultInjector: 0=LSB, 31=MSB
        bit_index = 31 - bit_position
        bit_list[bit_index] = '1' if bit_list[bit_index] == '0' else '0'
        modified_bits = ''.join(bit_list)
        
        return self.bits_to_float32(modified_bits)
    
    def inject_stuck_at_0(self, value: float, bit_position: int) -> float:
        """Inyectar un fallo stuck-at-0 en una posición específica."""
        if not (0 <= bit_position <= 31):
            return value
            
        bits = self.float32_to_bits(value)
        bit_list = list(bits)
        # Usar la misma convención que BitflipFaultInjector: 0=LSB, 31=MSB
        bit_index = 31 - bit_position
        bit_list[bit_index] = '0'  # Forzar el bit a 0
        modified_bits = ''.join(bit_list)
        
        return self.bits_to_float32(modified_bits)
    
    def inject_stuck_at_1(self, value: float, bit_position: int) -> float:
        """Inyectar un fallo stuck-at-1 en una posición específica."""
        if not (0 <= bit_position <= 31):
            return value
            
        bits = self.float32_to_bits(value)
        bit_list = list(bits)
        # Usar la misma convención que BitflipFaultInjector: 0=LSB, 31=MSB
        bit_index = 31 - bit_position
        bit_list[bit_index] = '1'  # Forzar el bit a 1
        modified_bits = ''.join(bit_list)
        
        return self.bits_to_float32(modified_bits)
    
    def inject_faults_in_weights(self, model) -> List[Dict]:
        """
        Inyectar fallos en los pesos del modelo según la configuración.
        
        Args:
            model: Modelo de TensorFlow/Keras
            
        Returns:
            Lista de fallos inyectados
        """
        injected_faults = []
        
        for i, layer in enumerate(model.layers):
            layer_name = f"{layer.__class__.__name__.lower()}_{i+1}"
            
            if layer_name not in self.fault_config:
                continue
                
            config = self.fault_config[layer_name]
            print(f"🔧 DEBUG WeightFaultInjector: Procesando capa {layer_name}")
            print(f"🔧 DEBUG WeightFaultInjector: Configuración: {config}")
            
            if not hasattr(layer, 'get_weights') or not layer.get_weights():
                print(f"⚠️ Capa {layer_name} no tiene pesos")
                continue
                
            weights = layer.get_weights()
            target_type = config.get('target_type', 'kernel')
            positions = config.get('positions', [])
            
            # Determinar qué tensor de pesos modificar
            weight_idx = self._get_weight_index(target_type, weights)
            if weight_idx is None:
                print(f"⚠️ Tipo de peso '{target_type}' no encontrado en {layer_name}")
                continue
                
            target_weights = weights[weight_idx].copy()
            print(f"✅ Modificando {target_type} en {layer_name}, forma: {target_weights.shape}")
            
            # Inyectar fallos en posiciones específicas
            for pos_config in positions:
                # Manejar tanto el formato nuevo como el legacy
                if isinstance(pos_config, dict):
                    # Formato nuevo: {'position': [0, 0, 0, 0], 'bit_positions': [20, 21, ...]}
                    position = tuple(pos_config['position'])
                    bit_positions_for_pos = pos_config.get('bit_positions', [15])
                else:
                    # Formato legacy: posición directa
                    position = tuple(pos_config)
                    bit_positions_for_pos = config.get('bit_positions', [15])
                
                if not self._is_valid_position(target_weights.shape, position):
                    print(f"⚠️ Posición inválida {position} para forma {target_weights.shape}")
                    continue
                
                # Obtener valor original una sola vez
                original_value = target_weights[position]
                current_value = original_value
                
                # Obtener tipo de fallo (por defecto bitflip)
                fault_type = config.get('fault_type', 'bitflip')
                
                # Inyectar fallos en todos los bits especificados para esta posición
                for bit_pos in bit_positions_for_pos:
                    # Aplicar el tipo de fallo correspondiente
                    if fault_type == 'stuck_at_0':
                        current_value = self.inject_stuck_at_0(current_value, bit_pos)
                    elif fault_type == 'stuck_at_1':
                        current_value = self.inject_stuck_at_1(current_value, bit_pos)
                    else:  # bitflip por defecto
                        current_value = self.inject_bitflip(current_value, bit_pos)
                    
                    # Registrar fallo
                    fault_info = {
                        'layer_name': layer_name,
                        'target_type': target_type,
                        'position': tuple(int(x) for x in position),
                        'bit_position': int(bit_pos),
                        'original_value': float(original_value),
                        'modified_value': float(current_value),
                        'fault_type': f'weight_{fault_type}'
                    }
                    injected_faults.append(fault_info)
                
                # Aplicar el valor final modificado
                target_weights[position] = current_value
                print(f"✅ Fallo inyectado en {layer_name}.{target_type}[{position}] bits {bit_positions_for_pos}")
                print(f"   Valor: {original_value:.6f} → {current_value:.6f}")
            
            # Actualizar pesos en el modelo
            weights[weight_idx] = target_weights
            layer.set_weights(weights)
            
        self.injected_faults.extend(injected_faults)
        return injected_faults
    
    def _get_weight_index(self, target_type: str, weights: List[np.ndarray]) -> Optional[int]:
        """Determinar el índice del tensor de pesos según el tipo."""
        if target_type in ['kernel', 'weights']:
            return 0 if len(weights) > 0 else None
        elif target_type == 'bias':
            return 1 if len(weights) > 1 else None
        else:
            return None
            
    def _is_valid_position(self, shape: Tuple, position: Tuple) -> bool:
        """Verificar si una posición es válida para la forma dada."""
        if len(position) != len(shape):
            return False
        return all(0 <= pos < dim for pos, dim in zip(position, shape))
    
    def get_fault_summary(self) -> Dict[str, Any]:
        """Obtener resumen de fallos inyectados en pesos."""
        summary = {
            'total_faults': len(self.injected_faults),
            'faults_by_layer': {},
            'faults_by_type': {},
            'fault_details': self.injected_faults.copy()
        }
        
        # Agrupar por capa y tipo
        for fault in self.injected_faults:
            layer = fault['layer_name']
            target_type = fault['target_type']
            
            if layer not in summary['faults_by_layer']:
                summary['faults_by_layer'][layer] = 0
            summary['faults_by_layer'][layer] += 1
            
            if target_type not in summary['faults_by_type']:
                summary['faults_by_type'][target_type] = 0
            summary['faults_by_type'][target_type] += 1
            
        return summary