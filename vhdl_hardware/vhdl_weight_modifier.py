import re
import os
import struct
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VHDLWeightModifier:
    """
    Clase para modificar pesos y sesgos en archivos VHDL de la primera capa convolucional.
    Maneja los filtros FMAP_1 a FMAP_6 y los sesgos BIAS_VAL_1 a BIAS_VAL_6.
    """
    
    def __init__(self):
        self.filter_pattern = r'constant\s+(FMAP_\d+):\s+FILTER_TYPE:=\s*\(\s*(.*?)\s*\);'
        self.bias_pattern = r'constant\s+(BIAS_VAL_\d+):\s+signed\s*\([^)]+\)\s*:=\s*"([01]+)";'
        
    def read_vhdl_file(self, file_path: str) -> str:
        """Lee el contenido del archivo VHDL."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error leyendo archivo VHDL: {e}")
            raise
    
    def write_vhdl_file(self, file_path: str, content: str) -> None:
        """Escribe el contenido modificado al archivo VHDL."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            logger.info(f"Archivo VHDL modificado guardado en: {file_path}")
        except Exception as e:
            logger.error(f"Error escribiendo archivo VHDL: {e}")
            raise
    
    def parse_filter_matrix(self, filter_content: str) -> List[List[str]]:
        """
        Parsea el contenido de un filtro FMAP y retorna una matriz de strings binarios.
        
        Ejemplo de entrada:
        ("11101010","11110001","11101101","11110111","00001111"), 
        ("00000101","00000111","00001100","00000110","00001001"), 
        ...
        """
        rows = []
        # Buscar todas las filas entre paréntesis
        row_pattern = r'\(\s*([^)]+)\s*\)'
        matches = re.findall(row_pattern, filter_content)
        
        for match in matches:
            # Extraer los valores binarios de cada fila
            binary_values = re.findall(r'"([01]+)"', match)
            if binary_values:
                rows.append(binary_values)
        
        return rows
    
    def format_filter_matrix(self, matrix: List[List[str]], filter_name: str) -> str:
        """
        Formatea una matriz de valores binarios de vuelta al formato VHDL.
        """
        lines = [f"constant {filter_name}: FILTER_TYPE:= ("]
        
        for i, row in enumerate(matrix):
            row_str = "     (" + ",".join([f'"{val}"' for val in row]) + ")"
            if i < len(matrix) - 1:
                row_str += ","
            lines.append(row_str)
        
        lines.append(" );")
        return "\n".join(lines)
    
    def extract_filters_and_biases(self, vhdl_content: str) -> Tuple[Dict, Dict]:
        """
        Extrae todos los filtros FMAP y sesgos BIAS_VAL del contenido VHDL.
        
        Returns:
            Tuple[Dict, Dict]: (filtros, sesgos)
        """
        filters = {}
        biases = {}
        
        # Extraer filtros FMAP
        filter_matches = re.finditer(self.filter_pattern, vhdl_content, re.DOTALL)
        for match in filter_matches:
            filter_name = match.group(1)
            filter_content = match.group(2)
            matrix = self.parse_filter_matrix(filter_content)
            filters[filter_name] = matrix
            logger.info(f"Extraído filtro {filter_name} con matriz {len(matrix)}x{len(matrix[0]) if matrix else 0}")
        
        # Extraer sesgos BIAS_VAL
        bias_matches = re.finditer(self.bias_pattern, vhdl_content)
        for match in bias_matches:
            bias_name = match.group(1)
            bias_value = match.group(2)
            biases[bias_name] = bias_value
            logger.info(f"Extraído sesgo {bias_name}: {bias_value}")
        
        return filters, biases
    
    def inject_bit_fault(self, binary_str: str, bit_position: int, fault_type: str) -> str:
        """
        Inyecta un fallo en una posición específica de un string binario.
        
        Args:
            binary_str: String binario (ej: "11101010")
            bit_position: Posición del bit (0 = LSB, 7 = MSB para 8 bits)
            fault_type: Tipo de fallo ('bitflip', 'stuck_at_0', 'stuck_at_1')
        
        Returns:
            String binario modificado
        """
        if bit_position >= len(binary_str) or bit_position < 0:
            logger.warning(f"Posición de bit {bit_position} fuera de rango para {binary_str}")
            return binary_str
        
        # Convertir a lista para modificar
        bits = list(binary_str)
        
        # Aplicar el fallo según el tipo
        if fault_type == 'bitflip':
            bits[-(bit_position + 1)] = '0' if bits[-(bit_position + 1)] == '1' else '1'
        elif fault_type == 'stuck_at_0':
            bits[-(bit_position + 1)] = '0'
        elif fault_type == 'stuck_at_1':
            bits[-(bit_position + 1)] = '1'
        else:
            logger.warning(f"Tipo de fallo desconocido: {fault_type}")
            return binary_str
        
        return ''.join(bits)
    
    def inject_faults_in_filters(self, filters: Dict, fault_config: Dict) -> Dict:
        """
        Inyecta fallos en los filtros según la configuración especificada.
        
        Args:
            filters: Diccionario de filtros extraídos
            fault_config: Configuración de fallos
                {
                    'filter_name': 'FMAP_1',
                    'row': 0,
                    'col': 0,
                    'bit_position': 7,
                    'fault_type': 'stuck_at_0'
                }
        
        Returns:
            Diccionario de filtros modificados
        """
        modified_filters = filters.copy()
        
        filter_name = fault_config.get('filter_name')
        row = fault_config.get('row')
        col = fault_config.get('col')
        bit_position = fault_config.get('bit_position')
        fault_type = fault_config.get('fault_type')
        
        if filter_name not in modified_filters:
            logger.error(f"Filtro {filter_name} no encontrado")
            return modified_filters
        
        if row >= len(modified_filters[filter_name]) or col >= len(modified_filters[filter_name][row]):
            logger.error(f"Posición [{row}][{col}] fuera de rango para {filter_name}")
            return modified_filters
        
        # Obtener el valor original
        original_value = modified_filters[filter_name][row][col]
        
        # Inyectar el fallo
        modified_value = self.inject_bit_fault(original_value, bit_position, fault_type)
        
        # Actualizar el filtro
        modified_filters[filter_name][row][col] = modified_value
        
        logger.info(f"Fallo inyectado en {filter_name}[{row}][{col}] bit {bit_position}: {original_value} -> {modified_value}")
        
        return modified_filters
    
    def inject_faults_in_biases(self, biases: Dict, fault_config: Dict) -> Dict:
        """
        Inyecta fallos en los sesgos según la configuración especificada.
        
        Args:
            biases: Diccionario de sesgos extraídos
            fault_config: Configuración de fallos
                {
                    'bias_name': 'BIAS_VAL_1',
                    'bit_position': 15,
                    'fault_type': 'stuck_at_0'
                }
        
        Returns:
            Diccionario de sesgos modificados
        """
        modified_biases = biases.copy()
        
        bias_name = fault_config.get('bias_name')
        bit_position = fault_config.get('bit_position')
        fault_type = fault_config.get('fault_type')
        
        if bias_name not in modified_biases:
            logger.error(f"Sesgo {bias_name} no encontrado")
            return modified_biases
        
        # Obtener el valor original
        original_value = modified_biases[bias_name]
        
        # Inyectar el fallo
        modified_value = self.inject_bit_fault(original_value, bit_position, fault_type)
        
        # Actualizar el sesgo
        modified_biases[bias_name] = modified_value
        
        logger.info(f"Fallo inyectado en {bias_name} bit {bit_position}: {original_value} -> {modified_value}")
        
        return modified_biases
    
    def replace_filters_in_vhdl(self, vhdl_content: str, modified_filters: Dict) -> str:
        """
        Reemplaza los filtros en el contenido VHDL con las versiones modificadas.
        """
        modified_content = vhdl_content
        
        for filter_name, matrix in modified_filters.items():
            # Crear el nuevo contenido del filtro
            new_filter_content = self.format_filter_matrix(matrix, filter_name)
            
            # Buscar y reemplazar el filtro original
            pattern = rf'constant\s+{filter_name}:\s+FILTER_TYPE:=\s*\([^;]*\);'
            modified_content = re.sub(pattern, new_filter_content, modified_content, flags=re.DOTALL)
        
        return modified_content
    
    def replace_biases_in_vhdl(self, vhdl_content: str, modified_biases: Dict) -> str:
        """
        Reemplaza los sesgos en el contenido VHDL con las versiones modificadas.
        """
        modified_content = vhdl_content
        
        for bias_name, bias_value in modified_biases.items():
            # Crear el nuevo contenido del sesgo
            new_bias_content = f'constant {bias_name}: signed (BIASES_SIZE-1 downto 0) := "{bias_value}";'
            
            # Buscar y reemplazar el sesgo original
            pattern = rf'constant\s+{bias_name}:\s+signed\s*\([^)]+\)\s*:=\s*"[01]+";'
            modified_content = re.sub(pattern, new_bias_content, modified_content)
        
        return modified_content
    
    def modify_vhdl_weights(self, input_file: str, output_file: str, fault_configs: List[Dict]) -> Dict:
        """
        Función principal para modificar pesos en archivo VHDL.
        
        Args:
            input_file: Ruta del archivo VHDL original
            output_file: Ruta del archivo VHDL modificado
            fault_configs: Lista de configuraciones de fallos
        
        Returns:
            Diccionario con información del proceso
        """
        try:
            # Leer archivo original
            vhdl_content = self.read_vhdl_file(input_file)
            
            # Extraer filtros y sesgos
            filters, biases = self.extract_filters_and_biases(vhdl_content)
            
            # Aplicar fallos
            modified_filters = filters.copy()
            modified_biases = biases.copy()
            
            applied_faults = []
            
            for fault_config in fault_configs:
                if 'filter_name' in fault_config:
                    # Fallo en filtro
                    modified_filters = self.inject_faults_in_filters(modified_filters, fault_config)
                    applied_faults.append({
                        'type': 'filter',
                        'target': fault_config['filter_name'],
                        'position': f"[{fault_config['row']}][{fault_config['col']}]",
                        'bit': fault_config['bit_position'],
                        'fault_type': fault_config['fault_type']
                    })
                elif 'bias_name' in fault_config:
                    # Fallo en sesgo
                    modified_biases = self.inject_faults_in_biases(modified_biases, fault_config)
                    applied_faults.append({
                        'type': 'bias',
                        'target': fault_config['bias_name'],
                        'bit': fault_config['bit_position'],
                        'fault_type': fault_config['fault_type']
                    })
            
            # Reemplazar en el contenido VHDL
            modified_content = self.replace_filters_in_vhdl(vhdl_content, modified_filters)
            modified_content = self.replace_biases_in_vhdl(modified_content, modified_biases)
            
            # Guardar archivo modificado
            self.write_vhdl_file(output_file, modified_content)
            
            return {
                'success': True,
                'input_file': input_file,
                'output_file': output_file,
                'applied_faults': applied_faults,
                'filters_found': len(filters),
                'biases_found': len(biases)
            }
            
        except Exception as e:
            logger.error(f"Error modificando archivo VHDL: {e}")
            return {
                'success': False,
                'error': str(e)
            }