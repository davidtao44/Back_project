import os
import tempfile
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .vhdl_weight_modifier import VHDLWeightModifier
from .vivado_controller import VivadoController
from .csv_processor import CSVProcessor

logger = logging.getLogger(__name__)

class HardwareFaultInjector:
    """
    Clase principal que combina la inyección de fallos con la modificación de archivos VHDL
    y la ejecución de simulaciones en Vivado.
    """
    
    def __init__(self, vivado_path: str = "vivado"):
        """
        Inicializa el inyector de fallos para hardware.
        
        Args:
            vivado_path: Ruta al ejecutable de Vivado
        """
        self.vhdl_modifier = VHDLWeightModifier()
        self.vivado_controller = VivadoController(vivado_path)
        self.csv_processor = CSVProcessor()
        self.temp_dir = None
        
    def create_temp_workspace(self) -> str:
        """
        Crea un directorio temporal para trabajar con archivos modificados.
        
        Returns:
            Ruta del directorio temporal
        """
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="vhdl_fault_injection_")
            logger.info(f"Directorio temporal creado: {self.temp_dir}")
        return self.temp_dir
    
    def cleanup_temp_workspace(self) -> None:
        """
        Limpia el directorio temporal.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Directorio temporal limpiado: {self.temp_dir}")
            self.temp_dir = None
    
    def validate_fault_config(self, fault_config: Dict) -> Tuple[bool, str]:
        """
        Valida la configuración de fallos.
        
        Args:
            fault_config: Configuración de fallos
        
        Returns:
            Tuple[bool, str]: (es_válido, mensaje_error)
        """
        required_fields = ['fault_type']
        valid_fault_types = ['bitflip', 'stuck_at_0', 'stuck_at_1']
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in fault_config:
                return False, f"Campo requerido faltante: {field}"
        
        # Verificar tipo de fallo válido
        if fault_config['fault_type'] not in valid_fault_types:
            return False, f"Tipo de fallo inválido: {fault_config['fault_type']}. Válidos: {valid_fault_types}"
        
        # Verificar configuración específica para filtros
        if 'filter_name' in fault_config:
            filter_required = ['row', 'col', 'bit_position']
            for field in filter_required:
                if field not in fault_config:
                    return False, f"Campo requerido para filtro faltante: {field}"
            
            # Validar rangos
            if not (0 <= fault_config['row'] <= 4):
                return False, "row debe estar entre 0 y 4"
            if not (0 <= fault_config['col'] <= 4):
                return False, "col debe estar entre 0 y 4"
            if not (0 <= fault_config['bit_position'] <= 7):
                return False, "bit_position debe estar entre 0 y 7 para filtros"
        
        # Verificar configuración específica para sesgos
        elif 'bias_name' in fault_config:
            if 'bit_position' not in fault_config:
                return False, "Campo requerido para sesgo faltante: bit_position"
            
            # Validar rango para sesgos (16 bits)
            if not (0 <= fault_config['bit_position'] <= 15):
                return False, "bit_position debe estar entre 0 y 15 para sesgos"
        
        else:
            return False, "Debe especificar 'filter_name' o 'bias_name'"
        
        return True, "Configuración válida"
    
    def inject_faults_and_simulate(self, 
                                  vhdl_file_path: str,
                                  fault_configs: List[Dict],
                                  simulation_config: Dict = None,
                                  keep_modified_files: bool = False) -> Dict:
        """
        Función principal que inyecta fallos en un archivo VHDL y ejecuta simulación.
        
        Args:
            vhdl_file_path: Ruta al archivo VHDL original
            fault_configs: Lista de configuraciones de fallos
            simulation_config: Configuración de simulación (opcional)
            keep_modified_files: Si mantener archivos modificados después de la simulación
        
        Returns:
            Diccionario con resultados completos del proceso
        """
        try:
            # Crear workspace temporal
            temp_dir = self.create_temp_workspace()
            
            # Validar configuraciones de fallos
            validation_errors = []
            for i, config in enumerate(fault_configs):
                is_valid, error_msg = self.validate_fault_config(config)
                if not is_valid:
                    validation_errors.append(f"Config {i}: {error_msg}")
            
            if validation_errors:
                return {
                    'success': False,
                    'error': 'Errores de validación',
                    'validation_errors': validation_errors
                }
            
            # Generar nombre para archivo modificado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name = os.path.basename(vhdl_file_path)
            name_without_ext = os.path.splitext(original_name)[0]
            modified_file_path = os.path.join(temp_dir, f"{name_without_ext}_faulted_{timestamp}.vhd")
            
            # Inyectar fallos en el archivo VHDL
            logger.info(f"Inyectando fallos en {vhdl_file_path}")
            modification_result = self.vhdl_modifier.modify_vhdl_weights(
                vhdl_file_path, 
                modified_file_path, 
                fault_configs
            )
            
            if not modification_result['success']:
                return {
                    'success': False,
                    'error': 'Error modificando archivo VHDL',
                    'modification_result': modification_result
                }
            
            # Configuración por defecto de simulación
            default_sim_config = {
                'simulation_time': '1000ns',
                'create_project': True,
                'project_name': f'fault_injection_project_{timestamp}',
                'part': 'xc7z020clg484-1'
            }
            
            if simulation_config:
                default_sim_config.update(simulation_config)
            
            simulation_result = None
            project_path = None
            
            # Ejecutar simulación si está configurada
            if default_sim_config.get('run_simulation', True):
                logger.info("Ejecutando simulación en Vivado")
                
                if default_sim_config['create_project']:
                    # Crear proyecto temporal
                    project_result = self.vivado_controller.create_project_from_vhdl(
                        default_sim_config['project_name'],
                        temp_dir,
                        [modified_file_path],
                        default_sim_config['part']
                    )
                    
                    if not project_result['project_creation_success']:
                        return {
                            'success': False,
                            'error': 'Error creando proyecto Vivado',
                            'project_result': project_result,
                            'modification_result': modification_result
                        }
                    
                    project_path = project_result['project_path']
                
                # Ejecutar simulación
                simulation_result = self.vivado_controller.run_simulation(
                    project_path,
                    simulation_time=default_sim_config['simulation_time']
                )
                
                # Obtener resultados de simulación
                if simulation_result['simulation_success']:
                    results_info = self.vivado_controller.get_simulation_results(project_path)
                    simulation_result.update(results_info)
            
            # Preparar resultado final
            result = {
                'success': True,
                'timestamp': timestamp,
                'original_file': vhdl_file_path,
                'modified_file': modified_file_path,
                'temp_directory': temp_dir,
                'modification_result': modification_result,
                'simulation_config': default_sim_config
            }
            
            if simulation_result:
                result['simulation_result'] = simulation_result
                result['simulation_success'] = simulation_result.get('simulation_success', False)
            
            if project_path:
                result['project_path'] = project_path
            
            # Manejar archivos modificados
            if keep_modified_files:
                # Copiar archivo modificado a ubicación permanente
                permanent_dir = os.path.dirname(vhdl_file_path)
                permanent_file = os.path.join(permanent_dir, f"{name_without_ext}_faulted_{timestamp}.vhd")
                
                import shutil
                shutil.copy2(modified_file_path, permanent_file)
                result['permanent_modified_file'] = permanent_file
                logger.info(f"Archivo modificado guardado permanentemente: {permanent_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en inyección de fallos y simulación: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        finally:
            # Limpiar workspace temporal si no se mantienen archivos
            if not keep_modified_files:
                self.cleanup_temp_workspace()
    
    def batch_fault_injection(self, 
                             vhdl_file_path: str,
                             fault_scenarios: List[List[Dict]],
                             simulation_config: Dict = None) -> List[Dict]:
        """
        Ejecuta múltiples escenarios de inyección de fallos en lote.
        
        Args:
            vhdl_file_path: Ruta al archivo VHDL original
            fault_scenarios: Lista de escenarios, cada uno con su lista de configuraciones de fallos
            simulation_config: Configuración de simulación
        
        Returns:
            Lista de resultados para cada escenario
        """
        results = []
        
        for i, fault_configs in enumerate(fault_scenarios):
            logger.info(f"Ejecutando escenario de fallos {i+1}/{len(fault_scenarios)}")
            
            result = self.inject_faults_and_simulate(
                vhdl_file_path,
                fault_configs,
                simulation_config,
                keep_modified_files=False
            )
            
            result['scenario_index'] = i
            result['scenario_description'] = f"Escenario {i+1}"
            results.append(result)
        
        return results
    
    def generate_fault_report(self, results: List[Dict]) -> Dict:
        """
        Genera un reporte consolidado de los resultados de inyección de fallos.
        
        Args:
            results: Lista de resultados de inyección de fallos
        
        Returns:
            Diccionario con reporte consolidado
        """
        try:
            total_scenarios = len(results)
            successful_modifications = sum(1 for r in results if r.get('success', False))
            successful_simulations = sum(1 for r in results if r.get('simulation_success', False))
            
            # Estadísticas de fallos aplicados
            total_faults = 0
            fault_types_count = {'bitflip': 0, 'stuck_at_0': 0, 'stuck_at_1': 0}
            target_types_count = {'filter': 0, 'bias': 0}
            
            for result in results:
                if result.get('success') and 'modification_result' in result:
                    applied_faults = result['modification_result'].get('applied_faults', [])
                    total_faults += len(applied_faults)
                    
                    for fault in applied_faults:
                        fault_type = fault.get('fault_type', 'unknown')
                        target_type = fault.get('type', 'unknown')
                        
                        if fault_type in fault_types_count:
                            fault_types_count[fault_type] += 1
                        if target_type in target_types_count:
                            target_types_count[target_type] += 1
            
            # Errores comunes
            error_summary = {}
            for result in results:
                if not result.get('success'):
                    error = result.get('error', 'Error desconocido')
                    error_summary[error] = error_summary.get(error, 0) + 1
            
            report = {
                'summary': {
                    'total_scenarios': total_scenarios,
                    'successful_modifications': successful_modifications,
                    'successful_simulations': successful_simulations,
                    'modification_success_rate': successful_modifications / total_scenarios if total_scenarios > 0 else 0,
                    'simulation_success_rate': successful_simulations / total_scenarios if total_scenarios > 0 else 0
                },
                'fault_statistics': {
                    'total_faults_applied': total_faults,
                    'fault_types_distribution': fault_types_count,
                    'target_types_distribution': target_types_count
                },
                'error_summary': error_summary,
                'detailed_results': results,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def process_simulation_csv(self, csv_path: str) -> Dict:
        """
        Procesa un archivo CSV de simulación y organiza los datos en matrices 28x28.
        
        Args:
            csv_path: Ruta al archivo CSV de simulación
            
        Returns:
            Diccionario con matrices organizadas y estadísticas
        """
        try:
            logger.info(f"Procesando CSV de simulación: {csv_path}")
            result = self.csv_processor.process_simulation_csv(csv_path)
            return result
            
        except Exception as e:
            logger.error(f"Error procesando CSV de simulación: {e}")
            return {
                'status': 'error',
                'message': f'Error procesando CSV: {str(e)}',
                'matrices': {},
                'statistics': {},
                'metadata': {}
            }
    
    def inject_faults_and_simulate_with_csv_processing(self, 
                                                      vhdl_file_path: str,
                                                      fault_configs: List[Dict],
                                                      simulation_config: Dict = None,
                                                      keep_modified_files: bool = False,
                                                      process_csv: bool = True) -> Dict:
        """
        Función principal que inyecta fallos, ejecuta simulación y procesa CSV resultante.
        
        Args:
            vhdl_file_path: Ruta al archivo VHDL original
            fault_configs: Lista de configuraciones de fallos
            simulation_config: Configuración de simulación (opcional)
            keep_modified_files: Si mantener archivos modificados después de la simulación
            process_csv: Si procesar el CSV resultante en matrices 28x28
        
        Returns:
            Diccionario con resultados completos incluyendo matrices procesadas
        """
        try:
            # Ejecutar inyección de fallos y simulación normal
            result = self.inject_faults_and_simulate(
                vhdl_file_path, 
                fault_configs, 
                simulation_config, 
                keep_modified_files
            )
            
            # Si la simulación fue exitosa y se solicita procesamiento CSV
            if (result.get('success') and 
                result.get('simulation_success') and 
                process_csv):
                
                # Buscar archivos CSV en los resultados de simulación
                simulation_result = result.get('simulation_result', {})
                csv_files = []
                
                # Buscar archivos CSV en diferentes ubicaciones posibles
                if 'output_files' in simulation_result:
                    for file_path in simulation_result['output_files']:
                        if file_path.endswith('.csv'):
                            csv_files.append(file_path)
                
                # Si no se encontraron en output_files, buscar en directorio de proyecto
                if not csv_files and 'project_path' in result:
                    project_path = result['project_path']
                    # Buscar archivos CSV en el directorio de simulación
                    sim_dir = os.path.join(project_path, "*.sim", "sim_1", "behav", "xsim")
                    if os.path.exists(sim_dir):
                        import glob
                        csv_pattern = os.path.join(sim_dir, "*.csv")
                        csv_files = glob.glob(csv_pattern)
                
                # Procesar cada archivo CSV encontrado
                csv_results = {}
                for csv_file in csv_files:
                    if os.path.exists(csv_file):
                        csv_name = os.path.basename(csv_file)
                        logger.info(f"Procesando CSV: {csv_name}")
                        csv_result = self.process_simulation_csv(csv_file)
                        csv_results[csv_name] = csv_result
                
                # Agregar resultados CSV al resultado principal
                if csv_results:
                    result['csv_processing'] = {
                        'processed_files': list(csv_results.keys()),
                        'results': csv_results,
                        'processing_success': all(
                            r.get('status') == 'success' for r in csv_results.values()
                        )
                    }
                    logger.info(f"CSV procesados exitosamente: {len(csv_results)} archivos")
                else:
                    result['csv_processing'] = {
                        'processed_files': [],
                        'results': {},
                        'processing_success': False,
                        'message': 'No se encontraron archivos CSV para procesar'
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error en inyección de fallos con procesamiento CSV: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }