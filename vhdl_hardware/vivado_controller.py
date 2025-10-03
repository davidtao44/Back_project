import subprocess
import os
import logging
import tempfile
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class VivadoController:
    """
    Controlador para ejecutar comandos de Vivado TCL y manejar simulaciones.
    """
    
    def __init__(self, vivado_path: str = "vivado"):
        """
        Inicializa el controlador de Vivado.
        
        Args:
            vivado_path: Ruta al ejecutable de Vivado (por defecto asume que está en PATH)
        """
        self.vivado_path = vivado_path
        self.verify_vivado_installation()
    
    def verify_vivado_installation(self) -> bool:
        """
        Verifica que Vivado esté instalado y accesible.
        """
        try:
            result = subprocess.run(
                [self.vivado_path, "-version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"Vivado encontrado: {result.stdout.split()[0:3]}")
                return True
            else:
                logger.error(f"Error verificando Vivado: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout verificando instalación de Vivado")
            return False
        except FileNotFoundError:
            logger.error(f"Vivado no encontrado en la ruta: {self.vivado_path}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado verificando Vivado: {e}")
            return False
    
    def create_tcl_script(self, commands: List[str], script_path: str) -> None:
        """
        Crea un script TCL con los comandos especificados.
        
        Args:
            commands: Lista de comandos TCL
            script_path: Ruta donde guardar el script
        """
        try:
            with open(script_path, 'w') as f:
                for command in commands:
                    f.write(f"{command}\n")
            logger.info(f"Script TCL creado en: {script_path}")
        except Exception as e:
            logger.error(f"Error creando script TCL: {e}")
            raise
    
    def run_tcl_script(self, script_path: str, working_dir: str = None, timeout: int = 300) -> Dict:
        """
        Ejecuta un script TCL en Vivado.
        
        Args:
            script_path: Ruta al script TCL
            working_dir: Directorio de trabajo (opcional)
            timeout: Timeout en segundos
        
        Returns:
            Diccionario con el resultado de la ejecución
        """
        try:
            # Preparar comando
            cmd = [self.vivado_path, "-mode", "batch", "-source", script_path]
            
            # Cambiar directorio de trabajo si se especifica
            original_dir = None
            if working_dir:
                original_dir = os.getcwd()
                os.chdir(working_dir)
                logger.info(f"Cambiando directorio de trabajo a: {working_dir}")
            
            # Ejecutar comando
            logger.info(f"Ejecutando: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Restaurar directorio original
            if original_dir:
                os.chdir(original_dir)
            
            # Procesar resultado
            success = result.returncode == 0
            
            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout ejecutando script TCL: {script_path}")
            return {
                'success': False,
                'error': 'Timeout',
                'execution_time': timeout
            }
        except Exception as e:
            logger.error(f"Error ejecutando script TCL: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Asegurar que se restaure el directorio original
            if original_dir and os.getcwd() != original_dir:
                os.chdir(original_dir)
    
    def run_simulation(self, project_path: str, testbench_file: str = None, 
                      simulation_time: str = "1000ns") -> Dict:
        """
        Ejecuta una simulación en Vivado.
        
        Args:
            project_path: Ruta al proyecto de Vivado
            testbench_file: Archivo de testbench (opcional)
            simulation_time: Tiempo de simulación
        
        Returns:
            Diccionario con el resultado de la simulación
        """
        try:
            # Crear script TCL para simulación
            tcl_commands = [
                f"open_project {project_path}",
                "update_compile_order -fileset sources_1",
                "launch_simulation",
                f"run {simulation_time}",
                "close_sim",
                "close_project"
            ]
            
            # Si se especifica un testbench, agregarlo
            if testbench_file:
                tcl_commands.insert(2, f"set_property top {testbench_file} [get_filesets sim_1]")
            
            # Crear archivo temporal para el script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tcl', delete=False) as f:
                script_path = f.name
                for command in tcl_commands:
                    f.write(f"{command}\n")
            
            # Ejecutar simulación
            result = self.run_tcl_script(script_path, timeout=600)  # 10 minutos timeout
            
            # Limpiar archivo temporal
            os.unlink(script_path)
            
            return {
                'simulation_success': result['success'],
                'simulation_time': simulation_time,
                'project_path': project_path,
                **result
            }
            
        except Exception as e:
            logger.error(f"Error ejecutando simulación: {e}")
            return {
                'simulation_success': False,
                'error': str(e)
            }
    
    def synthesize_design(self, project_path: str, top_module: str = None) -> Dict:
        """
        Sintetiza un diseño en Vivado.
        
        Args:
            project_path: Ruta al proyecto de Vivado
            top_module: Módulo top (opcional)
        
        Returns:
            Diccionario con el resultado de la síntesis
        """
        try:
            # Crear script TCL para síntesis
            tcl_commands = [
                f"open_project {project_path}",
                "update_compile_order -fileset sources_1"
            ]
            
            # Si se especifica un módulo top, configurarlo
            if top_module:
                tcl_commands.append(f"set_property top {top_module} [get_filesets sources_1]")
            
            tcl_commands.extend([
                "launch_runs synth_1 -jobs 4",
                "wait_on_run synth_1",
                "close_project"
            ])
            
            # Crear archivo temporal para el script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tcl', delete=False) as f:
                script_path = f.name
                for command in tcl_commands:
                    f.write(f"{command}\n")
            
            # Ejecutar síntesis
            result = self.run_tcl_script(script_path, timeout=1800)  # 30 minutos timeout
            
            # Limpiar archivo temporal
            os.unlink(script_path)
            
            return {
                'synthesis_success': result['success'],
                'project_path': project_path,
                **result
            }
            
        except Exception as e:
            logger.error(f"Error ejecutando síntesis: {e}")
            return {
                'synthesis_success': False,
                'error': str(e)
            }
    
    def create_project_from_vhdl(self, project_name: str, project_dir: str, 
                                vhdl_files: List[str], part: str = "xc7z020clg484-1") -> Dict:
        """
        Crea un nuevo proyecto de Vivado a partir de archivos VHDL.
        
        Args:
            project_name: Nombre del proyecto
            project_dir: Directorio del proyecto
            vhdl_files: Lista de archivos VHDL
            part: Parte FPGA objetivo
        
        Returns:
            Diccionario con el resultado de la creación del proyecto
        """
        try:
            project_path = os.path.join(project_dir, f"{project_name}.xpr")
            
            # Crear script TCL para crear proyecto
            tcl_commands = [
                f"create_project {project_name} {project_dir} -part {part}",
            ]
            
            # Agregar archivos VHDL
            for vhdl_file in vhdl_files:
                tcl_commands.append(f"add_files -norecurse {vhdl_file}")
            
            tcl_commands.extend([
                "update_compile_order -fileset sources_1",
                "close_project"
            ])
            
            # Crear archivo temporal para el script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tcl', delete=False) as f:
                script_path = f.name
                for command in tcl_commands:
                    f.write(f"{command}\n")
            
            # Ejecutar creación de proyecto
            result = self.run_tcl_script(script_path, timeout=300)
            
            # Limpiar archivo temporal
            os.unlink(script_path)
            
            return {
                'project_creation_success': result['success'],
                'project_path': project_path,
                'project_name': project_name,
                **result
            }
            
        except Exception as e:
            logger.error(f"Error creando proyecto: {e}")
            return {
                'project_creation_success': False,
                'error': str(e)
            }
    
    def get_simulation_results(self, project_path: str, results_dir: str = None) -> Dict:
        """
        Obtiene los resultados de la simulación.
        
        Args:
            project_path: Ruta al proyecto
            results_dir: Directorio de resultados (opcional)
        
        Returns:
            Diccionario con información de los resultados
        """
        try:
            if not results_dir:
                # Buscar directorio de resultados por defecto
                project_dir = os.path.dirname(project_path)
                sim_dir = os.path.join(project_dir, "*.sim", "sim_1", "behav", "xsim")
                
            # Buscar archivos de resultados comunes
            result_files = []
            log_files = []
            
            if os.path.exists(sim_dir):
                for root, dirs, files in os.walk(sim_dir):
                    for file in files:
                        if file.endswith(('.wdb', '.vcd', '.log')):
                            full_path = os.path.join(root, file)
                            if file.endswith('.log'):
                                log_files.append(full_path)
                            else:
                                result_files.append(full_path)
            
            return {
                'results_found': len(result_files) > 0,
                'result_files': result_files,
                'log_files': log_files,
                'simulation_dir': sim_dir
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados de simulación: {e}")
            return {
                'results_found': False,
                'error': str(e)
            }