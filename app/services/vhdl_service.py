import base64
import io
import json
import os
import subprocess
import time
import uuid

import keras
import numpy as np
import tensorflow as tf
from fastapi import HTTPException
from PIL import Image

from app.core.config import MODEL_WEIGHTS_OUTPUTS_DIR, VHDL_FILE_PATH, VHDL_OUTPUTS_DIR, VHDL_SIM_DIR
from app.schemas.cnn import ModelWeightsRequest
from app.schemas.vhdl import ImageToVHDLRequest
from app.utils.json_utils import sanitize_for_json
from app.utils.vhdl_utils import (
    generate_vhdl_code,
    inject_golden_values_to_vhdl,
    modify_vhdl_weights_and_bias,
    to_binary_c2,
)
from vhdl_hardware.csv_processor import CSVProcessor
from vhdl_hardware.vivado_controller import VivadoController


def convert_image_to_vhdl(request: ImageToVHDLRequest, current_user: dict):
    """Convierte una imagen base64 a código VHDL tipo Memoria_Imagen."""
    try:
        image_data = request.image_data
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((request.width, request.height))
        img = img.convert("L")

        pixel_matrix = np.array(img)

        decimal_matrix = pixel_matrix.tolist()
        hex_matrix = []
        vhdl_matrix = []

        for row in decimal_matrix:
            hex_row = []
            vhdl_row = []
            for pixel in row:
                hex_value = format(pixel, "02x")
                hex_row.append(hex_value)
                vhdl_row.append(f'x"{hex_value}"')
            hex_matrix.append(hex_row)
            vhdl_matrix.append(vhdl_row)

        vhdl_code = generate_vhdl_code(vhdl_matrix, request.width, request.height)

        session_id = f"user_{current_user.get('uid', 'anonymous')}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        session_dir = os.path.join(VHDL_OUTPUTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        output_filename = f"Memoria_Imagen_{session_id}.vhdl.txt"
        output_path = os.path.join(session_dir, output_filename)
        with open(output_path, "w") as f:
            f.write(vhdl_code)

        return {
            "success": True,
            "decimal_matrix": decimal_matrix,
            "hex_matrix": hex_matrix,
            "vhdl_code": vhdl_code,
            "file_path": output_filename,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _extract_filters(model, output_dir: str, bits_value: int = 8):
    """Extrae los filtros y sesgos de cada capa convolucional a archivos VHDL .txt."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filter_files = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            filters, biases = layer.get_weights()
            num_filters = filters.shape[-1]
            num_channels = filters.shape[-2]

            layer_filename = os.path.join(output_dir, f"layer_{i + 1}_filters.txt")
            filter_files.append(os.path.basename(layer_filename))

            with open(layer_filename, "w") as file:
                file.write(f"Filtros de la capa convolucional {i + 1}:\n")
                for j in range(num_filters):
                    for c in range(num_channels):
                        filter_matrix = filters[:, :, c, j]
                        file.write(f"constant FMAP_{c + 1}_{j + 1}: FILTER_TYPE:= (\n")
                        for k in range(filter_matrix.shape[0]):
                            row = [
                                '"' + to_binary_c2(int(filter_matrix[k, l]), bits=bits_value) + '"'
                                for l in range(filter_matrix.shape[1])
                            ]
                            file.write(
                                "    (" + ",".join(row) + ")"
                                + ("," if k < filter_matrix.shape[0] - 1 else "") + "\n"
                            )
                        file.write(");\n\n")

                file.write(f"Sesgos de la capa convolucional {i + 1}:\n")
                for idx, bias in enumerate(biases):
                    bias_bin = to_binary_c2(int(bias), bits=bits_value)
                    file.write(
                        f'constant BIAS_VAL_{idx + 1}: signed (BIASES_SIZE-1 downto 0) := "{bias_bin}";\n'
                    )
                file.write("\n")

    return filter_files


def _extract_dense_layers(model, output_dir: str, bits_value: int = 8):
    """Extrae pesos y sesgos de capas densas a archivos VHDL .txt."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dense_files = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            pesos = layer.get_weights()[0]
            biases = layer.get_weights()[1]

            num_filas, num_columnas = pesos.shape

            pesos_filename = os.path.join(output_dir, f"{layer.name}_pesos.txt")
            dense_files.append(os.path.basename(pesos_filename))

            with open(pesos_filename, "w") as f_pesos:
                for j in range(num_columnas):
                    for i in range(num_filas):
                        bin_value = to_binary_c2(int(pesos[i, j]), bits=bits_value)
                        f_pesos.write(
                            f'constant FMAP_{j + 1}_{i + 1}: signed(WEIGHT_SIZE-1 downto 0) := "{bin_value}";\n'
                        )

            biases_filename = os.path.join(output_dir, f"{layer.name}_biases.txt")
            dense_files.append(os.path.basename(biases_filename))

            with open(biases_filename, "w") as f_biases:
                for i, bias in enumerate(biases, start=1):
                    bin_value = to_binary_c2(int(bias), bits=bits_value)
                    f_biases.write(
                        f'constant BIAS_VAL_{i}: signed(BIASES_SIZE-1 downto 0) := "{bin_value}";\n'
                    )

    return dense_files


def extract_model_weights(request: ModelWeightsRequest, current_user: dict):
    """Extrae pesos y sesgos del modelo en formato VHDL."""
    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Modelo no encontrado: {os.path.basename(request.model_path)}",
            )

        session_id = f"user_{current_user.get('uid', 'anonymous')}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        session_dir = os.path.join(MODEL_WEIGHTS_OUTPUTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        output_dir = session_dir
        model = keras.models.load_model(request.model_path)
        bits_value = request.bits_value

        conv_files = _extract_filters(model, output_dir, bits_value)
        dense_files = _extract_dense_layers(model, output_dir, bits_value)

        generated_files = conv_files + dense_files
        relative_files = [os.path.basename(file_path) for file_path in generated_files]

        return {
            "success": True,
            "message": f"Pesos y sesgos extraídos exitosamente con {bits_value} bits",
            "model": os.path.basename(request.model_path),
            "output_dir": os.path.basename(output_dir),
            "files": relative_files,
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_supported_faults():
    """Devuelve los tipos de fallos soportados por la primera capa convolucional."""
    return {
        "filter_targets": [
            {"name": f"FMAP_{i}", "description": f"FMAP_{i} - Primera capa convolucional"}
            for i in range(1, 7)
        ],
        "bias_targets": [
            {"name": f"BIAS_VAL_{i}", "description": f"BIAS_VAL_{i} - Primera capa convolucional"}
            for i in range(1, 7)
        ],
        "fault_types": [
            {"name": "stuck_at_0", "description": "Forzar bit a valor 0"},
            {"name": "stuck_at_1", "description": "Forzar bit a valor 1"},
        ],
    }


def validate_vivado(vivado_path: str = None):
    """Valida si Vivado está instalado y accesible."""
    try:
        vivado_controller = VivadoController(vivado_path or "vivado")
        is_valid = vivado_controller.verify_vivado_installation()

        return {
            "status": "success",
            "vivado_valid": is_valid,
            "vivado_path": vivado_path or "default",
            "message": "Validación de Vivado completada" if is_valid else "Vivado no encontrado o no válido",
        }

    except Exception as e:
        return {
            "status": "error",
            "vivado_valid": False,
            "vivado_path": vivado_path or "default",
            "message": f"Error al validar Vivado: {str(e)}",
        }


def get_vhdl_file_status(file_path: str):
    """Devuelve el contenido y estadísticas de un archivo VHDL."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {file_path}")

    file_stats = os.stat(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "status": "success",
        "file_info": {
            "path": file_path,
            "size": file_stats.st_size,
            "last_modified": file_stats.st_mtime,
            "last_modified_ms": int(file_stats.st_mtime * 1000),
        },
        "content": content,
        "timestamp": int(time.time() * 1000),
    }


def _run_simulation_pipeline(base_dir: str):
    """Ejecuta compile.sh → elaborate.sh → xsim y procesa CSVs. Devuelve (simulation_results, csv_results)."""
    compile_script = os.path.join(base_dir, "compile.sh")
    elaborate_script = os.path.join(base_dir, "elaborate.sh")
    simulation_script = os.path.join(base_dir, "simulate.sh")

    simulation_results = {}
    csv_processing_results = {}

    if not all(os.path.exists(script) for script in [compile_script, elaborate_script, simulation_script]):
        return (
            {
                "status": "error",
                "message": f"Scripts de simulación no encontrados en: {base_dir}",
                "missing_scripts": {
                    "compile.sh": not os.path.exists(compile_script),
                    "elaborate.sh": not os.path.exists(elaborate_script),
                    "simulate.sh": not os.path.exists(simulation_script),
                },
            },
            {},
        )

    try:
        print("🔧 Ejecutando compile.sh...")
        compile_result = subprocess.run(
            ["bash", compile_script], cwd=base_dir, capture_output=True, text=True, timeout=300
        )

        if compile_result.returncode != 0:
            return (
                {
                    "status": "error",
                    "message": f"Compilación falló con código {compile_result.returncode}",
                    "step": "compile",
                    "steps_completed": [],
                    "output": compile_result.stdout[-1000:] if compile_result.stdout else "",
                    "errors": compile_result.stderr[-500:] if compile_result.stderr else "",
                },
                {},
            )

        print("✅ compile.sh ejecutado exitosamente")
        print("🔧 Ejecutando elaborate.sh...")
        elaborate_result = subprocess.run(
            ["bash", elaborate_script], cwd=base_dir, capture_output=True, text=True, timeout=300
        )

        if elaborate_result.returncode != 0:
            return (
                {
                    "status": "error",
                    "message": f"Elaboración falló con código {elaborate_result.returncode}",
                    "step": "elaborate",
                    "steps_completed": ["compile"],
                    "output": elaborate_result.stdout[-1000:] if elaborate_result.stdout else "",
                    "errors": elaborate_result.stderr[-500:] if elaborate_result.stderr else "",
                },
                {},
            )

        print("✅ elaborate.sh ejecutado exitosamente")
        print("🔧 Ejecutando simulación no-gráfica...")

        subprocess.run(
            ["pkill", "-f", "xsim.*CONV1_SAB_STUCKAT_DEC_RAM_TB_behav"],
            capture_output=True,
        )

        env = os.environ.copy()
        env.pop("DISPLAY", None)

        simulate_result = subprocess.run(
            [
                "xsim",
                "CONV1_SAB_STUCKAT_DEC_RAM_TB_behav",
                "-tclbatch",
                "CONV1_SAB_STUCKAT_DEC_RAM_TB_batch.tcl",
                "-log",
                "simulation.log",
            ],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        if simulate_result.returncode != 0:
            return (
                {
                    "status": "error",
                    "message": f"Simulación falló con código {simulate_result.returncode}",
                    "step": "simulate",
                    "steps_completed": ["compile", "elaborate"],
                    "output": simulate_result.stdout[-1000:] if simulate_result.stdout else "",
                    "errors": simulate_result.stderr[-500:] if simulate_result.stderr else "",
                },
                {},
            )

        print("✅ Simulación ejecutada exitosamente")
        simulation_results = {
            "status": "success",
            "message": "Secuencia completa de simulación completada exitosamente",
            "steps_completed": ["compile", "elaborate", "simulate"],
            "output": simulate_result.stdout[-1000:] if simulate_result.stdout else "",
            "errors": simulate_result.stderr[-500:] if simulate_result.stderr else "",
        }

        csv_processing_results = _process_simulation_csvs(base_dir)

        return simulation_results, csv_processing_results

    except subprocess.TimeoutExpired as timeout_error:
        print(f"❌ ERROR: Proceso excedió tiempo límite: {str(timeout_error)}")
        return (
            {
                "status": "timeout",
                "message": f"El proceso excedió el tiempo límite: {str(timeout_error)}",
            },
            {},
        )
    except Exception as sim_error:
        print(f"❌ ERROR en secuencia de simulación: {str(sim_error)}")
        return (
            {
                "status": "error",
                "message": f"Error ejecutando secuencia de simulación: {str(sim_error)}",
            },
            {},
        )


def _process_simulation_csvs(sim_dir: str, single_file: bool = False):
    """Procesa archivos CSV resultantes de la simulación."""
    print("🔍 Buscando archivos CSV para procesar...")
    csv_files = [
        os.path.join(sim_dir, f) if not single_file else f
        for f in os.listdir(sim_dir)
        if f.endswith(".csv") and "Conv1" in f
    ]

    if not csv_files:
        print("⚠️ No se encontraron archivos CSV para procesar")
        return {"status": "warning", "message": "No se encontraron archivos CSV para procesar"}

    print(f"✅ Encontrados {len(csv_files)} archivos CSV para procesar")
    csv_processor = CSVProcessor()
    processed_results = []

    targets = csv_files[:1] if single_file else csv_files
    for csv_file in targets:
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(sim_dir, csv_file)
        try:
            print(f"🔄 Procesando archivo CSV: {csv_path}")
            result = csv_processor.process_simulation_csv(csv_path)
            processed_results.append({"file": csv_path, "result": result})
            print(f"✅ Archivo CSV procesado exitosamente: {csv_path}")
        except Exception as csv_error:
            print(f"❌ Error procesando CSV {csv_path}: {str(csv_error)}")
            processed_results.append({"file": csv_path, "error": str(csv_error)})

    return {
        "status": "success",
        "processed_files": len(processed_results),
        "results": processed_results,
    }


def inject_vhdl_faults(filter_faults: str, bias_faults: str, current_user: dict):
    """Modifica pesos/bias en el archivo VHDL fijo y ejecuta la simulación completa."""
    print(f"🔍 DEBUG: Recibidos parámetros - filter_faults: {filter_faults}, bias_faults: {bias_faults}")
    print(f"🔍 DEBUG: Usuario actual: {current_user}")

    try:
        vhdl_file_path = VHDL_FILE_PATH
        print(f"🔍 DEBUG: Verificando archivo VHDL: {vhdl_file_path}")

        if not os.path.exists(vhdl_file_path):
            print(f"❌ ERROR: Archivo VHDL no encontrado: {vhdl_file_path}")
            raise HTTPException(status_code=404, detail=f"Archivo VHDL no encontrado: {vhdl_file_path}")

        print("✅ Archivo VHDL encontrado")

        try:
            filter_config = json.loads(filter_faults) if filter_faults else {}
            bias_config = json.loads(bias_faults) if bias_faults else {}
            print(f"✅ Configuraciones parseadas - filter: {filter_config}, bias: {bias_config}")
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Error parseando JSON: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error parseando configuración JSON: {str(e)}")

        with open(vhdl_file_path, "r", encoding="utf-8") as f:
            vhdl_content = f.read()
        print("✅ Contenido del archivo VHDL leído")

        fault_config = {"filter_faults": filter_config, "bias_faults": bias_config}

        try:
            modified_content = modify_vhdl_weights_and_bias(vhdl_content, fault_config)
            print("✅ Contenido del archivo VHDL modificado")
            modification_results = {
                "status": "success",
                "message": "Pesos y bias modificados correctamente",
                "filter_modifications": len(filter_config),
                "bias_modifications": len(bias_config),
            }
        except Exception as mod_error:
            print(f"❌ ERROR modificando VHDL: {str(mod_error)}")
            modification_results = {
                "status": "error",
                "message": f"Error modificando VHDL: {str(mod_error)}",
            }
            modified_content = vhdl_content

        with open(vhdl_file_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
        print("✅ Archivo VHDL modificado guardado")

        simulation_results, csv_processing_results = _run_simulation_pipeline(VHDL_SIM_DIR)

        response_data = {
            "status": "success",
            "modification_results": modification_results,
            "simulation_results": simulation_results,
            "csv_processing_results": csv_processing_results,
            "vhdl_file": vhdl_file_path,
            "message": "Modificación de pesos, simulación y procesamiento CSV completado",
            "file_modified": True,
            "modification_timestamp": int(time.time() * 1000),
            "file_info": {
                "path": vhdl_file_path,
                "size": os.path.getsize(vhdl_file_path) if os.path.exists(vhdl_file_path) else 0,
                "last_modified": os.path.getmtime(vhdl_file_path) if os.path.exists(vhdl_file_path) else 0,
            },
        }

        print("✅ Respuesta preparada exitosamente")
        return sanitize_for_json(response_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR general: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el proceso: {str(e)}")


def run_golden_simulation(current_user: dict):
    """Inyecta valores golden al VHDL y ejecuta la simulación."""
    try:
        print("🚀 Iniciando simulación golden...")

        vhdl_file_path = VHDL_FILE_PATH

        if not os.path.exists(vhdl_file_path):
            raise HTTPException(status_code=404, detail=f"Archivo VHDL no encontrado: {vhdl_file_path}")

        print(f"📁 Archivo VHDL encontrado: {vhdl_file_path}")

        with open(vhdl_file_path, "r", encoding="utf-8") as file:
            original_content = file.read()

        modified_content = inject_golden_values_to_vhdl(original_content)

        with open(vhdl_file_path, "w", encoding="utf-8") as file:
            file.write(modified_content)

        print("✅ Valores golden inyectados en el archivo VHDL")

        modification_results = {
            "status": "success",
            "message": "Valores golden inyectados correctamente",
            "values_injected": {
                "filters": [f"FMAP_{i}" for i in range(1, 7)],
                "bias": [f"BIAS_VAL_{i}" for i in range(1, 7)],
            },
        }

        simulation_results, csv_processing_results = _run_simulation_pipeline(VHDL_SIM_DIR)

        response_data = {
            "status": "success",
            "simulation_type": "golden",
            "modification_results": modification_results,
            "simulation_results": simulation_results,
            "csv_processing_results": csv_processing_results,
            "vhdl_file": vhdl_file_path,
            "message": "Simulación golden completada",
        }

        print("✅ Simulación golden completada exitosamente")
        return sanitize_for_json(response_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR general en simulación golden: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en simulación golden: {str(e)}")
