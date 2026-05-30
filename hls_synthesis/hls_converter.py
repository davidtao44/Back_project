import os
import re
import shutil
import uuid

import tensorflow as tf
from fastapi import HTTPException

from app.core.config import HLS_OUTPUTS_DIR, MODELS_DIR

DEFAULT_PART = "xc7z020clg400-1"  # Pynq-Z2 (disponible en Vitis HLS 2025.1 por defecto)


def _patch_project_tcl(project_dir: str, part: str) -> None:
    """
    hls4ml ignora el parámetro part y siempre escribe xcvu13p por defecto.
    Este parche sobreescribe el part en project.tcl con el valor correcto.
    """
    tcl_path = os.path.join(project_dir, "project.tcl")
    if not os.path.exists(tcl_path):
        return
    with open(tcl_path, "r") as f:
        content = f.read()
    content = re.sub(r'set part "[^"]*"', f'set part "{part}"', content)
    with open(tcl_path, "w") as f:
        f.write(content)


def _patch_build_prj_tcl(project_dir: str) -> None:
    """
    Elimina la línea obsoleta 'config_array_partition -maximum_size' que
    hls4ml genera y que Vitis HLS 2025.1 no reconoce, evitando el error
    [HLS 200-101] en el log.
    """
    tcl_path = os.path.join(project_dir, "build_prj.tcl")
    if not os.path.exists(tcl_path):
        return
    with open(tcl_path, "r") as f:
        lines = f.readlines()
    filtered = [
        line for line in lines
        if "config_array_partition" not in line
    ]
    with open(tcl_path, "w") as f:
        f.writelines(filtered)


def convert_to_hls(model_name: str, user_id: str, config: dict) -> dict:
    """
    Convierte un modelo Keras a proyecto HLS usando hls4ml.
    Genera el proyecto C++ sintetizable — no requiere Vivado instalado.
    Vivado HLS / Vitis se necesita solo para síntesis RTL posterior.
    """
    try:
        import hls4ml
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="hls4ml no está instalado. Ejecuta: pip install hls4ml",
        )

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")

    part = config.get("part") or DEFAULT_PART

    session_id = f"hls_{user_id}_{uuid.uuid4().hex[:8]}"
    session_dir = os.path.join(HLS_OUTPUTS_DIR, session_id)
    project_dir = os.path.join(session_dir, "hls_project")
    os.makedirs(session_dir, exist_ok=True)

    model = tf.keras.models.load_model(model_path)

    hls_config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="model",
        default_precision=config.get("precision", "ap_fixed<16,6>"),
        default_reuse_factor=config.get("reuse_factor", 1),
    )

    # Strategy controla si hls4ml desenrolla todo (Latency) o serializa con
    # reuse_factor (Resource). Para FPGAs pequeñas Resource evita la explosión
    # de instrucciones que mata la síntesis en Vitis HLS 2025.1.
    hls_config.setdefault("Model", {})
    hls_config["Model"]["Strategy"] = config.get("strategy", "Latency")

    hls_model = hls4ml.converters.convert_from_keras_model(
        model=model,
        hls_config=hls_config,
        output_dir=project_dir,
        backend=config.get("backend", "Vivado"),
        clock_period=config.get("clock_period", 5),
        io_type=config.get("io_type", "io_parallel"),
    )
    hls_model.write()

    # hls4ml ignora el part — lo parcheamos directamente en project.tcl
    _patch_project_tcl(project_dir, part)

    # Elimina comandos TCL obsoletos que Vitis HLS 2025.1 no soporta
    _patch_build_prj_tcl(project_dir)

    zip_path = shutil.make_archive(
        os.path.join(session_dir, "hls_project"),
        "zip",
        session_dir,
        "hls_project",
    )

    return {
        "session_id": session_id,
        "model": model_name,
        "backend": config.get("backend", "Vivado"),
        "precision": config.get("precision", "ap_fixed<16,6>"),
        "reuse_factor": config.get("reuse_factor", 1),
        "clock_period_ns": config.get("clock_period", 5),
        "part": part,
        "project_dir": project_dir,
        "zip_path": zip_path,
        "download_path": os.path.relpath(zip_path),
    }
