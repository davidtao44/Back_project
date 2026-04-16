import json
import os
import time
import uuid

from fastapi import HTTPException, UploadFile

from app.core.config import LAYER_OUTPUTS_DIR
from app.schemas.fault import FaultInjectionConfig
from app.utils.json_utils import sanitize_for_json
from fault_injection.manual_inference import ManualInference


def configure_fault_injection(fault_config: FaultInjectionConfig):
    """Valida la configuración de inyección de fallos en activaciones."""
    if not fault_config.layers:
        raise HTTPException(
            status_code=400,
            detail="Debe especificar al menos una capa para inyección de fallos",
        )

    valid_fault_types = ["bit_flip", "stuck_at_0", "stuck_at_1", "random_noise"]
    for layer_name, layer_config in fault_config.layers.items():
        if layer_config.fault_type not in valid_fault_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Tipo de fallo inválido '{layer_config.fault_type}' para la capa "
                    f"'{layer_name}'. Tipos válidos: {valid_fault_types}"
                ),
            )

        if layer_config.fault_rate < 0 or layer_config.fault_rate > 1:
            raise HTTPException(
                status_code=400,
                detail=f"La tasa de fallos debe estar entre 0 y 1 para la capa '{layer_name}'",
            )

    return {
        "success": True,
        "message": "Configuración de inyección de fallos validada correctamente",
        "config": fault_config.dict(),
        "layers_configured": len(fault_config.layers),
        "enabled": fault_config.enabled,
    }


async def run_fault_injector_inference(
    file: UploadFile,
    model_path: str,
    fault_config: str,
    current_user: dict,
):
    """Ejecuta inferencia manual con inyección de fallos para una imagen."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    image_data = await file.read()

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    fault_config_dict = None
    if fault_config:
        try:
            fault_config_dict = json.loads(fault_config)
            print(f"🔧 DEBUG: Configuración de fallos recibida: {fault_config_dict}")

            if "activation_faults" in fault_config_dict or "weight_faults" in fault_config_dict:
                print("🔧 DEBUG: Detectada configuración combinada de fallos")
            else:
                print("🔧 DEBUG: Configuración de fallos legacy (solo activaciones)")
        except json.JSONDecodeError:
            print(f"❌ ERROR: Configuración de fallos inválida: {fault_config}")
            raise HTTPException(status_code=400, detail="Configuración de fallos inválida")
    else:
        print("ℹ️ DEBUG: No se recibió configuración de fallos")

    session_id = f"user_{current_user.get('uid', 'anonymous')}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    manual_inference = ManualInference(
        model_path,
        output_dir=LAYER_OUTPUTS_DIR,
        session_id=session_id,
        fault_config=fault_config_dict,
    )

    results = manual_inference.perform_manual_inference(image_data)

    excel_filenames = [os.path.basename(f) for f in results["excel_files"] if f]
    image_filenames = [os.path.basename(f) for f in results["image_files"] if f]

    final_prediction = results["final_prediction"]
    if not final_prediction.get("success", True):
        response_data = {
            "success": False,
            "error_type": "numerical_overflow_underflow",
            "error_details": final_prediction["error"],
            "session_id": results["session_id"],
            "layer_outputs": results["layer_outputs"],
            "excel_files": excel_filenames,
            "image_files": image_filenames,
            "model_used": os.path.basename(model_path),
            "fault_injection": results["fault_injection"],
            "message": "Error numérico detectado durante la inferencia",
        }
    else:
        response_data = {
            "success": True,
            "session_id": results["session_id"],
            "predicted_class": final_prediction["predicted_class"],
            "confidence": final_prediction["confidence"],
            "all_probabilities": final_prediction["all_probabilities"],
            "layer_outputs": results["layer_outputs"],
            "excel_files": excel_filenames,
            "image_files": image_filenames,
            "model_used": os.path.basename(model_path),
            "fault_injection": results["fault_injection"],
            "message": "Inferencia manual completada exitosamente",
        }

    return sanitize_for_json(response_data)
