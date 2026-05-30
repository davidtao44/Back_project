"""Configuración de GPU para TensorFlow.

Debe invocarse al arrancar el backend, antes de cargar o entrenar modelos.
"""

import tensorflow as tf


def configure_gpu() -> dict:
    """Activar memory growth en las GPUs disponibles y registrar su estado.

    memory_growth evita que TensorFlow reserve toda la VRAM de golpe, lo que
    permite que el backend conviva con otros procesos en la misma GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("ℹ️ GPU no disponible — TensorFlow usará CPU")
        return {"gpu_available": False, "devices": []}

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        names = [gpu.name for gpu in gpus]
        print(f"✅ GPU habilitada para TensorFlow: {names}")
        return {"gpu_available": True, "devices": names}
    except RuntimeError as e:
        # set_memory_growth debe llamarse antes de inicializar las GPUs
        print(f"⚠️ No se pudo configurar memory growth en la GPU: {e}")
        return {
            "gpu_available": True,
            "devices": [gpu.name for gpu in gpus],
            "warning": str(e),
        }
