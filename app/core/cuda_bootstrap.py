"""Bootstrap de librerías CUDA para TensorFlow.

Los wheels nvidia-*-cu12 instalan las librerías CUDA dentro de site-packages,
pero el enlazador dinámico no las encuentra a menos que estén en
LD_LIBRARY_PATH. Este módulo añade esas rutas y re-lanza el proceso UNA sola
vez para que el enlazador las tome.

Debe importarse y ejecutarse ANTES de importar TensorFlow (es decir, como lo
primero de main.py).
"""

import glob
import os
import sys

_GUARD_ENV = "HURA_CUDA_BOOTSTRAPPED"


def ensure_cuda_libpath() -> None:
    """Añadir las libs CUDA de los wheels nvidia a LD_LIBRARY_PATH y re-exec.

    Si ya se ejecutó en este árbol de procesos (marca _GUARD_ENV), no hace nada.
    Si no hay wheels CUDA instalados, tampoco hace nada (se usará CPU).
    """
    if os.environ.get(_GUARD_ENV):
        return  # el bootstrap ya se aplicó en este proceso

    nvidia_lib_dirs = []
    for site_dir in sys.path:
        if not site_dir:
            continue
        nvidia_lib_dirs.extend(glob.glob(os.path.join(site_dir, "nvidia", "*", "lib")))

    # Marcar antes de re-lanzar para evitar un bucle de re-exec.
    os.environ[_GUARD_ENV] = "1"

    if not nvidia_lib_dirs:
        print("ℹ️ Sin wheels CUDA en site-packages — TensorFlow usará CPU")
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    new_path = os.pathsep.join(sorted(set(nvidia_lib_dirs)))
    os.environ["LD_LIBRARY_PATH"] = new_path + (
        os.pathsep + current if current else ""
    )

    # Una invocación con `python -c` o `python -` no puede re-lanzarse de forma
    # fiable (el comando no está en sys.argv). En ese caso no se re-lanza.
    if not sys.argv or sys.argv[0] in ("-c", "-", ""):
        print("⚠️ Invocación no re-lanzable (-c/-): exporta LD_LIBRARY_PATH manualmente")
        return

    print("🔧 Configurando LD_LIBRARY_PATH para CUDA y re-lanzando el proceso...")
    # Re-lanzar para que el enlazador dinámico tome el nuevo LD_LIBRARY_PATH.
    os.execv(sys.executable, [sys.executable] + sys.argv)
