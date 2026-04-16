import numpy as np


def sanitize_for_json(obj):
    """Recursivamente limpia un objeto para asegurar que sea serializable a JSON.

    Reemplaza inf, -inf y NaN con valores válidos y convierte tipos numpy.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return 0.0
        if np.isinf(obj):
            return 1.0 if obj > 0 else 0.0
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        float_val = float(obj)
        if np.isnan(float_val):
            return 0.0
        if np.isinf(float_val):
            return 1.0 if float_val > 0 else 0.0
        return float_val
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if hasattr(obj, "__dict__"):
        return sanitize_for_json(obj.__dict__)
    try:
        return str(obj)
    except Exception:
        return "unknown_type"
