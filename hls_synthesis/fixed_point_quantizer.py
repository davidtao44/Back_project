import os

import numpy as np
import tensorflow as tf
from fastapi import HTTPException

from app.core.config import MODELS_DIR


def apply_fixed_point_ptq(model_name: str, total_bits: int = 16, int_bits: int = 6) -> dict:
    """
    Post-Training Quantization: redondea pesos al formato ap_fixed<total_bits, int_bits>.
    No requiere datos de entrenamiento — trabaja directamente sobre los pesos del modelo.

    ap_fixed<total_bits, int_bits>:
      - rango representable: [-2^(int_bits-1),  2^(int_bits-1) - 2^-frac_bits]
      - resolución: 2^-(total_bits - int_bits)
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")

    frac_bits = total_bits - int_bits
    if frac_bits <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"int_bits ({int_bits}) debe ser menor que total_bits ({total_bits})",
        )

    scale = 2.0 ** frac_bits
    max_val = (2 ** (int_bits - 1)) - (1.0 / scale)
    min_val = -(2 ** (int_bits - 1))

    model = tf.keras.models.load_model(model_path)

    layer_stats = []
    total_params = 0
    total_clipped = 0

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        quantized_weights = []
        layer_clipped = 0
        layer_params = 0

        for w in weights:
            layer_params += w.size
            clipped = int(np.sum((w < min_val) | (w > max_val)))
            layer_clipped += clipped

            w_clipped = np.clip(w, min_val, max_val)
            w_quantized = np.round(w_clipped * scale) / scale
            quantized_weights.append(w_quantized.astype(np.float32))

        layer.set_weights(quantized_weights)
        total_params += layer_params
        total_clipped += layer_clipped

        layer_stats.append({
            "layer": layer.name,
            "type": layer.__class__.__name__,
            "params": layer_params,
            "clipped": layer_clipped,
            "clip_pct": round(100.0 * layer_clipped / layer_params, 3) if layer_params > 0 else 0.0,
        })

    base, ext = os.path.splitext(model_name)
    quantized_name = f"{base}_ptq_b{total_bits}i{int_bits}{ext}"
    quantized_path = os.path.join(MODELS_DIR, quantized_name)
    model.save(quantized_path)

    return {
        "original_model": model_name,
        "quantized_model": quantized_name,
        "quantized_model_path": quantized_path,
        "format": f"ap_fixed<{total_bits},{int_bits}>",
        "total_bits": total_bits,
        "int_bits": int_bits,
        "frac_bits": frac_bits,
        "representable_range": [float(min_val), float(max_val)],
        "resolution": float(1.0 / scale),
        "total_params": total_params,
        "total_clipped": total_clipped,
        "clip_percentage": round(100.0 * total_clipped / total_params, 3) if total_params > 0 else 0.0,
        "layer_stats": layer_stats,
    }
