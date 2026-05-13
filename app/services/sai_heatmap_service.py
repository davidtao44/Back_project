"""Servicio para construir los datos del heatmap 3D de SAI/MAI.

Los datos NO se promedian. Cada campaña queda como un punto crudo en su
grupo (layer, target_type). El frontend se encarga del render con
ECharts GL.
"""

import json
import sqlite3
from typing import Any, Dict, List

from app.utils.campaign_store import DB_PATH, init_db


def _parse_position(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (json.JSONDecodeError, TypeError):
        return []


def get_sai_heatmap_data() -> Dict[str, Any]:
    """Devuelve las campañas SAI agrupadas por (layer, target_type).

    Toma solo las filas con `fault_type = 'stuck_at_0'` para no duplicar
    (s@0 y s@1 comparten layer/posición/bit; cada par representa una
    única campaña y SAI/MAI son métricas de la campaña, no del fault
    individual).
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                f.layer_name,
                f.target_type,
                f.position,
                f.bit_position,
                c.campaign_id,
                c.timestamp,
                c.model_name,
                c.sai,
                c.mai_misc,
                c.f_prop_s0,
                c.f_prop_s1,
                c.f_misc_s0,
                c.f_misc_s1
            FROM faults f
            JOIN campaigns c ON c.campaign_id = f.campaign_id
            WHERE f.fault_type = 'stuck_at_0'
            ORDER BY f.layer_name, f.target_type, f.bit_position
            """
        ).fetchall()
    finally:
        conn.close()

    groups: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r["layer_name"] or "", r["target_type"] or "")
        if key not in groups:
            groups[key] = {
                "layer": r["layer_name"],
                "target_type": r["target_type"],
                "campaigns": [],
            }
        position = _parse_position(r["position"])
        groups[key]["campaigns"].append(
            {
                "campaign_id": r["campaign_id"],
                "timestamp": r["timestamp"],
                "model_name": r["model_name"],
                "position": position,
                "position_label": f"[{','.join(str(p) for p in position)}]",
                "bit_position": r["bit_position"],
                "sai": r["sai"],
                "mai_misc": r["mai_misc"],
                "f_prop_s0": r["f_prop_s0"],
                "f_prop_s1": r["f_prop_s1"],
                "f_misc_s0": r["f_misc_s0"],
                "f_misc_s1": r["f_misc_s1"],
            }
        )

    return {"groups": list(groups.values())}
