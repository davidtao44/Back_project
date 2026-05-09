"""Persistencia de campañas SAI en SQLite.

Solo se almacenan campañas con un único fallo inyectado
(1 capa × 1 posición × 1 bit). El control de esa condición lo hace el
caller; este módulo se limita a escribir y leer.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "campaign_results")
DB_PATH = os.path.join(DB_DIR, "sai_campaigns.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user TEXT,
                model_name TEXT,
                num_samples INTEGER,
                granularity TEXT,
                golden_accuracy REAL,
                sai REAL,
                mai_misc REAL,
                interpretation TEXT,
                interpretation_misc TEXT,
                n_inj_s0 INTEGER,
                n_prop_s0 INTEGER,
                n_misc_s0 INTEGER,
                f_prop_s0 REAL,
                f_misc_s0 REAL,
                n_inj_s1 INTEGER,
                n_prop_s1 INTEGER,
                n_misc_s1 INTEGER,
                f_prop_s1 REAL,
                f_misc_s1 REAL,
                execution_time_seconds REAL,
                json_blob TEXT
            );

            CREATE TABLE IF NOT EXISTS faults (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                fault_type TEXT,
                layer_name TEXT,
                target_type TEXT,
                position TEXT,
                bit_position INTEGER,
                original_value REAL,
                modified_value REAL,
                FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
            );

            CREATE INDEX IF NOT EXISTS idx_faults_campaign ON faults(campaign_id);
            CREATE INDEX IF NOT EXISTS idx_faults_layer ON faults(layer_name);
            CREATE INDEX IF NOT EXISTS idx_campaigns_user ON campaigns(user);
            CREATE INDEX IF NOT EXISTS idx_campaigns_timestamp ON campaigns(timestamp);
            """
        )


def _fault_row(campaign_id: str, fault_type: str, fault: Dict[str, Any]) -> tuple:
    position = fault.get("position", [])
    if isinstance(position, tuple):
        position = list(position)
    return (
        campaign_id,
        fault_type,
        fault.get("layer_name"),
        fault.get("target_type"),
        json.dumps(position),
        fault.get("bit_position"),
        fault.get("original_value"),
        fault.get("modified_value"),
    )


def save_sai_campaign(
    results: Dict[str, Any],
    user: str,
    faults_s0: List[Dict[str, Any]],
    faults_s1: List[Dict[str, Any]],
) -> str:
    """Persiste una campaña SAI en la BD.

    Returns:
        El campaign_id usado.
    """
    init_db()

    campaign_info = results.get("campaign_info", {}) or {}
    sai_global = results.get("sai_global", {}) or {}
    golden_metrics = (results.get("golden_results", {}) or {}).get("metrics", {}) or {}

    campaign_id = campaign_info.get("session_id") or f"sai_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    model_name = os.path.basename(campaign_info.get("model_path", "") or "")

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO campaigns (
                campaign_id, timestamp, user, model_name, num_samples, granularity,
                golden_accuracy, sai, mai_misc, interpretation, interpretation_misc,
                n_inj_s0, n_prop_s0, n_misc_s0, f_prop_s0, f_misc_s0,
                n_inj_s1, n_prop_s1, n_misc_s1, f_prop_s1, f_misc_s1,
                execution_time_seconds, json_blob
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                campaign_id,
                datetime.utcnow().isoformat(),
                user,
                model_name,
                campaign_info.get("num_samples"),
                campaign_info.get("granularity"),
                golden_metrics.get("accuracy"),
                sai_global.get("sai"),
                sai_global.get("mai_misc"),
                sai_global.get("interpretation"),
                sai_global.get("interpretation_misc"),
                sai_global.get("n_inj_s0"),
                sai_global.get("n_prop_s0"),
                sai_global.get("n_misc_s0"),
                sai_global.get("f_prop_s0"),
                sai_global.get("f_misc_s0"),
                sai_global.get("n_inj_s1"),
                sai_global.get("n_prop_s1"),
                sai_global.get("n_misc_s1"),
                sai_global.get("f_prop_s1"),
                sai_global.get("f_misc_s1"),
                campaign_info.get("execution_time_seconds"),
                json.dumps(results, default=str),
            ),
        )

        conn.execute("DELETE FROM faults WHERE campaign_id = ?", (campaign_id,))

        rows = [_fault_row(campaign_id, "stuck_at_0", f) for f in faults_s0]
        rows += [_fault_row(campaign_id, "stuck_at_1", f) for f in faults_s1]
        if rows:
            conn.executemany(
                """
                INSERT INTO faults (
                    campaign_id, fault_type, layer_name, target_type,
                    position, bit_position, original_value, modified_value
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                rows,
            )

    return campaign_id


def get_campaigns_summary() -> List[Dict[str, Any]]:
    init_db()
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT campaign_id, timestamp, user, model_name, num_samples, granularity,
                   golden_accuracy, sai, mai_misc, interpretation, interpretation_misc,
                   n_inj_s0, n_prop_s0, n_misc_s0, f_prop_s0, f_misc_s0,
                   n_inj_s1, n_prop_s1, n_misc_s1, f_prop_s1, f_misc_s1,
                   execution_time_seconds
            FROM campaigns
            ORDER BY timestamp DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_campaign_faults(campaign_id: str) -> List[Dict[str, Any]]:
    init_db()
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT fault_type, layer_name, target_type, position, bit_position,
                   original_value, modified_value
            FROM faults
            WHERE campaign_id = ?
            ORDER BY fault_type, layer_name, bit_position
            """,
            (campaign_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["position"] = json.loads(d["position"]) if d["position"] else []
        except (json.JSONDecodeError, TypeError):
            pass
        result.append(d)
    return result
