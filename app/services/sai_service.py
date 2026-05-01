"""Stuck-at Asymmetry Index (SAI) computation.

Two flavors are computed from each paired stuck-at-0 / stuck-at-1 campaign:

  SAI_prop = (F_prop_s@1 − F_prop_s@0) / (F_prop_s@1 + F_prop_s@0)
      F_prop = n_prop / n_inj           (any output change per injection)

  SAI_misc = (F_misc_s@1 − F_misc_s@0) / (F_misc_s@1 + F_misc_s@0)
      F_misc = n_misc / n_prop          (conditional: misclassification rate
                                         given the fault propagated)

Domain: SAI ∈ [−1, 1]. Only meaningful for permanent (stuck-at) faults.

Interpretation (same axis for both flavors):
  |SAI| < 0.1   → DNN approximately symmetric to s@0 vs s@1.
  SAI > 0       → DNN more sensitive to s@1 (false activations dominate).
  SAI < 0       → DNN more sensitive to s@0 (suppressed activations dominate).
"""

from typing import Any, Dict, Optional

SYMMETRY_THRESHOLD = 0.1


def compute_sai(f_s0: Optional[float], f_s1: Optional[float]) -> Optional[float]:
    """Return SAI given two factors. None if either is undefined or sum is 0."""
    if f_s0 is None or f_s1 is None:
        return None
    denom = f_s1 + f_s0
    if denom == 0:
        return None
    return (f_s1 - f_s0) / denom


def interpret_sai(sai: Optional[float], flavor: str = "prop") -> str:
    if sai is None:
        if flavor == "misc":
            return (
                "undefined: at least one stuck-at type produced no propagated faults — "
                "conditional misclassification rate is not defined"
            )
        return "undefined: no propagated faults observed for either stuck-at type"
    if abs(sai) < SYMMETRY_THRESHOLD:
        return "approximately symmetric — comparable sensitivity to s@0 and s@1"
    if sai > 0:
        if flavor == "misc":
            return "asymmetric toward s@1 — propagated stuck-at-1 faults are more often misclassifying"
        return "asymmetric toward s@1 — DNN more sensitive to stuck-at-1 (false activations)"
    if flavor == "misc":
        return "asymmetric toward s@0 — propagated stuck-at-0 faults are more often misclassifying"
    return "asymmetric toward s@0 — DNN more sensitive to stuck-at-0 (suppressed activations)"


def compute_sai_from_runs(run_s0: Dict[str, Any], run_s1: Dict[str, Any]) -> Dict[str, Any]:
    """Build the SAI summary (both prop and misc flavors) from two campaign runs.

    Each run dict may expose:
      - n_inj  (int)  number of injections performed
      - n_prop (int)  number of injections that propagated to the output
      - n_misc (int)  propagated injections that turned a correct golden
                      prediction into a wrong one (fault-induced misclass)

    F_prop = n_prop / n_inj.
    F_misc = n_misc / n_prop  (conditional — undefined when n_prop = 0).
    """
    n_inj_s0 = int(run_s0.get("n_inj", 0))
    n_prop_s0 = int(run_s0.get("n_prop", 0))
    n_misc_s0 = int(run_s0.get("n_misc", 0))
    n_inj_s1 = int(run_s1.get("n_inj", 0))
    n_prop_s1 = int(run_s1.get("n_prop", 0))
    n_misc_s1 = int(run_s1.get("n_misc", 0))

    f_prop_s0 = (n_prop_s0 / n_inj_s0) if n_inj_s0 > 0 else 0.0
    f_prop_s1 = (n_prop_s1 / n_inj_s1) if n_inj_s1 > 0 else 0.0
    sai_prop = compute_sai(f_prop_s0, f_prop_s1)

    f_misc_s0 = (n_misc_s0 / n_prop_s0) if n_prop_s0 > 0 else None
    f_misc_s1 = (n_misc_s1 / n_prop_s1) if n_prop_s1 > 0 else None
    sai_misc = compute_sai(f_misc_s0, f_misc_s1)

    return {
        "n_inj_s0": n_inj_s0,
        "n_inj_s1": n_inj_s1,
        "n_prop_s0": n_prop_s0,
        "n_prop_s1": n_prop_s1,
        "n_misc_s0": n_misc_s0,
        "n_misc_s1": n_misc_s1,
        "f_prop_s0": round(f_prop_s0, 6),
        "f_prop_s1": round(f_prop_s1, 6),
        "f_misc_s0": round(f_misc_s0, 6) if f_misc_s0 is not None else None,
        "f_misc_s1": round(f_misc_s1, 6) if f_misc_s1 is not None else None,
        "sai": round(sai_prop, 6) if sai_prop is not None else None,
        "sai_misc": round(sai_misc, 6) if sai_misc is not None else None,
        "interpretation": interpret_sai(sai_prop, "prop"),
        "interpretation_misc": interpret_sai(sai_misc, "misc"),
    }
