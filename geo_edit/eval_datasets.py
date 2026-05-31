"""Eval dataset registry — single source of truth for dataset id → parquet + eval template.

Used by both the inference driver (`geo_edit.scripts.async_generate_with_tool_call_api`)
and the eval driver (`geo_edit.evaluation.eval_unified`) so that callers only need to
pass a single `--dataset` id; the parquet path and eval template name are auto-resolved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

# dataset_id -> (parquet_relpath_under_data_root, eval_template_name)
DATASET_REGISTRY: Dict[str, Tuple[str, str]] = {
    # ─── in-distribution ───
    "visual_probe_easy":   ("eval/id/visual_probe_easy.parquet",   "visual_probe"),
    "visual_probe_medium": ("eval/id/visual_probe_medium.parquet", "visual_probe"),
    "visual_probe_hard":   ("eval/id/visual_probe_hard.parquet",   "visual_probe"),
    "reason_map":          ("eval/id/reason_map.parquet",          "reason_map"),
    "reason_map_plus":     ("eval/id/reason_map_plus.parquet",     "reason_map_plus"),
    "map_trace":           ("eval/id/map_trace.parquet",           "map_trace"),
    # ─── out-of-distribution ───
    "visworld_cube":         ("eval/ood/visworld_cube.parquet",         "visworld_eval"),
    "visworld_mmsi":         ("eval/ood/visworld_mmsi.parquet",         "visworld_eval"),
    "visworld_ballgame":     ("eval/ood/visworld_ballgame.parquet",     "visworld_eval"),
    "visworld_paperfolding": ("eval/ood/visworld_paperfolding.parquet", "visworld_eval"),
    "mapeval_visual":        ("eval/ood/mapeval_visual.parquet",        "mapeval_visual"),
    "babyvision":            ("eval/ood/babyvision.parquet",            "babyvision"),
    "vstar_bench":           ("eval/ood/vstar_bench.parquet",           "vstar_bench"),
}


def resolve_dataset(dataset_id: str, data_root: str = "./pedia_data") -> Tuple[str, str]:
    """Return (parquet_path, eval_template_name) for `dataset_id`.

    Raises KeyError with the list of registered ids if unknown.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset id {dataset_id!r}. "
            f"Registered: {sorted(DATASET_REGISTRY)}"
        )
    relpath, eval_template = DATASET_REGISTRY[dataset_id]
    return str(Path(data_root) / relpath), eval_template
