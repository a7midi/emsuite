# emsuite/analysis/stats.py
from __future__ import annotations
from typing import Callable, Dict, Any, List
import math
import statistics as stats

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.physics.geometry import estimate_g_star, alpha_from_g_star


def sample_coupling_distribution(
    ensemble_fn: Callable[..., DiGraph],
    ensemble_params: Dict[str, Any],
    n_samples: int,
    grammar_builder: Callable[[DiGraph, int], Grammar],
    scales: List[int] | None = None,
) -> Dict[str, Any]:
    """
    Monte Carlo sampling of g* and α_est across a macro-ensemble.

    ensemble_fn(seed=seed, **ensemble_params) -> graph
    grammar_builder(graph, seed) -> Grammar

    Returns:
      {
        "g_values": [...],
        "alpha_values": [...],
        "g_mean": ...,
        "g_std": ...,
        "alpha_mean": ...,
        "alpha_std": ...,
        "stable_frac": ...,
        "statuses": [...],
        "flow_samples": [...],
      }

    Note: There is *no* parameter tuning to target any particular value.
    """
    if scales is None:
        scales = [1, 2, 4, 8]

    g_values: List[float] = []
    alpha_values: List[float] = []
    statuses: List[str] = []
    flow_samples: List[Dict[int, float | None]] = []

    for seed in range(n_samples):
        graph = ensemble_fn(seed=seed, **ensemble_params)
        dag, _ = graph.condensation()
        grammar = grammar_builder(graph, seed)
        res = estimate_g_star(dag, scales=scales)
        g_star = res.get("g_star")
        statuses.append(res.get("status", "UNKNOWN"))
        flow_samples.append(res.get("flow", {}))
        if g_star is not None and g_star > 0:
            g_values.append(g_star)
            alpha_values.append(alpha_from_g_star(g_star))

    def safe_mean(xs: List[float]) -> float:
        return float(stats.mean(xs)) if xs else float("nan")

    def safe_std(xs: List[float]) -> float:
        return float(stats.pstdev(xs)) if xs else float("nan")

    stable_frac = (
        statuses.count("CONVERGED") / len(statuses) if statuses else 0.0
    )

    return {
        "g_values": g_values,
        "alpha_values": alpha_values,
        "g_mean": safe_mean(g_values),
        "g_std": safe_std(g_values),
        "alpha_mean": safe_mean(alpha_values),
        "alpha_std": safe_std(alpha_values),
        "stable_frac": stable_frac,
        "statuses": statuses,
        "flow_samples": flow_samples,
    }
