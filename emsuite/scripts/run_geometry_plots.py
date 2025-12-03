# emsuite/scripts/run_geometry_plots.py
from __future__ import annotations

import os
import math
import json
from typing import Dict, List

import matplotlib.pyplot as plt

from emsuite.evolution.selection import EvolutionConfig, evolve_population
from emsuite.analysis.ensembles import MicroTopologyStats, macro_layered_dag_from_stats
from emsuite.physics.geometry import estimate_g_star, alpha_from_g_star
from emsuite.core.graph import DiGraph


def _estimate_dimension_via_balls(
    graph: DiGraph,
    max_radius: int = 5,
    n_centers: int = 10,
) -> float:
    """
    Crude Minkowski dimension estimator via ball growth:

        - Choose up to n_centers random vertices.
        - For each center, compute ball sizes B(r) for r = 1..max_radius
          using BFS on the *undirected* underlying graph.
        - Fit log B(r) ~ D log r by least squares over all (r, B) pairs.

    This is intentionally simple and does not claim to be precise; it's
    just a diagnostic observable for emergent dimension.
    """
    import random
    verts = list(graph.vertices())
    if not verts:
        return 0.0

    centers = random.sample(verts, min(len(verts), n_centers))

    xs: List[float] = []
    ys: List[float] = []

    for c in centers:
        # undirected adjacency for distances
        adj: Dict[int, List[int]] = {}
        for v in verts:
            adj[v] = []
        for u in verts:
            for v in graph.successors(u):
                adj[u].append(v)
                adj[v].append(u)
        # BFS from c
        from collections import deque
        dist = {c: 0}
        q = deque([c])
        order = [c]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if dist[v] <= max_radius:
                        q.append(v)
                        order.append(v)

        # ball sizes
        for r in range(1, max_radius + 1):
            count = sum(1 for v in dist.values() if v <= r)
            if count > 0:
                xs.append(math.log(r))
                ys.append(math.log(count))

    if len(xs) < 2:
        return 0.0

    # Least-squares slope D = cov(x,y) / var(x)
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    den = sum((x - x_bar) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return num / den


def main() -> None:
    # --- Step 1: micro evolution to get stats ---
    evo_cfg = EvolutionConfig(
        n_nodes=8,
        population_size=32,
        generations=20,
        edge_prob=0.3,
        elite_fraction=0.25,
        cycle_penalty_weight=1.0,
        random_seed=555,
    )
    population, avg_topology = evolve_population(evo_cfg)
    micro_stats = MicroTopologyStats(**avg_topology)

    # --- Step 2: generate macro ensemble, record g_R flows, α_est, D_est ---
    N_macro = 200
    n_samples = 30
    scales = [1, 2, 4, 8]

    g_flows: List[Dict[int, float]] = []
    alphas: List[float] = []
    dims: List[float] = []

    for seed in range(n_samples):
        G = macro_layered_dag_from_stats(micro_stats, N=N_macro, seed=seed)
        dag, _ = G.condensation()

        g_info = estimate_g_star(dag, scales=scales)
        g_star = g_info.get("g_star", None)
        if g_star is not None and g_star > 0:
            alphas.append(alpha_from_g_star(g_star, D=4, q=2))

        flow = g_info.get("flow", {})
        g_flows.append(flow)

        dims.append(_estimate_dimension_via_balls(G, max_radius=5, n_centers=10))

    os.makedirs("figures", exist_ok=True)

    # --- Plot 1: RG flow g_R vs R for a representative sample ---
    if g_flows:
        sample_flow = g_flows[0]
        Rs = sorted(R for R, g in sample_flow.items() if g is not None)
        g_vals = [sample_flow[R] for R in Rs]
        plt.figure()
        plt.plot(Rs, g_vals, marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Block scale R")
        plt.ylabel("g_R")
        plt.title("Sample RG flow g_R(R)")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "rg_flow_sample.png"))
        plt.close()

    # --- Plot 2: histogram of α_est ---
    if alphas:
        plt.figure()
        plt.hist(alphas, bins=10, edgecolor="black")
        plt.xlabel("α_est")
        plt.ylabel("count")
        plt.title("Distribution of α_est across macro ensemble")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "alpha_hist.png"))
        plt.close()

    # --- Plot 3: histogram of crude dimension estimates ---
    if dims:
        plt.figure()
        plt.hist(dims, bins=10, edgecolor="black")
        plt.xlabel("D_est")
        plt.ylabel("count")
        plt.title("Crude dimension estimates from ball growth")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "dimension_hist.png"))
        plt.close()

    summary = {
        "micro_stats": avg_topology,
        "n_samples": n_samples,
        "alpha_mean": sum(alphas) / len(alphas) if alphas else None,
        "dim_mean": sum(dims) / len(dims) if dims else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
