# emsuite/analysis/ensembles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random

from emsuite.core.graph import DiGraph
from emsuite.evolution.selection import measure_topology


@dataclass
class MicroTopologyStats:
    """
    Summary stats extracted from evolved small graphs.

    These are exactly the quantities we want macro ensembles to mimic.
    """
    avg_out_degree: float
    avg_in_degree: float
    diamond_fraction: float
    cycle_fraction: float


def summarize_micro_topology(graphs: List[DiGraph]) -> MicroTopologyStats:
    """
    Aggregate topology stats over a list of micrographs.
    """
    if not graphs:
        return MicroTopologyStats(0.0, 0.0, 0.0, 0.0)

    agg = {
        "avg_out_degree": 0.0,
        "avg_in_degree": 0.0,
        "diamond_fraction": 0.0,
        "cycle_fraction": 0.0,
    }
    for g in graphs:
        topo = measure_topology(g)
        for k in agg.keys():
            agg[k] += topo[k]
    n = float(len(graphs))
    for k in agg.keys():
        agg[k] /= n

    return MicroTopologyStats(**agg)


def macro_layered_dag_from_stats(
    stats: MicroTopologyStats,
    N: int,
    seed: int = 0,
) -> DiGraph:
    """
    Build a layered DAG with N vertices whose coarse observables mimic
    the given micro topology stats:

      - avg_out_degree ~ stats.avg_out_degree
      - diamond_fraction ~ stats.diamond_fraction
      - cycle_fraction ~ 0 by construction (DAG)

    Construction:
      - We create L layers; vertices only connect from layer ℓ to ℓ+1.
      - For each vertex x in layer ℓ+1 we pick a number of parents in
        layer ℓ; we choose 2+ parents with higher probability when
        stats.diamond_fraction is large, to encourage diamonds.
    """
    rng = random.Random(seed)
    g = DiGraph()

    if N <= 0:
        return g

    # --- Layer sizes ---
    # Choose number of layers roughly sqrt(N) to get depth ~ sqrt(N)
    L = max(3, int(N ** 0.5))
    base = N // L
    rem = N % L
    layer_sizes: List[int] = []
    for i in range(L):
        layer_sizes.append(base + (1 if i < rem else 0))

    # Assign vertex ids and layers
    layers: List[List[int]] = [[] for _ in range(L)]
    vid = 0
    for ell in range(L):
        for _ in range(layer_sizes[ell]):
            g.add_vertex(vid)
            layers[ell].append(vid)
            vid += 1

    # --- Edge generation ---
    target_branch = max(1.0, stats.avg_out_degree)

    for ell in range(L - 1):
        parents = layers[ell]
        children = layers[ell + 1]
        if not parents or not children:
            continue

        for x in children:
            # With probability ~diamond_fraction, ensure x has at least 2 parents.
            if rng.random() < stats.diamond_fraction:
                min_parents = 2
            else:
                min_parents = 1

            # Desired parent count around target_branch, but capped.
            desired = max(min_parents, int(round(target_branch)))
            desired = min(desired, len(parents))
            if desired <= 0:
                continue

            chosen_parents = rng.sample(parents, desired)
            for p in chosen_parents:
                g.add_edge(p, x)

    # By construction this is a DAG: edges only go forward in layer index.
    return g


def macro_polynomial_dag(
    N: int,
    depth: int = 10,
    branching: float = 2.0,
    seed: int = 0,
) -> DiGraph:
    """
    Simple 'toy' polynomial-growth DAG used for diagnostics.

    This is NOT micro-informed. It exists only as a control ensemble:
      - depth layers,
      - layer sizes roughly increasing with a polynomial profile,
      - each node in layer ℓ connects to ~branching children in ℓ+1.
    """
    rng = random.Random(seed)
    g = DiGraph()

    if N <= 0 or depth <= 1:
        return g

    # Allocate layer sizes with a crude polynomial profile
    # s_ℓ ∝ (ℓ+1)^p, normalized to sum to N.
    p = 2.0
    weights = [(ell + 1) ** p for ell in range(depth)]
    total_w = sum(weights)
    raw_sizes = [w / total_w * N for w in weights]
    sizes = [max(1, int(round(x))) for x in raw_sizes]

    # Fix rounding so total is N
    diff = sum(sizes) - N
    if diff > 0:
        for i in range(diff):
            idx = depth - 1 - (i % depth)
            if sizes[idx] > 1:
                sizes[idx] -= 1
    elif diff < 0:
        diff = -diff
        for i in range(diff):
            idx = i % depth
            sizes[idx] += 1

    layers: List[List[int]] = [[] for _ in range(depth)]
    vid = 0
    for ell in range(depth):
        for _ in range(sizes[ell]):
            g.add_vertex(vid)
            layers[ell].append(vid)
            vid += 1

    for ell in range(depth - 1):
        parents = layers[ell]
        children = layers[ell + 1]
        if not parents or not children:
            continue
        for u in parents:
            # Binomial-ish: each edge added with prob p so E[deg+] ~ branching
            p_edge = min(1.0, branching / max(1, len(children)))
            for v in children:
                if rng.random() < p_edge:
                    g.add_edge(u, v)

    return g
