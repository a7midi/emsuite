# emsuite/analysis/observables.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math

from emsuite.core.graph import DiGraph, Vertex
from emsuite.evolution.selection import measure_topology


def estimate_minkowski_dimension(
    graph: DiGraph,
    radii: List[int],
    sample: int | None = None,
) -> float | None:
    """
    Estimate an effective dimension from ball-growth:

        |B(r)| ~ r^D  => log |B(r)| ~ D log r.

    We do a linear fit on (log r, log avg|B(r)|).
    """
    growth = graph.average_ball_growth(radii=radii, sample=sample, directed=False)
    xs = []
    ys = []
    for r, size in growth.items():
        if r > 0 and size > 0:
            xs.append(math.log(float(r)))
            ys.append(math.log(float(size)))
    if len(xs) < 2:
        return None
    # simple least-squares slope
    n = len(xs)
    sumx = sum(xs)
    sumy = sum(ys)
    sumx2 = sum(x * x for x in xs)
    sumxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sumx2 - sumx * sumx
    if denom == 0:
        return None
    D = (n * sumxy - sumx * sumy) / denom
    return float(D)


def diamond_density(graph: DiGraph) -> float:
    """
    Fraction of vertices that participate in at least one diamond
    (reusing the micro-level definition on condensation DAG).
    """
    from emsuite.core.projection import find_diamonds
    diamonds = find_diamonds(graph)
    if not graph.vertices():
        return 0.0
    involved: set[Vertex] = set()
    for d in diamonds:
        involved.update([d.i, d.v, d.w, d.x])
    return len(involved) / len(graph.vertices())


def basic_observables(graph: DiGraph) -> Dict[str, float]:
    """
    Convenience wrapper: degrees, diamonds, cycles, dimension.
    """
    topo = measure_topology(graph)
    dim = estimate_minkowski_dimension(graph, radii=[1, 2, 3, 4], sample=None)
    topo["minkowski_dim"] = float(dim) if dim is not None else float("nan")
    topo["diamond_density"] = diamond_density(graph)
    return topo
