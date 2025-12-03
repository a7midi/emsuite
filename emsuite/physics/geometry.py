# emsuite/physics/geometry.py
from __future__ import annotations

from typing import Dict, Optional, List, Tuple
import math

from emsuite.core.graph import DiGraph, Vertex


def V1(graph: DiGraph, v: Vertex) -> int:
    """
    One-step cone volume V1(v).

    Paper definition (Def. 10.3):
        V_r(v) = #{ γ : path in G↓, |γ| ≤ r, target(γ) = v }.

    For r = 1, this is the number of length-1 paths ending at v, which
    is |Pred(v)|. Since κ(i→j) only uses *differences* of V1, we can set:

        V1(v) := |Pred(v)| = graph.in_degree(v).
    """
    return graph.in_degree(v)


def kappa_edge(graph: DiGraph, u: Vertex, v: Vertex) -> float:
    """
    Discrete sectional curvature κ(i→j) on edges.

    Paper (Def. 10.5):
        κ(i→j) := V1(j) − V1(i) = |Pred(j)| − |Pred(i)|.
    """
    return float(V1(graph, v) - V1(graph, u))


def rho_mem_node(graph: DiGraph, v: Vertex) -> float:
    """
    Memory density ρ_mem(v) at a node.

    Paper (Def. 3.4 / 10.7):
      - On minimal-depth layer D0 of G↓:
            ρ_mem(v) := deg⁺(v) − 1 = out_degree(v) − 1.
      - For higher depths: use block-averaging along depth windows.

    Numerical approximation used here:
        ρ_mem(v) := max(0, out_degree(v) − 1)

    at all depths, then depth-block averaged in block_averages().
    """
    return max(0.0, float(graph.out_degree(v) - 1))


# Backwards-compatible aliases used elsewhere in the suite
def rho_mem(graph: DiGraph, v: Vertex) -> float:
    return rho_mem_node(graph, v)


def kappa_curvature(graph: DiGraph, u: Vertex, v: Vertex) -> float:
    return kappa_edge(graph, u, v)


def block_averages(
    dag: DiGraph,
    R: int,
) -> Optional[Tuple[float, float, float]]:
    """
    Block-averaged curvature, memory density, and Einstein–Memory ratio
    on a condensation DAG.

    Paper (Def. 10.12):

      For a fixed block size R, define a slice consisting of edges i→j with
      depth(i) ≡ 0 (mod R), and set:

          κ_R       := (1 / N_R) ∑ κ(i→j)
          (ρ_mem)_R := (1 / N_R) ∑ ρ_mem(j),

      where the sum runs over all such edges in the slice. Then

          g_R := κ_R / (ρ_mem)_R
               = [∑ κ(i→j)] / [∑ ρ_mem(j)].

    Implementation details:

      - depth_map() gives depth(i) on the DAG.
      - We restrict to edges with depth(i) ≡ 0 (mod R).
      - We compute:

            sum_kappa = ∑ κ(i→j)
            sum_rho   = ∑ ρ_mem(j)
            N_R       = number of such edges

        then:

            κ_R       = sum_kappa / N_R
            (ρ_mem)_R = sum_rho   / N_R
            g_R       = κ_R / (ρ_mem)_R = sum_kappa / sum_rho

    Returns:
        (κ_R, (ρ_mem)_R, g_R) or None if the slice has no edges or sum_rho ≤ 0.
    """
    depths: Dict[Vertex, int] = dag.depth_map()
    if not depths:
        return None

    sum_kappa = 0.0
    sum_rho = 0.0
    edge_count = 0

    for u in dag.vertices():
        du = depths[u]
        if du % R != 0:
            continue
        for v in dag.successors(u):
            edge_count += 1
            sum_kappa += kappa_edge(dag, u, v)
            sum_rho += rho_mem_node(dag, v)

    if edge_count == 0 or sum_rho <= 0.0:
        return None

    k_R = sum_kappa / edge_count
    rho_R = sum_rho / edge_count
    g_R = k_R / rho_R

    return (k_R, rho_R, g_R)


def estimate_g_star(dag: DiGraph, scales: List[int] = [1, 2, 4, 8]) -> Dict:
    """
    Estimate the RG fixed point g* from block-averaged g_R across multiple scales.

    For each R in `scales`:

        (κ_R, ρ_R, g_R) = block_averages(dag, R)

    We collect valid g_R’s and classify:

      - No valid values:
            status = FAIL_NO_DATA, g_star = None
      - Exactly one:
            status = CONVERGED, g_star = that value
      - Two or more:
            let g_prev, g_last be the last two (largest R’s),
            drift = |g_last - g_prev| / (|g_prev| + eps)

            if drift > 0.2:
                status = UNSTABLE
            else:
                status = CONVERGED

    Returns:
        {
          "g_star": float | None,
          "status": "CONVERGED" | "UNSTABLE" | "FAIL_NO_DATA",
          "flow":   {R: g_R or None},
          "drift":  float (only if ≥ 2 valid values)
        }
    """
    flow: Dict[int, Optional[float]] = {}
    valid: List[float] = []

    for R in scales:
        res = block_averages(dag, R)
        if res is None:
            flow[R] = None
        else:
            k_R, rho_R, g_R = res
            flow[R] = g_R
            valid.append(g_R)

    if not valid:
        return {
            "g_star": None,
            "status": "FAIL_NO_DATA",
            "flow": flow,
        }

    if len(valid) == 1:
        return {
            "g_star": valid[0],
            "status": "CONVERGED",
            "flow": flow,
        }

    g_last = valid[-1]
    g_prev = valid[-2]
    denom = abs(g_prev) + 1e-9
    drift = abs(g_last - g_prev) / denom

    status = "UNSTABLE" if drift > 0.2 else "CONVERGED"

    result = {
        "g_star": g_last,
        "status": status,
        "flow": flow,
        "drift": drift,
    }
    return result


def alpha_from_g_star(
    g_star: float,
    D: int = 4,
    q: int = 2,
) -> float:
    """
    Compute the effective α_est from g_star using the normalization:

        α ≡ π g* / (9 D^2 q)

    with defaults D = 4, q = 2 as in the electromagnetic normalization
    in the paper (§11.5).

    IMPORTANT:
      - This is used only in analysis scripts for *descriptive* comparison;
        it must never be used to tune parameters toward 137.
    """
    if D <= 0 or q <= 0:
        raise ValueError("D and q must be positive.")
    return math.pi * g_star / (9.0 * (D ** 2) * float(q))
