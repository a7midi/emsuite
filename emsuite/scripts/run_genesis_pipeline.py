# emsuite/scripts/run_genesis_pipeline.py
from __future__ import annotations

import json

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.core.projection import project_to_consistent
from emsuite.evolution.selection import EvolutionConfig, evolve_population
from emsuite.analysis.ensembles import MicroTopologyStats, macro_layered_dag_from_stats
from emsuite.physics.geometry import estimate_g_star, alpha_from_g_star


def _small_diamond_graph() -> DiGraph:
    """
    Construct a tiny 'diamond' graph:

        i -> v, i -> w, v -> x, w -> x, plus a tail node z.

    Used for law/projection tests in Epoch 1.
    """
    g = DiGraph()
    for v in ["i", "v", "w", "x", "z"]:
        g.add_vertex(v)
    g.add_edge("i", "v")
    g.add_edge("i", "w")
    g.add_edge("v", "x")
    g.add_edge("w", "x")
    g.add_edge("x", "z")
    return g


def epoch1_law(num_seeds: int = 8) -> dict:
    """
    Epoch 1 – 'Law': demonstrate that many random grammars on a fixed
    diamond graph flow to the same projected rule under P.
    """
    g = _small_diamond_graph()
    fixed_serials = set()

    def serialise(grammar: Grammar) -> str:
        parts = []
        for v in sorted(grammar.rules.keys(), key=str):
            rule = grammar.rules[v]
            items = sorted(rule.table.items(), key=lambda kv: kv[0])
            parts.append(
                f"{v}:" + "[" + ",".join(f"({inp}->{out})" for inp, out in items) + "]"
            )
        return "|".join(parts)

    for seed in range(num_seeds):
        base = Grammar.random_binary(graph=g, seed=seed)
        proj = project_to_consistent(base)
        fixed_serials.add(serialise(proj))

    return {
        "projection_fixed_grammars_count": len(fixed_serials),
    }


def epoch2_time() -> dict:
    """
    Epoch 2 – 'Time': run small-N evolution with entropy-based fitness
    and measure the emergent micro-topology.
    """
    cfg = EvolutionConfig(
        n_nodes=8,
        population_size=32,
        generations=20,
        edge_prob=0.3,
        elite_fraction=0.25,
        cycle_penalty_weight=1.0,
        random_seed=321,
    )

    population, avg_topology = evolve_population(cfg)
    return {
        "avg_topology_after_selection": avg_topology,
        "population": population,  # not JSON-serialisable; not printed directly
    }


def epoch3_geometry(avg_topology: dict, N_macro: int = 200, seed: int = 0) -> dict:
    """
    Epoch 3 – 'Geometry': build macro DAGs informed by the micro topology,
    run the Einstein-Memory RG, and report g* and α_est.
    """
    micro_stats = MicroTopologyStats(**avg_topology)

    # Single representative macro graph for the pipeline demo
    G_macro = macro_layered_dag_from_stats(micro_stats, N=N_macro, seed=seed)
    dag, _ = G_macro.condensation()

    g_info = estimate_g_star(dag, scales=[1, 2, 4, 8])

    g_star = g_info.get("g_star", None)
    alpha_est = None
    if g_star is not None and g_star > 0:
        alpha_est = alpha_from_g_star(g_star, D=4, q=2)

    # Physical fine-structure constant for reference ONLY.
    alpha_phys = 1.0 / 137.035999084  # PDG 2018 value (approx)

    return {
        "g_flow": g_info,
        "alpha_est": alpha_est,
        "alpha_empirical_reference": alpha_phys,
    }


def main() -> None:
    law_info = epoch1_law()
    time_info = epoch2_time()
    avg_topology = time_info["avg_topology_after_selection"]
    geom_info = epoch3_geometry(avg_topology)

    result = {
        "epoch1_law": law_info,
        "epoch2_time": {
            "avg_topology_after_selection": avg_topology,
        },
        "epoch3_geometry": geom_info,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
