# emsuite/scripts/run_coupling_statistics.py
from __future__ import annotations

import json

from emsuite.evolution.selection import EvolutionConfig, evolve_population
from emsuite.analysis.ensembles import MicroTopologyStats, macro_layered_dag_from_stats
from emsuite.analysis.stats import sample_coupling_distribution


def main() -> None:
    # --- Epoch 2: micro evolution to get topology stats ---
    evo_cfg = EvolutionConfig(
        n_nodes=8,
        population_size=32,
        generations=20,
        edge_prob=0.3,
        elite_fraction=0.25,
        cycle_penalty_weight=1.0,
        random_seed=123,
    )

    population, avg_topology = evolve_population(evo_cfg)
    micro_stats = MicroTopologyStats(**avg_topology)

    # --- Epoch 3: macro ensembles informed by micro_stats ---
    N_macro = 200

    def ensemble_fn(seed: int, N: int) -> "DiGraph":
        return macro_layered_dag_from_stats(micro_stats, N=N, seed=seed)

    res = sample_coupling_distribution(
        ensemble_fn=ensemble_fn,
        ensemble_params={"N": N_macro},
        n_samples=20,
        scales=[1, 2, 4, 8],
    )

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
