# emsuite/scripts/run_evolution_demo.py
from __future__ import annotations

import json

from emsuite.evolution.selection import EvolutionConfig, evolve_population


def main() -> None:
    cfg = EvolutionConfig(
        n_nodes=8,
        population_size=32,
        generations=20,
        edge_prob=0.3,
        elite_fraction=0.25,
        cycle_penalty_weight=1.0,
        random_seed=42,
    )

    population, avg_topology = evolve_population(cfg)

    result = {
        "avg_topology_final": avg_topology,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
