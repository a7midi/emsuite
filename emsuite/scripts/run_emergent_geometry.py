# emsuite/scripts/run_emergent_geometry.py
import json

from emsuite.analysis.ensembles import macro_layered_dag_from_stats, MicroStats
from emsuite.physics.geometry import estimate_g_star
from emsuite.analysis.observables import basic_observables


def main():
    # example micro stats (would normally be measured from evolution runs)
    micro = MicroStats(
        avg_out_degree=2.5,
        avg_in_degree=2.5,
        diamond_fraction=0.3,
        cycle_fraction=0.05,
    )

    N = 200
    g = macro_layered_dag_from_stats(micro, N=N, seed=0)
    dag, _ = g.condensation()
    obs = basic_observables(g)

    res = estimate_g_star(dag, scales=[1, 2, 4, 8])

    data = {
        "N": N,
        "observables": obs,
        "g_estimate": res,
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
