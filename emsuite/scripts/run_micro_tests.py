# emsuite/scripts/run_micro_tests.py
import json

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.core.projection import project_to_consistent
from emsuite.evolution.selection import measure_topology
from emsuite.physics.entropy import entropy_growth_rate


def build_toy_graph() -> DiGraph:
    g = DiGraph()
    for v in ["i", "v", "w", "x", "z"]:
        g.add_vertex(v)
    g.add_edge("i", "v")
    g.add_edge("i", "w")
    g.add_edge("v", "x")
    g.add_edge("w", "x")
    g.add_edge("x", "z")
    return g


def main():
    g = build_toy_graph()
    topo = measure_topology(g)

    data = {
        "topology": topo,
    }

    # projection test
    fixed_serials = set()
    for seed in range(5):
        grammar = Grammar.random_binary(g, seed=seed)
        proj = project_to_consistent(grammar)
        serial = "|".join(
            f"{v}:{sorted(rule.table.items())}"
            for v, rule in proj.rules.items()
        )
        fixed_serials.add(serial)

    data["projection_fixed_serials_count"] = len(fixed_serials)

    # entropy growth on full visible set
    grammar = Grammar.random_binary(g, seed=123)
    vis = set(g.vertices())
    rate = entropy_growth_rate(grammar, vis, steps=3, seed=456, method="local")
    data["entropy_growth_rate_local"] = rate

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
