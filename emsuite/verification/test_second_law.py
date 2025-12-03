# emsuite/verification/test_second_law.py
from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.physics.entropy import entropy_growth_rate


def test_entropy_growth_positive_on_random_graph():
    g = DiGraph()
    for i in range(5):
        g.add_vertex(i)
    # make a small branching DAG
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(3, 4)

    grammar = Grammar.random_binary(g, seed=123)

    vis = {0, 1, 2, 3, 4}
    rate_local = entropy_growth_rate(grammar, vis, steps=3, seed=5, method="local")
    # Very weak: just ensure we don't get a negative rate
    assert rate_local >= -1e-9
