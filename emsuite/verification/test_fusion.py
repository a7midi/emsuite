# emsuite/verification/test_fusion.py
from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.physics.entropy import entropy_growth_rate


def test_entropy_non_decrease_under_visible_enlargement():
    """
    Toy check: if we enlarge the visible set, local entropy growth
    (via local proxy) should not decrease on average.
    """
    g = DiGraph()
    for v in range(4):
        g.add_vertex(v)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)

    grammar = Grammar.random_binary(g, seed=0)

    vis_small = {0, 1}
    vis_large = {0, 1, 2, 3}

    r_small = entropy_growth_rate(grammar, vis_small, steps=3, seed=1, method="local")
    r_large = entropy_growth_rate(grammar, vis_large, steps=3, seed=1, method="local")

    assert r_large >= r_small - 1e-9
