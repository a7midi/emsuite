# emsuite/verification/test_em_law.py
from emsuite.core.graph import DiGraph
from emsuite.physics.geometry import block_averages, estimate_g_star


def simple_layered_dag() -> DiGraph:
    g = DiGraph()
    for i in range(6):
        g.add_vertex(i)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    return g


def test_block_averages_and_g_star():
    g = simple_layered_dag()
    dag, _ = g.condensation()

    res1 = block_averages(dag, R=1)
    assert res1 is not None
    k1, rho1, g1 = res1
    assert rho1 >= 0

    res = estimate_g_star(dag, scales=[1, 2])
    assert res["g_star"] is None or isinstance(res["g_star"], float)
    assert res["status"] in {"CONVERGED", "UNSTABLE", "FAIL_NO_DATA"}
