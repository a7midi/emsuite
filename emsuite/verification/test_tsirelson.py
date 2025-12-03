# emsuite/verification/test_tsirelson.py
import math
from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.physics.koopman import assign_phases, path_weights, CHSHConfig, chsh_value_from_paths


def tiny_diamond_graph() -> DiGraph:
    g = DiGraph()
    for v in ["s", "a0", "a1", "b0", "b1"]:
        g.add_vertex(v)
    g.add_edge("s", "a0")
    g.add_edge("s", "a1")
    g.add_edge("s", "b0")
    g.add_edge("s", "b1")
    return g


def test_tsirelson_not_violated_in_toy_model():
    g = tiny_diamond_graph()
    grammar = Grammar.random_binary(g, seed=0)
    phases = assign_phases(grammar, N=16, seed=1)
    state = grammar.random_state(seed=2)
    weights = path_weights(g, grammar, phases, state, max_length=1)

    # Simple CHSH config: treat each outgoing edge as a "path"
    config = CHSHConfig(
        A=["a0", "a1"],
        B=["b0", "b1"],
        settings_A={
            "a0": [("s", "a0")],
            "a1": [("s", "a1")],
        },
        settings_B={
            "b0": [("s", "b0")],
            "b1": [("s", "b1")],
        },
    )

    S = chsh_value_from_paths(weights, config)
    assert abs(S) <= 2.8284271247 + 1e-6  # 2√2 + small tolerance
