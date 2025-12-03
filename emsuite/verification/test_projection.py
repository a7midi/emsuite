# emsuite/verification/test_projection.py
import copy

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.core.projection import project_to_consistent


def small_diamond_graph() -> DiGraph:
    g = DiGraph()
    # 5-vertex toy of the paper: i -> v,w; v,w -> x; x->z
    for v in ["i", "v", "w", "x", "z"]:
        g.add_vertex(v)
    g.add_edge("i", "v")
    g.add_edge("i", "w")
    g.add_edge("v", "x")
    g.add_edge("w", "x")
    g.add_edge("x", "z")
    return g


def canonical_serialisation(grammar: Grammar) -> str:
    pieces = []
    for v in sorted(grammar.rules.keys(), key=repr):
        rule = grammar.rules[v]
        entries = sorted(rule.table.items(), key=lambda kv: kv[0])
        pieces.append(f"{v}:{entries}")
    return "|".join(pieces)


def test_projection_idempotence_and_attractor():
    g = small_diamond_graph()
    fixed_serials = set()

    for seed in range(5):
        base = Grammar.random_binary(graph=g, seed=seed)
        proj1 = project_to_consistent(copy.deepcopy(base))
        proj2 = project_to_consistent(copy.deepcopy(proj1))

        # idempotence
        assert canonical_serialisation(proj1) == canonical_serialisation(proj2)

        fixed_serials.add(canonical_serialisation(proj1))

    # many seeds should flow to a very small number of fixed grammars
    assert len(fixed_serials) <= 3
