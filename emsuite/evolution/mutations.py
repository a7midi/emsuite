# emsuite/evolution/mutations.py
from __future__ import annotations
from typing import Tuple, Dict
import random

from emsuite.core.graph import DiGraph, Vertex
from emsuite.core.grammar import Grammar, LocalRule, Symbol, InputTuple


def mutate_graph_edges(
    graph: DiGraph,
    edge_flip_prob: float,
    seed: int,
) -> DiGraph:
    """
    Simple graph mutation: with probability edge_flip_prob, flip the presence
    of each potential edge between existing vertices.

    Intended for *small* graphs only; for larger graphs you may want sparse
    moves (add/remove a few edges only).
    """
    rnd = random.Random(seed)
    vs = graph.vertices()
    new = DiGraph()
    for v in vs:
        new.add_vertex(v)

    existing = set(graph.edges())
    for u in vs:
        for v in vs:
            if u == v:
                continue
            has_edge = (u, v) in existing
            if rnd.random() < edge_flip_prob:
                has_edge = not has_edge
            if has_edge:
                new.add_edge(u, v)
    return new


def mutate_grammar_tables(
    grammar: Grammar,
    flip_prob: float,
    seed: int,
) -> Grammar:
    """
    Flip outputs in local tables with probability flip_prob (per entry).
    """
    rnd = random.Random(seed)
    new_rules: Dict[Vertex, LocalRule] = {}
    for v, rule in grammar.rules.items():
        alphabet = grammar.alphabets[v]
        table = {}
        for inp, out in rule.table.items():
            if rnd.random() < flip_prob:
                # flip to a *different* symbol in alphabet
                alternatives = [s for s in alphabet if s != out]
                table[inp] = rnd.choice(alternatives) if alternatives else out
            else:
                table[inp] = out
        new_rules[v] = LocalRule(preds=rule.preds, table=table)
    return Grammar(graph=grammar.graph, rules=new_rules, alphabets=grammar.alphabets)
