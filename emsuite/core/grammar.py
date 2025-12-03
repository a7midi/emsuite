# emsuite/core/grammar.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Hashable, Any, Mapping
import random

from .graph import DiGraph, Vertex


Symbol = Hashable
InputTuple = Tuple[Symbol, ...]


@dataclass
class LocalRule:
    """
    Local update rule:

        preds: ordered tuple of predecessor vertices
        table: mapping from input-symbol tuples to output symbol

    No probabilistic semantics here: pure deterministic lookup.
    """
    preds: Tuple[Vertex, ...]
    table: Dict[InputTuple, Symbol]

    def eval(self, state: Mapping[Vertex, Symbol]) -> Symbol:
        key = tuple(state[p] for p in self.preds)
        try:
            return self.table[key]
        except KeyError:
            # Should not happen if Grammar is well-formed
            raise KeyError(f"Missing table entry for input {key!r} at preds {self.preds!r}")


@dataclass
class Grammar:
    """
    Deterministic synchronous grammar on a fixed graph.

    This object *does not know* about projection/fusion/entropy;
    it just updates symbols.
    """
    graph: DiGraph
    rules: Dict[Vertex, LocalRule]
    alphabets: Dict[Vertex, Tuple[Symbol, ...]]

    @classmethod
    def random_binary(cls, graph: DiGraph, seed: int = 0) -> "Grammar":
        """
        Assigns each node alphabet {0,1} and a random truth table
        over its predecessors. Predecessors are sorted by vertex id
        to form a canonical order.
        """
        rnd = random.Random(seed)
        alphabets: Dict[Vertex, Tuple[Symbol, ...]] = {}
        rules: Dict[Vertex, LocalRule] = {}

        for v in graph.vertices():
            alphabet = (0, 1)
            alphabets[v] = alphabet
            preds = tuple(sorted(graph.predecessors(v), key=repr))
            table: Dict[InputTuple, Symbol] = {}

            if preds:
                domain_sizes = [len(alphabet)] * len(preds)
                # iterate over all input tuples in lexicographic order
                def build_keys(idx: int, prefix: Tuple[int, ...]):
                    if idx == len(preds):
                        yield prefix
                        return
                    for val in range(domain_sizes[idx]):
                        yield from build_keys(idx + 1, prefix + (val,))

                for key_idx in build_keys(0, ()):
                    # Map integer indices to actual symbols (here just 0/1)
                    input_tuple = tuple(alphabet[i] for i in key_idx)
                    table[input_tuple] = rnd.choice(alphabet)
            else:
                # no predecessors: constant rule
                table[()] = rnd.choice(alphabet)

            rules[v] = LocalRule(preds=preds, table=table)

        return cls(graph=graph, rules=rules, alphabets=alphabets)

    # --- state utilities ---

    def random_state(self, seed: int = 0) -> Dict[Vertex, Symbol]:
        rnd = random.Random(seed)
        return {v: rnd.choice(self.alphabets[v]) for v in self.graph.vertices()}

    def update(self, state: Dict[Vertex, Symbol]) -> Dict[Vertex, Symbol]:
        """
        One synchronous tick: applies all local rules.
        """
        new_state: Dict[Vertex, Symbol] = {}
        for v, rule in self.rules.items():
            new_state[v] = rule.eval(state)
        return new_state

    def orbit(self, state: Dict[Vertex, Symbol], steps: int) -> Tuple[Dict[Vertex, Symbol], list]:
        """
        Simulate synchronous evolution for 'steps' ticks.
        Returns (final_state, history), where history includes the initial state.
        """
        history = [state]
        cur = state
        for _ in range(steps):
            cur = self.update(cur)
            history.append(cur)
        return cur, history
