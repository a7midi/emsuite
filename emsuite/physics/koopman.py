# emsuite/physics/koopman.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Hashable, Iterable
import cmath
import math
import random

from emsuite.core.graph import DiGraph, Vertex
from emsuite.core.grammar import Grammar, Symbol


Phase = complex  # element of µ_N


def assign_phases(
    grammar: Grammar,
    N: int = 8,
    seed: int = 0,
) -> Dict[Tuple[Vertex, Symbol], Phase]:
    """
    Assign injective phase tags φ_v: A_v -> µ_N for each vertex v.

    Implementation: randomly assign distinct roots of unity to symbols at each site.
    """
    rnd = random.Random(seed)
    phase_map: Dict[Tuple[Vertex, Symbol], Phase] = {}
    for v, alphabet in grammar.alphabets.items():
        used: Dict[Symbol, Phase] = {}
        for s in alphabet:
            m = rnd.randrange(N)
            phase = cmath.exp(2j * math.pi * m / N)
            used[s] = phase
        for s, phase in used.items():
            phase_map[(v, s)] = phase
    return phase_map


def path_weights(
    graph: DiGraph,
    grammar: Grammar,
    phase_map: Dict[Tuple[Vertex, Symbol], Phase],
    state: Dict[Vertex, Symbol],
    max_length: int,
) -> Dict[Tuple[Vertex, ...], Phase]:
    """
    Compute path weights w(γ) = ∏ φ_v(a_v) for all directed paths up to length max_length.

    This ignores time-dependence for simplicity: we use a fixed state.
    """
    weights: Dict[Tuple[Vertex, ...], Phase] = {}

    def dfs(path: List[Vertex], length: int):
        if length > max_length:
            return
        v = path[-1]
        # weight of current path
        w = 1+0j
        for u in path:
            w *= phase_map[(u, state[u])]
        weights[tuple(path)] = w
        for wv in graph.successors(v):
            if length < max_length:
                dfs(path + [wv], length + 1)

    for v in graph.vertices():
        dfs([v], 0)

    return weights


@dataclass
class CHSHConfig:
    """
    Minimal CHSH experiment configuration based on path amplitudes.

    A, B: disjoint sets of vertices representing the two parties.
    settings: labels a0,a1,b0,b1 mapped to subsets of edges/paths.
    """
    A: List[Vertex]
    B: List[Vertex]
    settings_A: Dict[str, List[Tuple[Vertex, ...]]]
    settings_B: Dict[str, List[Tuple[Vertex, ...]]]


def chsh_value_from_paths(
    weights: Dict[Tuple[Vertex, ...], Phase],
    config: CHSHConfig,
) -> float:
    """
    Given path weights and a CHSHConfig that associates measurement
    settings to sets of paths, compute an effective CHSH S value.

    This is a deliberately simple toy: for each pair of settings (a,b),
    we define an "observable" as sign(Re amplitude), and treat its ±1
    expectation as the average over included paths.

    This does *not* try to reproduce the full analytical operator bound;
    it's a numerical playground to see if constructed experiments
    accidentally violate Tsirelson.
    """
    def obs(paths: List[Tuple[Vertex, ...]]) -> float:
        if not paths:
            return 0.0
        vals = []
        for p in paths:
            amp = weights.get(p, 0+0j)
            vals.append(1.0 if amp.real >= 0 else -1.0)
        return sum(vals) / len(vals)

    # define four correlators E(a,b)
    A = config.settings_A
    B = config.settings_B

    def E(a: str, b: str) -> float:
        # naive: treat overlapping path sets as samples
        paths_a = A[a]
        paths_b = B[b]
        # For simplicity, pairwise product of signs on each side's amplitude
        # using the same index where possible.
        m = min(len(paths_a), len(paths_b))
        if m == 0:
            return 0.0
        vals = []
        for i in range(m):
            pa = paths_a[i]
            pb = paths_b[i]
            sa = 1.0 if weights.get(pa, 0+0j).real >= 0 else -1.0
            sb = 1.0 if weights.get(pb, 0+0j).real >= 0 else -1.0
            vals.append(sa * sb)
        return sum(vals) / len(vals)

    S = E("a0", "b0") + E("a0", "b1") + E("a1", "b0") - E("a1", "b1")
    return float(S)
