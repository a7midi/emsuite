# emsuite/physics/entropy.py
from __future__ import annotations
from typing import Dict, Set, List, Tuple, Hashable, Iterable, Optional
from itertools import product
import math
import random

from emsuite.core.graph import Vertex
from emsuite.core.grammar import Grammar, Symbol
from emsuite.core.dynamics import synchronous_orbit

State = Dict[Vertex, Symbol]


def _all_assignments(
    vars_: List[Vertex],
    alphabets: Dict[Vertex, Tuple[Symbol, ...]],
) -> Iterable[State]:
    """
    Enumerate all assignments on vars_ using their local alphabets.
    Intended only for very small graphs (N <= ~10).
    """
    if not vars_:
        yield {}
        return
    symbols_lists = [alphabets[v] for v in vars_]
    for combo in product(*symbols_lists):
        yield {v: s for v, s in zip(vars_, combo)}


def fiber_size_full(
    grammar: Grammar,
    visible_nodes: Set[Vertex],
    t: int,
    observed_history: List[Dict[Vertex, Symbol]],
) -> int:
    """
    Full-history fibre size |M_t| as defined in the paper:

    Number of *global* hidden trajectories (states on V\visible_nodes at
    times 0..t) consistent with a fixed visible history on visible_nodes.

    Here we brute-force by enumerating all global initial states consistent
    with observed_history[0] on visible_nodes and simulating forward t steps.
    This is only feasible for very small substrates.
    """
    assert len(observed_history) == t + 1
    V = grammar.graph.vertices()
    visibles = set(visible_nodes)
    hiddens = [v for v in V if v not in visibles]

    alphabets = grammar.alphabets
    count = 0

    for hidden_init in _all_assignments(hiddens, alphabets):
        # build initial global state consistent with visible t=0
        init: State = {}
        for v in V:
            if v in visibles:
                init[v] = observed_history[0][v]
            else:
                init[v] = hidden_init[v]
        _, history = grammar.orbit(init, steps=t)
        # compare visible trajectories
        ok = True
        for tau in range(t + 1):
            visible_state = {v: history[tau][v] for v in visibles}
            if visible_state != observed_history[tau]:
                ok = False
                break
        if ok:
            count += 1
    return count


def fiber_size_local(
    grammar: Grammar,
    visible_nodes: Set[Vertex],
    prev_visible: Dict[Vertex, Symbol],
    curr_visible: Dict[Vertex, Symbol],
) -> int:
    """
    Local/Markovian fibre size |M_{t-1 -> t}|:

    Number of hidden assignments at t-1 consistent with seeing prev_visible
    at t-1 and curr_visible at t after one synchronous update.

    Implementation: enumerate hidden assignments at t-1 only, simulate one
    step, and count how many give the requested (prev, curr) visible slice.
    """
    V = grammar.graph.vertices()
    visibles = set(visible_nodes)
    hiddens = [v for v in V if v not in visibles]
    alphabets = grammar.alphabets

    count = 0
    for hidden_prev in _all_assignments(hiddens, alphabets):
        prev_state: State = {}
        for v in V:
            if v in visibles:
                prev_state[v] = prev_visible[v]
            else:
                prev_state[v] = hidden_prev[v]
        next_state = grammar.update(prev_state)

        if all(next_state[v] == curr_visible[v] for v in visibles):
            count += 1
    return count


def observer_entropy_from_fibre_size(fibre_size: int) -> float:
    """
    S_t := ceil(log2 |M_t|). Returns float for convenience.
    """
    if fibre_size <= 0:
        return 0.0
    return float(math.ceil(math.log2(fibre_size)))


def entropy_growth_rate(
    grammar: Grammar,
    visible_nodes: Set[Vertex],
    steps: int = 3,
    seed: int = 42,
    method: str = "local",
) -> float:
    """
    Approximate average per-tick entropy increment ΔS over 'steps' ticks.

    method='full':
        uses full-history fibre sizes (expensive; only for tiny graphs).
    method='local':
        uses single-step local fibres as a proxy.

    NOTE: As per your spec, method='local' is treated explicitly as a proxy,
    not as the theorem's canonical S_t.
    """
    rnd = random.Random(seed)
    init_state = grammar.random_state(seed=rnd.randint(0, 10**9))
    history = synchronous_orbit(grammar, init_state, steps=steps)

    visibles = set(visible_nodes)
    if not visibles:
        return 0.0

    Ss: List[float] = []
    if method == "full":
        # Build observed visible history
        observed_hist = [
            {v: state[v] for v in visibles} for state in history
        ]
        for t in range(1, steps + 1):
            size = fiber_size_full(
                grammar=grammar,
                visible_nodes=visibles,
                t=t,
                observed_history=observed_hist[: t + 1],
            )
            Ss.append(observer_entropy_from_fibre_size(size))
    elif method == "local":
        S_prev = None
        for t in range(steps):
            prev_vis = {v: history[t][v] for v in visibles}
            curr_vis = {v: history[t + 1][v] for v in visibles}
            size = fiber_size_local(
                grammar=grammar,
                visible_nodes=visibles,
                prev_visible=prev_vis,
                curr_visible=curr_vis,
            )
            S_curr = observer_entropy_from_fibre_size(size)
            Ss.append(S_curr)
            S_prev = S_curr
    else:
        raise ValueError(f"Unknown method {method!r}")

    if len(Ss) < 2:
        return 0.0
    # average ΔS over adjacent ticks
    deltas = [Ss[i + 1] - Ss[i] for i in range(len(Ss) - 1)]
    return sum(deltas) / len(deltas)
