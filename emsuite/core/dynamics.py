# emsuite/core/dynamics.py
from __future__ import annotations
from typing import Dict, Hashable, Iterable, Callable, List
import random

from .grammar import Grammar, Symbol
from .graph import Vertex


State = Dict[Vertex, Symbol]
Schedule = Iterable[Vertex]


def synchronous_orbit(grammar: Grammar, init: State, steps: int) -> List[State]:
    """
    Pure synchronous orbit (already available via Grammar.orbit, but kept here
    to mirror a future async schedule implementation).
    """
    _, history = grammar.orbit(init, steps)
    return history


def asynchronous_orbit(
    grammar: Grammar,
    init: State,
    steps: int,
    schedule_fn: Callable[[int, Grammar], Schedule] | None = None,
) -> List[State]:
    """
    Asynchronous update: at tick t, only vertices in schedule_fn(t, grammar)
    are updated; others keep their tag.

    By default, schedule_fn selects a random permutation of vertices each tick.

    The paper argues schedule is a gauge choice; we keep this here to let you
    test schedule-independence numerically.
    """
    if schedule_fn is None:
        def schedule_fn(_t: int, gr: Grammar) -> Schedule:
            vs = gr.graph.vertices()
            rnd = random.Random(_t)
            rnd.shuffle(vs)
            return vs

    state: State = dict(init)
    history: List[State] = [dict(state)]
    for t in range(steps):
        new_state = dict(state)
        for v in schedule_fn(t, grammar):
            new_state[v] = grammar.rules[v].eval(state)
        state = new_state
        history.append(dict(state))
    return history
