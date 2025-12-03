# emsuite/evolution/selection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from emsuite.core.graph import DiGraph, Vertex
from emsuite.core.grammar import Grammar
from emsuite.core.projection import find_diamonds
from emsuite.physics.entropy import entropy_growth_rate



def measure_topology(graph: DiGraph) -> Dict[str, float]:
    """
    Basic topological observables on a finite digraph:

        - avg_out_degree
        - avg_in_degree
        - diamond_fraction: fraction of vertices participating in >=1 diamond
        - cycle_fraction:   fraction of vertices belonging to non-trivial SCCs

    This is the bridge from micro dynamics to macro ensemble parameters.
    """
    verts = list(graph.vertices())
    n = len(verts)
    if n == 0:
        return {
            "avg_out_degree": 0.0,
            "avg_in_degree": 0.0,
            "diamond_fraction": 0.0,
            "cycle_fraction": 0.0,
        }

    # Degrees
    total_out = 0
    total_in = 0
    for v in verts:
        total_out += graph.out_degree(v)
        total_in += graph.in_degree(v)
    avg_out = total_out / n
    avg_in = total_in / n

    # Diamonds: use the diamond finder from the projection layer
    diamonds = find_diamonds(graph)
    if diamonds:
        diamond_nodes = set()
        for d in diamonds:
            diamond_nodes.add(d.i)
            diamond_nodes.add(d.v)
            diamond_nodes.add(d.w)
            diamond_nodes.add(d.x)
        diamond_fraction = len(diamond_nodes) / n
    else:
        diamond_fraction = 0.0

    # Cycles: use condensation to detect non-trivial SCCs
    dag, comp = graph.condensation()
    comp_sizes: Dict[int, int] = {}
    for v, c in comp.items():
        comp_sizes[c] = comp_sizes.get(c, 0) + 1
    nodes_in_cycles = sum(size for size in comp_sizes.values() if size > 1)
    cycle_fraction = nodes_in_cycles / n

    return {
        "avg_out_degree": float(avg_out),
        "avg_in_degree": float(avg_in),
        "diamond_fraction": float(diamond_fraction),
        "cycle_fraction": float(cycle_fraction),
    }


@dataclass
class EvolutionConfig:
    """
    Configuration for small-N evolution experiments.

    These parameters are dimensionless knobs for the *toy* evolution engine.
    They are NOT tuned to match any physical constant; they only control how
    strongly the engine disfavors cycles vs. rewards entropy growth.
    """
    n_nodes: int = 8
    population_size: int = 32
    generations: int = 20
    edge_prob: float = 0.3

    elite_fraction: float = 0.25
    cycle_penalty_weight: float = 1.0

    random_seed: int = 42


@dataclass
class Individual:
    graph: DiGraph
    grammar: Grammar


def _random_graph(n_nodes: int, edge_prob: float, rng: random.Random) -> DiGraph:
    """
    Simple Erdos–Renyi-style digraph on n_nodes with edge_prob for each
    ordered pair u != v. Cycles are allowed; selection will penalise them.
    """
    g = DiGraph()
    # use integer labels 0..n_nodes-1 for convenience
    for v in range(n_nodes):
        g.add_vertex(v)
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v:
                continue
            if rng.random() < edge_prob:
                g.add_edge(u, v)
    return g

def remove_edge(self, u: Vertex, v: Vertex) -> None:
    """
    Remove a directed edge u -> v if it exists.

    No-op if the edge is absent.
    """
    if u in self._succ and v in self._succ[u]:
        self._succ[u].remove(v)
    if v in self._pred and u in self._pred[v]:
        self._pred[v].remove(u)



def _random_grammar_for_graph(graph: DiGraph, rng: random.Random) -> Grammar:
    """
    Wrapper around Grammar.random_binary so we can plug in our own RNG.
    """
    seed = rng.randint(0, 2**31 - 1)
    return Grammar.random_binary(graph, seed=seed)


def _entropy_score(grammar: Grammar, rng: random.Random) -> float:
    """
    Robust wrapper around entropy_growth_rate, using the local proxy by default
    and clamping negatives to zero. This is used purely as a *fitness signal*.
    """
    seed = rng.randint(0, 2**31 - 1)
    # Try to call with full signature; fall back gracefully if needed.
    try:
        val = entropy_growth_rate(grammar, steps=3, seed=seed, method="local")
    except TypeError:
        try:
            val = entropy_growth_rate(grammar, steps=3, seed=seed)
        except TypeError:
            val = entropy_growth_rate(grammar)
    try:
        return max(0.0, float(val))
    except Exception:
        return 0.0


def _mutate_individual(ind: Individual, cfg: EvolutionConfig, rng: random.Random) -> Individual:
    """
    Simple mutation operator:

      - Randomly perturb the graph edges.
      - Re-sample a fresh random grammar consistent with the new graph.

    This avoids delicate bookkeeping for preds/tables and keeps graph+grammar
    consistent at each step.
    """
    g_old = ind.graph
    g_new = DiGraph()
    for v in g_old.vertices():
        g_new.add_vertex(v)
    # Copy edges
    for u in g_old.vertices():
        for v in g_old.successors(u):
            g_new.add_edge(u, v)

    # Edge mutation: flip a small number of edges
    # 50% chance remove a random edge; 50% chance add a random edge.
    if rng.random() < 0.5:
        # remove
        all_edges: List[Tuple[Vertex, Vertex]] = []
        for u in g_new.vertices():
            for v in g_new.successors(u):
                all_edges.append((u, v))
        if all_edges:
            u, v = rng.choice(all_edges)
            g_new.remove_edge(u, v)
    else:
        # add
        verts = list(g_new.vertices())
        if len(verts) >= 2:
            u = rng.choice(verts)
            v = rng.choice(verts)
            if u != v and v not in g_new.successors(u):
                g_new.add_edge(u, v)

    # New grammar for mutated graph
    new_grammar = _random_grammar_for_graph(g_new, rng)
    return Individual(graph=g_new, grammar=new_grammar)


def initialise_population(cfg: EvolutionConfig) -> List[Individual]:
    rng = random.Random(cfg.random_seed)
    pop: List[Individual] = []
    for _ in range(cfg.population_size):
        g = _random_graph(cfg.n_nodes, cfg.edge_prob, rng)
        grammar = _random_grammar_for_graph(g, rng)
        pop.append(Individual(graph=g, grammar=grammar))
    return pop


def evolve_population(cfg: EvolutionConfig) -> Tuple[List[Individual], Dict[str, float]]:
    """
    Run a small-N evolutionary loop:

      - Population of (graph, grammar) individuals.
      - Fitness = entropy_growth_rate(local) - cycle_penalty_weight * cycle_fraction.
      - Selection: keep top elite_fraction, refill via mutation of elites.

    Returns:
      - final population (list of Individuals),
      - averaged topology statistics over the final population.
    """
    rng = random.Random(cfg.random_seed)
    population = initialise_population(cfg)

    for _ in range(cfg.generations):
        scored: List[Tuple[float, Individual, Dict[str, float]]] = []
        for ind in population:
            topo = measure_topology(ind.graph)
            entropy = _entropy_score(ind.grammar, rng)
            fitness = entropy - cfg.cycle_penalty_weight * topo["cycle_fraction"]
            scored.append((fitness, ind, topo))

        # Sort by fitness descending
        scored.sort(key=lambda x: x[0], reverse=True)

        elite_count = max(1, int(cfg.population_size * cfg.elite_fraction))
        elites = scored[:elite_count]

        # Build next generation
        new_pop: List[Individual] = [ind for _, ind, _ in elites]
        while len(new_pop) < cfg.population_size:
            parent = rng.choice(elites)[1]
            child = _mutate_individual(parent, cfg, rng)
            new_pop.append(child)

        population = new_pop

    # Aggregate topology stats over final population
    agg = {
        "avg_out_degree": 0.0,
        "avg_in_degree": 0.0,
        "diamond_fraction": 0.0,
        "cycle_fraction": 0.0,
    }
    for ind in population:
        topo = measure_topology(ind.graph)
        for k in agg.keys():
            agg[k] += topo[k]
    for k in agg.keys():
        agg[k] /= float(len(population))

    return population, agg
