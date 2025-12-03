# emsuite/core/graph.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Hashable, Iterable, Tuple, Deque, Optional
from collections import defaultdict, deque

Vertex = Hashable


@dataclass
class DiGraph:
    """
    Simple directed multigraph (edges have no IDs; multiplicity can be
    encoded separately if needed).

    This layer is *pure combinatorics*. No physics.
    """
    _succ: Dict[Vertex, Set[Vertex]] = field(default_factory=lambda: defaultdict(set))
    _pred: Dict[Vertex, Set[Vertex]] = field(default_factory=lambda: defaultdict(set))

    # --- basic construction ---

    def add_vertex(self, v: Vertex) -> None:
        self._succ.setdefault(v, set())
        self._pred.setdefault(v, set())

    def add_edge(self, u: Vertex, v: Vertex) -> None:
        self.add_vertex(u)
        self.add_vertex(v)
        self._succ[u].add(v)
        self._pred[v].add(u)

    # --- accessors ---

    def vertices(self) -> List[Vertex]:
        return list(self._succ.keys())

    def successors(self, v: Vertex) -> Set[Vertex]:
        return self._succ.get(v, set())

    def predecessors(self, v: Vertex) -> Set[Vertex]:
        return self._pred.get(v, set())

    def out_degree(self, v: Vertex) -> int:
        return len(self._succ.get(v, ()))

    def in_degree(self, v: Vertex) -> int:
        return len(self._pred.get(v, ()))

    def edges(self) -> List[Tuple[Vertex, Vertex]]:
        return [(u, v) for u, succs in self._succ.items() for v in succs]

    # --- SCC + condensation DAG ---

    def strongly_connected_components(self) -> List[Set[Vertex]]:
        """
        Tarjan SCC algorithm.
        Returns a list of SCCs (each a set of vertices).
        """
        index = 0
        stack: List[Vertex] = []
        indices: Dict[Vertex, int] = {}
        lowlink: Dict[Vertex, int] = {}
        on_stack: Set[Vertex] = set()
        sccs: List[Set[Vertex]] = []

        def strongconnect(v: Vertex) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in self.successors(v):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                comp: Set[Vertex] = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    comp.add(w)
                    if w == v:
                        break
                sccs.append(comp)

        for v in self.vertices():
            if v not in indices:
                strongconnect(v)

        return sccs

    def condensation(self) -> Tuple["DiGraph", Dict[Vertex, int]]:
        """
        Contract SCCs to build the condensation DAG.
        Returns (dag, component_map) where component_map[v] = component_id.
        """
        sccs = self.strongly_connected_components()
        comp_id: Dict[Vertex, int] = {}
        for i, comp in enumerate(sccs):
            for v in comp:
                comp_id[v] = i

        dag = DiGraph()
        for i in range(len(sccs)):
            dag.add_vertex(i)
        for u, v in self.edges():
            cu, cv = comp_id[u], comp_id[v]
            if cu != cv:
                dag.add_edge(cu, cv)
        return dag, comp_id

    # --- DAG utilities ---

    def topological_order(self) -> List[Vertex]:
        """
        Kahn's algorithm. Raises ValueError if the graph has a cycle.
        """
        indeg: Dict[Vertex, int] = {v: self.in_degree(v) for v in self.vertices()}
        q: Deque[Vertex] = deque([v for v, d in indeg.items() if d == 0])
        order: List[Vertex] = []

        while q:
            v = q.popleft()
            order.append(v)
            for w in list(self.successors(v)):
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)

        if len(order) != len(indeg):
            raise ValueError("Graph has at least one directed cycle; no topo order exists.")
        return order

    def depth_map(self) -> Dict[Vertex, int]:
        """
        Longest-path depth from sources on a DAG.
        If cycles exist, raises ValueError via topological_order().
        """
        order = self.topological_order()
        depth: Dict[Vertex, int] = {v: 0 for v in order}
        for u in order:
            du = depth[u]
            for w in self.successors(u):
                if depth[w] < du + 1:
                    depth[w] = du + 1
        return depth

    # --- distances & balls ---

    def bfs_distance(self, src: Vertex, directed: bool = False) -> Dict[Vertex, int]:
        """
        BFS distance from src. If directed=False, uses underlying undirected graph.
        """
        seen: Dict[Vertex, int] = {src: 0}
        q: Deque[Vertex] = deque([src])
        while q:
            v = q.popleft()
            dv = seen[v]
            neighbors: Iterable[Vertex]
            if directed:
                neighbors = self.successors(v)
            else:
                neighbors = self.successors(v) | self.predecessors(v)
            for w in neighbors:
                if w not in seen:
                    seen[w] = dv + 1
                    q.append(w)
        return seen

    def ball(self, src: Vertex, r: int, directed: bool = False) -> Set[Vertex]:
        dists = self.bfs_distance(src, directed=directed)
        return {v for v, d in dists.items() if d <= r}

    def average_ball_growth(
        self,
        radii: List[int],
        sample: Optional[int] = None,
        directed: bool = False,
    ) -> Dict[int, float]:
        """
        For each radius r, estimate average |B(src,r)| over a sample of vertices.
        Used later as a dimension estimator.
        """
        vs = self.vertices()
        if not vs:
            return {r: 0.0 for r in radii}
        if sample is not None and sample < len(vs):
            import random
            vs = random.sample(vs, sample)
        growth: Dict[int, float] = {}
        for r in radii:
            total = 0
            for v in vs:
                total += len(self.ball(v, r, directed=directed))
            growth[r] = total / len(vs)
        return growth
