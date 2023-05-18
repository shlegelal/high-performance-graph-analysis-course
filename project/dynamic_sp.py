from typing import Hashable
import networkx as nx
import math
from boltons.queueutils import HeapPriorityQueue
import itertools

from project.sp import dijkstra_sssp


class DynamicSSSP:
    """
    Dynamic single-source the shortest paths algorithm, that uses the classic Dijkstra algorithm as its basis,
    based on article "An Incremental Algorithm for a Generalization of the Shortest-Path Problem"
    https://doi.org/10.1006/jagm.1996.0046.
    """

    def __init__(self, graph: nx.DiGraph, start: Hashable):
        """
        Initializes the algorithm to run on the provided graph and start vertex.
        :param graph: Graph to run the algorithm on. Its edges weight is 1.
        :param start: Start vertex of the graph.
        """
        self._graph = graph
        self._start = start
        self._dists = dijkstra_sssp(graph, start)
        self._modified = set()

    def increment(self, u: Hashable, v: Hashable):
        """
        Insert the edge in the graph.
        """
        self._graph.add_edge(u, v)
        self._modified.add(v)

        # If add non-existent vertices
        if u not in self._dists:
            self._dists[u] = math.inf
        if v not in self._dists:
            self._dists[v] = math.inf

    def decrement(self, u: Hashable, v: Hashable):
        """
        Delete the edge from the graph.
        """
        self._graph.remove_edge(u, v)
        self._modified.add(v)

    def query(self) -> dict[Hashable, float]:
        """
        Returns the distances from start to each vertex in the graph, or +inf if a
        vertex is unreachable.
        """
        if len(self._modified) > 0:
            self._update_dists()
            self._modified = set()
        return self._dists

    def _update_dists(self):
        """
        Applies the accumulated graph updates
        """
        rhs: dict[Hashable, float] = {}
        heap = HeapPriorityQueue(priority_key=lambda x: x)
        for u in self._modified:
            rhs[u] = self._get_rhs(u)
            if rhs[u] != self._dists[u]:
                heap.add(u, priority=min(rhs[u], self._dists[u]))

        while heap:
            u = heap.pop()

            if rhs[u] < self._dists[u]:
                self._dists[u] = rhs[u]
                to_update_rhs = self._graph.successors(u)

            else:
                self._dists[u] = math.inf
                to_update_rhs = itertools.chain(self._graph.successors(u), [u])

            for v in to_update_rhs:
                rhs[v] = self._get_rhs(v)
                if rhs[v] != self._dists[v]:
                    heap.add(v, priority=min(rhs[v], self._dists[v]))
                else:
                    if v in heap.entry_map:
                        heap.remove(v)

    def _get_rhs(self, u: Hashable):
        if u == self._start:
            return 0
        return min(
            (self._dists[v] + 1 for v in self._graph.predecessors(u)), default=math.inf
        )
