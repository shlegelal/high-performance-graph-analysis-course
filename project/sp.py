import math
import heapq
from typing import Collection, Hashable
import pygraphblas as gb
import networkx as nx

from project.utils import _is_correct_input, _is_correct_adj


def sssp(adj: gb.Matrix, start: int) -> list[int]:
    """
    Finds single-source the shortest paths using an algebraic Bellman-Ford algorithm.
    Assume that all vertices are reachable from themselves
    :param adj: Bool adjacency matrix of the graph with edge lengths, where a no-value
    is treated as unreachable. The graph should not have negative-length cycles
    :param start: Index of start vertex in adjacency matrix
    :return: List lengths of the sortest path from start to the corresponding vertex,
    or *inf* if it is unreachable.
    """
    return mssp(adj, [start])[0][1]


def mssp(adj: gb.Matrix, starts: Collection[int]) -> list[tuple[int, list[int]]]:
    """
    Finds multiple-source the shortest paths using an algebraic Bellman-Ford algorithm.
    Assume that all vertices are reachable from themselves
    :param adj: Bool adjacency matrix of the graph with edge lengths, where a no-value
    is treated as unreachable. The graph should not have negative-length cycles
    :param starts: Indexes of start vertexes in adjacency matrix
    :return: list of start and distances where for each start vertex there is a list
    of lengths of the sortest path from this start to the corresponding vertex,
    or *inf* if it is unreachable.
    """
    _is_correct_input(adj, starts, gb.FP64)
    adj = adj.eadd(gb.Matrix.identity(gb.FP64, adj.nrows, 0.0), gb.FP64.MIN)

    dists = gb.Matrix.sparse(gb.FP64, nrows=len(starts), ncols=adj.ncols)
    for row, start in enumerate(starts):
        dists[row, start] = 0

    for _ in range(adj.ncols):
        old_dists = dists
        dists = dists.mxm(adj, gb.FP64.MIN_PLUS)
        if old_dists.iseq(dists):
            return [
                (
                    start,
                    [dists.get(row, col, default=math.inf) for col in range(adj.ncols)],
                )
                for row, start in enumerate(starts)
            ]

    raise ValueError("Graph has a reachable negative-length cycle")


def apsp(adj: gb.Matrix) -> list[tuple[int, list[int]]]:
    """
    Finds all-pairs shortest paths using an algebraic Floyd–Warshall algorithm.
    Vertices are reachable from themselves if there is a self-loop
    :param adj: Bool adjacency matrix of the graph with edge lengths, where a no-value
    is treated as unreachable. The graph should not have negative-length cycles
    :return: List of start, distances where for each start vertex there is a list
    of lengths of the sortest path from this start to the corresponding vertex,
    or *inf* if it is unreachable
    """
    _is_correct_adj(adj, gb.FP64)
    adj.select("!=", thunk=math.inf, out=adj)

    dists = adj.dup()

    for k in range(adj.ncols):
        step = dists.extract_matrix(col_index=k).mxm(
            dists.extract_matrix(row_index=k), semiring=gb.FP64.MIN_PLUS
        )
        dists.eadd(step, add_op=gb.FP64.MIN, out=dists)

    for k in range(adj.ncols):
        step = dists.extract_matrix(col_index=k).mxm(
            dists.extract_matrix(row_index=k), semiring=gb.FP64.MIN_PLUS
        )
        if dists.isne(dists.eadd(step, add_op=gb.FP64.MIN)):
            raise ValueError("Graph has a reachable negative-length cycle")

    return [
        (row, [dists.get(row, col, default=math.inf) for col in range(adj.ncols)])
        for row in range(adj.nrows)
    ]


def dijkstra_sssp(graph: nx.Graph, start: Hashable) -> dict[Hashable, float]:
    """
    Finds single-source the shortest paths using Dijkstra's algorithm.
    :param graph: Graph to run the algorithm on. Its edges weight is 1.
    :param start: Start vertex of the graph.
    :return: Dict of vertexes of the graph and lengths of the sortest path from start to the corresponding vertex,
    or *inf* if it is unreachable.
    """
    if not graph.has_node(start):
        raise ValueError(f"The graph does not contain a start vertex")

    dists = {node: math.inf for node in graph.nodes}
    dists[start] = 0
    queue = [(0, start)]

    while queue:
        dist, node = heapq.heappop(queue)
        if dist > dists[node]:
            continue

        for neighbor in graph.neighbors(node):
            new_dist = dists[node] + 1
            if new_dist < dists[neighbor]:
                dists[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))

    return dists
