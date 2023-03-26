import math
import pygraphblas as gb

from project.utils import _is_correct_adj, _is_undirected


def count_vertex_triangles(adj: gb.Matrix) -> list[int]:
    """
    Counts the number of triangles in which each graph vertex participates
    :param adj: Bool adjacency matrix of the undirected graph
    :return: list where for each vertex specifies how many triangles it participates in
    """
    _is_correct_adj(adj)
    _is_undirected(adj)

    triangles = (adj.mxm(adj, mask=adj.S, semiring=gb.INT64.PLUS_TIMES)).reduce_vector()
    return [math.ceil(triangles.get(i, default=0) / 2) for i in range(triangles.size)]


def cohen_algorithm(adj: gb.Matrix) -> int:
    """
    Cohen's algorithm which calculates the number of triangles of an undirected graph
    :param adj: Bool adjacency matrix of the undirected graph
    :return: The number of unique triangles in the graph. Note: the resulting number of triangles
    may depend on its vertex numeration, if the graph contains self loops
    """
    _is_correct_adj(adj)
    _is_undirected(adj)

    counts = adj.tril().mxm(adj.triu(), semiring=gb.INT64.PLUS_TIMES, mask=adj)
    return math.ceil(counts.reduce_int() / 2)


def sandia_algorithm(adj: gb.Matrix) -> int:
    """
    Sandia algorithm which calculates the number of triangles of an undirected graph
    :param adj: Bool adjacency matrix of the undirected graph
    :return: The number of unique triangles in a graph. Note: self loops is triangle
    """
    _is_correct_adj(adj)
    _is_undirected(adj)

    tril = adj.tril()
    return (tril.mxm(tril, semiring=gb.INT64.PLUS_TIMES, mask=tril)).reduce_int()
