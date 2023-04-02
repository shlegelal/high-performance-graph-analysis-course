import pygraphblas as gb

from project.utils import _is_correct_input


def bfs(adj: gb.Matrix, start: int) -> list[int]:
    """
    Implementation of BFS algorithm for given directed graph and start vertex
    :param adj: Bool adjacency matrix of the graph
    :param start: Index of start vertex in adjacency matrix
    :return: List with counts of steps from start vertex to others.
        If vertex is not reachable, value in list equals -1.
    """
    _is_correct_input(adj, [start])

    steps = gb.Vector.sparse(gb.INT64, size=adj.nrows)
    front = gb.Vector.sparse(gb.BOOL, size=adj.nrows)

    steps[start] = 0
    front[start] = True
    step = 1

    while front.reduce():
        front.vxm(adj, out=front, mask=steps.S, desc=gb.descriptor.RC)
        steps.assign_scalar(step, mask=front)
        step += 1

    return [steps.get(i, default=-1) for i in range(steps.size)]


def msbfs(adj: gb.Matrix, starts: list[int]) -> list[tuple[int, list[int]]]:
    """
    Implementation of Multi-source BFS algorithm for given directed graph and start vertexes
    :param adj: Bool adjacency matrix of the graph
    :param starts: Indexes of start vertexes in adjacency matrix
    :return: List of start and parents where for each start vertex there is a list
    of parent vertices for the corresponding vertex. If there are several possible
    parent vertices, the one with the smaller index will be picked. The start vertices
    will have -1 in these lists and unreachable vertices will have -2.
    """
    _is_correct_input(adj, starts)

    parents = gb.Matrix.sparse(gb.INT64, nrows=len(starts), ncols=adj.ncols)
    front = gb.Matrix.sparse(gb.INT64, nrows=len(starts), ncols=adj.ncols)
    for row, start in enumerate(starts):
        front[row, start] = -1

    while front.nvals != 0:
        parents.assign(front, mask=front.S)
        front.apply(gb.INT64.POSITIONJ, out=front)
        front.mxm(
            adj,
            semiring=gb.INT64.MIN_FIRST,
            out=front,
            mask=parents,
            desc=gb.descriptor.RSC,
        )

    parents.assign_scalar(-2, mask=parents, desc=gb.descriptor.S & gb.descriptor.C)
    return [(start, list(parents[i, :].vals)) for i, start in enumerate(starts)]
