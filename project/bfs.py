import pygraphblas as gb


def bfs(adj: gb.Matrix, start: int) -> list[int]:
    """
    Implementation of BFS algorithm for given directed graph and start vertex
    :param adj: Bool adjacency matrix of the graph
    :param start: Index of starting vertex in adjacency matrix
    :return: List with counts of steps from start vertex to others.
        If vertex is not reachable, value in list equals -1.
    """
    _is_correct_matrix(adj, start)

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


def _is_correct_matrix(adj: gb.Matrix, start: int):
    """
    Check that graph matrix is square, matrix has boolean type,
    start vertices are not negative and not more the number of vertices
    """
    if not adj.square:
        raise ValueError("Adjacency matrix must be square")
    if adj.type != gb.BOOL:
        raise ValueError(f"Matrix type {adj.type}, expected {gb.BOOL}")
    if start < 0 or start >= adj.nrows:
        raise ValueError(f"Start vertex is {start}, expected 0..{adj.nrows}")
