import pygraphblas as gb


def _is_correct_adj(adj: gb.Matrix):
    """
    Check that graph matrix is square, matrix has boolean type
    """
    if not adj.square:
        raise ValueError("Adjacency matrix must be square")
    if adj.type != gb.BOOL:
        raise ValueError(f"Matrix type {adj.type}, expected {gb.BOOL}")


def _is_correct_input(adj: gb.Matrix, starts: list[int]):
    """
    Check that graph matrix is square, matrix has boolean type,
    start vertices are not negative and not more the number of vertices
    """
    _is_correct_adj(adj)
    for start in starts:
        if start < 0 or start >= adj.nrows:
            raise ValueError(f"Start vertex is {start}, expected 0..{adj.nrows}")


def _is_undirected(adj: gb.Matrix):
    """
    Check that a graph is undirected, i.e. its matrix is symmetric.
    """
    for i, j in zip(adj.I, adj.J):
        if adj.get(i, j) != adj.get(j, i):
            raise ValueError(f"Matrix is not symmetric, the edge {i}-{j} is oriented")
