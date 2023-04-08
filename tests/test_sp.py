import math
import random
import pygraphblas as gb
import pytest

import utils
from project.sp import sssp, mssp, apsp


@pytest.mark.parametrize(
    "adj, ssp",
    map(
        lambda r: (
            utils.adj_from_res(r["adj"], gb.types.FP64),
            r["expected"],
        ),
        utils.load_res("test_sp_algorithms"),
    ),
    ids=utils.get_title("test_sp_algorithms"),
)  # 🍉
def test_sp_algorithms(adj: gb.Matrix, ssp: list):
    # apsp 👹
    expected = [
        (start, [dist if dist is not None else math.inf for dist in dists])
        for start, dists in enumerate(ssp)
    ]
    actual = apsp(adj)
    assert actual == expected

    # mssp 🤖
    for i in range(len(expected)):
        expected[i][1][i] = 0
    starts = range(adj.ncols)
    actual = mssp(adj, starts)
    assert actual == expected

    # sssp 👻
    start = random.choice(starts)
    actual = sssp(adj, start)
    assert actual == expected[start][1]
