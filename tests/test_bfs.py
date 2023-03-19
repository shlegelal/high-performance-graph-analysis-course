import pytest
import utils
import pygraphblas as gb

from project.bfs import bfs, msbfs


@pytest.mark.parametrize(
    "adj, start, expected",
    map(
        lambda r: (utils.adj_from_res(r["adj"]), r["start"], r["expected"]),
        utils.load_res("test_bfs_algorithm"),
    ),
    ids=utils.get_title("test_bfs_algorithm"),
)
def test_bfs_algorithm(adj: gb.Matrix, start: int, expected: list[int]):
    actual = bfs(adj, start)
    assert actual == expected


@pytest.mark.parametrize(
    "adj, starts, expected",
    map(
        lambda r: (
            utils.adj_from_res(r["adj"]),
            r["starts"],
            [tuple(pair_list) for pair_list in r["expected"]],
        ),
        utils.load_res("test_msbfs"),
    ),
    ids=utils.get_title("test_msbfs"),
)
def test_msbfs(adj: gb.Matrix, starts: list[int], expected: list[int]):
    actual = msbfs(adj, starts)

    assert actual == expected


@pytest.mark.parametrize(
    "adj, start",
    map(
        lambda r: (
            utils.adj_from_res(r["adj"], gb.types.BOOL if r["type"] else gb.INT64),
            r["start"],
        ),
        utils.load_res("test_input"),
    ),
    ids=utils.get_title("test_input"),
)
def test_input(adj: gb.Matrix, start: int):
    with pytest.raises(ValueError):
        bfs(adj, start)
        msbfs(adj, [start])
