import pytest
import utils
import pygraphblas as gb

from project.bfs import bfs


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
    "adj, start",
    map(
        lambda r: (
            utils.adj_from_res(r["adj"], gb.types.BOOL if r["type"] else gb.INT64),
            r["start"],
        ),
        utils.load_res("test_bfs_input"),
    ),
    ids=utils.get_title("test_bfs_input"),
)
def test_bfs_input(adj: gb.Matrix, start: int):
    with pytest.raises(ValueError):
        bfs(adj, start)
