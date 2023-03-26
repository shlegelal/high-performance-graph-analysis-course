import pytest

import utils
from project import triangle


@pytest.mark.parametrize(
    "f_and_source",
    [
        (triangle.count_vertex_triangles, "vertex_triangles"),
        (triangle.cohen_algorithm, "cohen_algorithm"),
        (triangle.sandia_algorithm, "sandia_algorithm"),
    ],
)
@pytest.mark.parametrize(
    "adj, res",
    map(
        lambda r: (utils.adj_from_res(r["adj"]), r),
        utils.load_res("test_triangle"),
    ),
    ids=utils.get_title("test_triangle"),
)
def test_triangles(adj, res, f_and_source: tuple):
    actual = f_and_source[0](adj)
    assert actual == res[f_and_source[1]]


@pytest.mark.parametrize(
    "f",
    [
        triangle.count_vertex_triangles,
        triangle.cohen_algorithm,
        triangle.sandia_algorithm,
    ],
)
@pytest.mark.parametrize(
    "adj",
    map(
        lambda r: (utils.adj_from_res(r["adj"])),
        utils.load_res("test_input"),
    ),
    ids=utils.get_title("test_input"),
)
def test_input(adj, f):
    with pytest.raises(ValueError):
        f(adj)
