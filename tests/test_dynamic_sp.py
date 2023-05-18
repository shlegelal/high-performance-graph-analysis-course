import networkx as nx
import pytest

from project.dynamic_sp import DynamicSSSP
import utils


@pytest.mark.parametrize(
    "graph, start, update_chunks, expected_list",
    map(
        lambda r: (
            nx.node_link_graph(r["graph"], directed=True, multigraph=False),
            r["start"],
            r["updates"],
            [{n: float(dist) for n, dist in exp.items()} for exp in r["expected_list"]],
        ),
        utils.load_res("test_dynamic_sssp"),
    ),
    ids=utils.get_title("test_dynamic_sssp"),
)
def test_dynamic_sssp(
    graph: nx.DiGraph, start: str, update_chunks: list, expected_list: list
):
    algo = DynamicSSSP(graph, start)

    actual = algo.query()
    assert actual == expected_list[0]

    for updates, expected in zip(update_chunks, expected_list[1:]):
        for update in updates:
            if update[0] == "inc":
                algo.increment(*update[1])
            if update[0] == "dec":
                algo.decrement(*update[1])
        actual = algo.query()
        assert actual == expected
