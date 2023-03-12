import inspect
import json
import pathlib
import pygraphblas as gb


def _get_filename() -> str:
    with pathlib.Path(inspect.stack()[2].filename) as f:
        parent = f.parent
        filename = f.stem
    return f"{parent}/res/{filename}.json"


def load_res(test_name: str) -> list[tuple]:
    with open(_get_filename()) as f:
        raw_res = json.load(f)
    return raw_res[test_name]


def get_title(test_name: str) -> list[str]:
    with open(_get_filename()) as f:
        raw_res = json.load(f)
    return [res["title"] for res in raw_res[test_name]]


def adj_from_res(res: list, t: gb.types.MetaType = gb.types.BOOL) -> gb.Matrix:
    size = len(res)
    if size == 0:
        return gb.Matrix.sparse(t, 0, 0)
    I, J = [], []
    for i in range(size):
        for j in range(len(res[0])):
            if res[i][j]:
                I.append(i)
                J.append(j)
    return gb.Matrix.from_lists(I, J, nrows=size, ncols=len(res[0]), typ=t)
