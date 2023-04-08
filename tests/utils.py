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
    I, J, V = [], [], []
    for i in range(size):
        for j in range(len(res[0])):
            val = res[i][j]
            if (val and t == gb.BOOL) or (val is not None and t == gb.types.FP64):
                I.append(i)
                J.append(j)
                V.append(val)
    return gb.Matrix.from_lists(I, J, V, nrows=size, ncols=len(res[0]), typ=t)
