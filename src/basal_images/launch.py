from __future__ import annotations

import os
import sys
from collections.abc import Mapping

KNOBS = ("BASAL_TOKEN_BUDGET", "BASAL_GRAPH_LEN", "BASAL_TREE_ROWS")


def _int(env: Mapping[str, str], name: str) -> int | None:
    value = env.get(name, "").strip()
    if not value:
        return None
    number = int(value)
    if number < 1:
        raise SystemExit(f"{name} must be a positive integer")
    return number


def _cap(values: list[int], limit: int) -> list[int]:
    return [v for v in values if v <= limit] or [values[0]]


def wanted(env: Mapping[str, str]) -> bool:
    return any(env.get(name, "").strip() for name in KNOBS)


def tune(cls: type, env: Mapping[str, str]) -> None:
    budget = _int(env, "BASAL_TOKEN_BUDGET")
    length = _int(env, "BASAL_GRAPH_LEN")
    rows = _int(env, "BASAL_TREE_ROWS")
    if budget:
        cls.TOKEN_BUDGET = budget
    if length:
        cls.PLAIN_LENS = _cap(list(cls.PLAIN_LENS), length)
        cls.LENS = _cap(list(cls.LENS), length)
    if rows:
        cls.TREE_ROWS = _cap(list(cls.TREE_ROWS), rows)


def main() -> None:
    from basal.engine import GraphBackend
    from basal.server import main as serve

    tune(GraphBackend, os.environ)
    print(
        "graphs:",
        f"token budget {GraphBackend.TOKEN_BUDGET},",
        f"longest {GraphBackend.PLAIN_LENS[-1]},",
        f"tree rows {GraphBackend.TREE_ROWS[-1]}",
        flush=True,
    )
    sys.argv[0] = "basal-serve"
    serve()


if __name__ == "__main__":
    main()
