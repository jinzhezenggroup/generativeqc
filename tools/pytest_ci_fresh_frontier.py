"""Seed fresh CI qualifications across existing worksteal worker frontiers.

The shell selector owns file selection and explicitly opts in fresh,
function-scoped suites. This plugin changes only collection order: no case,
fixture, worker or timeout is removed, shared or replaced.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest


def _frontier_indices(priority: Sequence[bool], workers: int) -> list[int]:
    """Preserve every item while frontloading fresh cases in each worker queue."""
    if workers < 1:
        raise ValueError("CI fresh frontiers require a positive worker count")
    count = len(priority)
    if not count or workers == 1 or not any(priority):
        return list(range(count))

    workers = min(workers, count)
    capacities: list[int] = []
    remaining = count
    for worker in range(workers):
        capacity = remaining // (workers - worker)
        capacities.append(capacity)
        remaining -= capacity

    # Worksteal initially divides the complete collection into contiguous
    # queues, with remainder items assigned to the later workers. A single
    # frontloaded file would still concentrate its fresh work on one worker.
    groups: list[list[int]] = [[] for _ in capacities]
    cursor = 0
    for index, selected in enumerate(priority):
        if not selected:
            continue
        while len(groups[cursor]) == capacities[cursor]:
            cursor = (cursor + 1) % workers
        groups[cursor].append(index)
        cursor = (cursor + 1) % workers

    ordinary = (index for index, selected in enumerate(priority) if not selected)
    for group, capacity in zip(groups, capacities, strict=True):
        group.extend(next(ordinary) for _ in range(capacity - len(group)))
    return [index for group in groups for index in group]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit, non-filtering metadata supplied by the CI selector."""
    parser.getgroup("generativeqc-ci").addoption(
        "--ci-fresh-priority",
        dest="ci_fresh_priority",
        action="append",
        default=[],
        metavar="TEST_PATH",
        help="Start this fresh suite early on each existing worksteal worker",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Validate the controller before xdist resets each worker's local options."""
    if not config.getoption("ci_fresh_priority") or hasattr(config, "workerinput"):
        return
    if config.getoption("numprocesses", default=None) and (
        config.getoption("dist", default="no") != "worksteal"
    ):
        raise pytest.UsageError("CI fresh frontiers require --dist=worksteal")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Reorder the final selected collection identically on every worker.

    Run after ordinary marker/deselection hooks, without changing their
    selection. Serial execution remains unchanged; parallel use is explicit
    and restricted to the worksteal startup contract qualified by real tests.
    """
    selected = config.getoption("ci_fresh_priority")
    if not selected:
        return
    worker_input = getattr(config, "workerinput", {})
    workers = worker_input.get(
        "workercount", config.getoption("numprocesses", default=1) or 1
    )
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise pytest.UsageError("CI fresh frontiers require a known worker count")
    if workers == 1:
        return
    invocation = config.invocation_params.dir
    paths = {(invocation / Path(path)).resolve() for path in selected}
    priority = [item.path.resolve() in paths for item in items]
    order = _frontier_indices(priority, workers)
    items[:] = [items[index] for index in order]
