"""Continue disjoint residual endpoints on separate qz GPUs."""

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import run_gpu as native


def rank(output: Path, visible: str, tasks: list[dict[str, str]]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["GENERATIVEQC_STATIONARY_CACHE"] = str(output / "runtime-cache")
    os.environ["GENERATIVEQC_JIT_SCCACHE_ROOT"] = str(output / "jit-owners")
    output.mkdir()
    inputs = json.loads((native.ROOT / "inputs.json").read_text())["systems"]
    pairs = tuple(native.RECIPES.items())
    for dataset in ("HTBH38", "NHTBH38"):
        for recipe, sizes in pairs:
            subset = {
                task["system"]: inputs[task["system"]]
                for task in tasks
                if task["recipe"] == recipe and task["system"].startswith(dataset + "/")
            }
            if subset:
                native.RECIPES = {recipe: sizes}
                native.campaign(
                    dataset, subset, output / (dataset.lower() + "-" + recipe)
                )


def launch(output: Path, assignments: Path, cards: int) -> None:
    from cupy.cuda import runtime

    count = runtime.getDeviceCount()
    assert count == cards
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else list(map(str, range(count)))
    assert len(set(devices)) == count
    output.mkdir()
    (output / "waiting.txt").write_text(f"{count} visible devices\n")
    for _ in range(180):
        if assignments.is_file():
            break
        time.sleep(5)
    tasks = json.loads(assignments.read_text())
    flattened = [(task["system"], task["recipe"]) for shard in tasks for task in shard]
    assert len(tasks) == len(devices) and len(flattened) == len(set(flattened))
    (output / "assignments.json").write_bytes(assignments.read_bytes())
    context = get_context("spawn")
    workers = [
        context.Process(target=rank, args=(output / f"rank-{i}", gpu, shard))
        for i, (gpu, shard) in enumerate(zip(devices, tasks, strict=True))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert all(worker.exitcode == 0 for worker in workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("assignments", type=Path)
    parser.add_argument("cards", type=int)
    args = parser.parse_args()
    launch(args.output, args.assignments, args.cards)
