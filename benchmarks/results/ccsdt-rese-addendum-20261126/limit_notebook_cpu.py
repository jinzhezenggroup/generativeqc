"""Cap the detached aziridine session's CPU affinity across both sockets."""

import argparse
import itertools
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    sid = int((run / "pid.txt").read_text())
    if os.getsid(sid) != sid or Path(f"/proc/{sid}/cwd").resolve() != run:
        raise ValueError("the recorded PID is outside the expected notebook session")
    allowed = os.sched_getaffinity(sid)
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)
    sockets: dict[int, dict[int, int]] = {}
    for line in topology.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket = map(int, line.split(","))
        if cpu in allowed:
            sockets.setdefault(socket, {}).setdefault(core, cpu)
    selected = set(
        itertools.islice(
            (
                cpu
                for row in itertools.zip_longest(
                    *(cores.values() for cores in sockets.values())
                )
                for cpu in row
                if cpu is not None
            ),
            40,
        )
    )
    tasks = subprocess.check_output(
        ["ps", "-s", str(sid), "-L", "-o", "lwp="], text=True
    ).split()
    modified = 0
    for task in map(int, tasks):
        try:
            if os.getsid(task) == sid:
                os.sched_setaffinity(task, selected)
                modified += 1
        except ProcessLookupError:
            continue
    receipt = {
        "session": sid,
        "started_utc": datetime.now(UTC).isoformat(),
        "allowed_logical_cpus": sorted(selected),
        "tasks_updated": modified,
    }
    (run / "affinity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
