"""Independent byte oracle for the no-GPU public ordered-Gram launch driver."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys


def _invoke(executable: str, inputs: str) -> dict[str, int]:
    result = subprocess.run(
        [executable],
        input=inputs,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if result.returncode:
        msg = f"admission driver exited {result.returncode}:\n{result.stdout}\n{result.stderr}"
        raise RuntimeError(msg)
    record = json.loads(result.stdout)
    assert record["fixed_rejections"] == 37
    assert record["initial_admissions"] == 3
    return record


def main() -> None:
    """Exercise actual public-wrapper admission using unbounded byte products."""
    executable = sys.argv[1]
    # Explicit EOF also works with an attached terminal or capture disabled.
    width = _invoke(executable, "")
    assert width["checked_boundaries"] == 0
    maximum_bytes, scalar_bytes = width["maximum_bytes"], width["scalar_bytes"]
    cases = []
    for batch in (1, 2, 65, 1048576, 2**31 - 1):
        for spins in (1, 2):
            for history in (0, 1, 2, 3, 32, 64):
                slots = max(1, history)
                boundary = math.isqrt(
                    maximum_bytes // (scalar_bytes * batch * slots * spins)
                )
                for n in sorted({1, boundary - 1, boundary, boundary + 1, 2**31 - 1}):
                    if not 1 <= n <= 2**31 - 1:
                        continue
                    history_bytes = batch * n * n * spins * slots * scalar_bytes
                    solve_bytes = batch * (history + 1) ** 2 * scalar_bytes
                    admitted = history_bytes <= maximum_bytes and (
                        history < 2 or solve_bytes <= maximum_bytes
                    )
                    cases.append(f"{batch} {n} {spins} {history} {int(admitted)}\n")
    checked = _invoke(executable, "".join(cases))
    assert checked["maximum_bytes"] == maximum_bytes
    assert checked["scalar_bytes"] == scalar_bytes
    assert checked["checked_boundaries"] == len(cases)
    print(
        f"37 rejections, 3 original admissions, poison/routing and {len(cases)} byte cases passed"
    )


if __name__ == "__main__":
    main()
