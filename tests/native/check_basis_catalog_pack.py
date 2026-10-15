"""Compare executed native catalog records with the independent decimal pack."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Keep full semantic input coverage without pinning generator/body spelling."""
    executable, pack_path = sys.argv[1:]
    pack = json.loads(Path(pack_path).read_text())["bases"]
    completed = subprocess.run(
        [executable, "--dump"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual = {}
    for line in completed.stdout.splitlines():
        row = json.loads(line)
        key = (row["basis"], row["atomic_number"])
        if key in actual:
            raise AssertionError(f"duplicate native catalog record: {key}")
        actual[key] = row
    expected = {
        (name, int(number))
        for name, basis in pack.items()
        for number in basis["elements"]
    }
    if actual.keys() != expected:
        raise AssertionError("native catalog element coverage differs from the pack")
    primitive_count = 0
    for key, record in actual.items():
        expected_shells, expected_primitives = [], []
        for shell in pack[key[0]]["elements"][str(key[1])]:
            expected_shells.append(
                [
                    shell["angular_momentum"],
                    len(expected_primitives),
                    len(shell["exponents"]),
                ]
            )
            expected_primitives.extend(
                [float(exponent), float(coefficient)]
                for exponent, coefficient in zip(
                    shell["exponents"], shell["coefficients"], strict=True
                )
            )
        if record["shells"] != expected_shells:
            raise AssertionError(f"native shell order/ownership differs: {key}")
        if record["primitives"] != expected_primitives:
            raise AssertionError(f"native decimal primitives differ: {key}")
        primitive_count += len(expected_primitives)
    print(f"All {len(actual)} catalog records /{primitive_count} primitives match.")


if __name__ == "__main__":
    main()
