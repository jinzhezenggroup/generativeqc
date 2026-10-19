"""Same-PID pre-exec environment witness for the Linux CC campaign launcher."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from cc_tile_environment import (
    CHILD_SCHEMA,
    OBSERVATION_KEYS,
    SEMANTIC_KEYS,
    atomic_json,
    canonical_hash,
    execution_environment,
    validate_execution,
)


def main() -> None:
    """Verify the selected inherited values and exec the frozen native argv."""
    plan = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    environment = None
    try:
        actual = {
            key: os.environ[key]
            for key in (*SEMANTIC_KEYS, *OBSERVATION_KEYS)
            if key in os.environ
        }
        environment = execution_environment(actual)
        native_environment = validate_execution(environment, plan["profile_sha256"])
        if environment != plan["environment"]:
            raise ValueError("child inherited a different execution environment")
        atomic_json(
            output,
            {
                "schema": CHILD_SCHEMA,
                "status": "PREPARED_FOR_EXEC",
                "pid": os.getpid(),
                "argv": plan["argv"],
                "environment": environment,
                "environment_sha256": canonical_hash(environment),
            },
        )
        os.execve(plan["argv"][0], plan["argv"], native_environment)  # noqa: S606 -- frozen argv, no shell interpolation
    except BaseException as error:
        atomic_json(
            output,
            {
                "schema": CHILD_SCHEMA,
                "status": "INCOMPLETE",
                "environment": environment,
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise


if __name__ == "__main__":
    main()
